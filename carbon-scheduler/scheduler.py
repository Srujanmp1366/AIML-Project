"""Pure scheduling policies for hourly carbon-aware job placement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Job:
    """A job's remaining work and latest acceptable completion time."""

    job_id: str
    gpus: int
    hours_left: float
    deadline: pd.Timestamp
    deferrable: bool
    has_started: bool = False
    ran_last_hour: bool = False


Decision = Callable[[pd.Timestamp, list[Job], np.ndarray, int | None], list[str]]


def slack(job: Job, now: pd.Timestamp) -> float:
    """Return deadline slack in hours."""
    return (job.deadline - now).total_seconds() / 3600.0 - job.hours_left


def _fitting(jobs: list[Job], free_gpus: int | None) -> list[str]:
    selected: list[str] = []
    remaining = free_gpus
    for job in jobs:
        if remaining is None or job.gpus <= remaining:
            selected.append(job.job_id)
            if remaining is not None:
                remaining -= job.gpus
    return selected


def run_now(
    now: pd.Timestamp,
    waiting_jobs: list[Job],
    forecast: np.ndarray,
    free_gpus: int | None,
) -> list[str]:
    """Run every waiting job that fits, preserving queue order."""
    del now, forecast
    return _fitting(waiting_jobs, free_gpus)


def make_threshold(trailing_median: float | pd.Series) -> Decision:
    """Build a threshold policy from a past-only median or hourly median series."""

    def decide(
        now: pd.Timestamp,
        waiting_jobs: list[Job],
        forecast: np.ndarray,
        free_gpus: int | None,
    ) -> list[str]:
        if isinstance(trailing_median, pd.Series):
            median = trailing_median.get(now.floor("h"), float("nan"))
        else:
            median = trailing_median
        must_run = [job for job in waiting_jobs if not job.deferrable or slack(job, now) <= 0]
        wants = [job for job in waiting_jobs if job not in must_run and len(forecast) > 0 and forecast[0] < median]
        must_run.sort(key=lambda job: slack(job, now))
        return _fitting(must_run + wants, free_gpus)

    return decide


def _wants_to_run(job: Job, now: pd.Timestamp, forecast: np.ndarray) -> bool:
    if slack(job, now) <= 0:
        return True
    current_hour = now.floor("h")
    hours_to_deadline = max(
        0,
        math.ceil((job.deadline - current_hour).total_seconds() / 3600.0),
    )
    horizon = min(24, hours_to_deadline)
    if job.hours_left >= horizon:
        return True
    if horizon <= 0:
        return True
    count = math.ceil(job.hours_left)
    best_hours = np.argsort(forecast[:horizon], kind="stable")[:count]
    return bool(np.any(best_hours == 0))


def forecast_greedy(
    now: pd.Timestamp,
    waiting_jobs: list[Job],
    forecast: np.ndarray,
    free_gpus: int | None,
) -> list[str]:
    """Run urgent jobs, then jobs whose cheapest forecast hours include now."""
    must_run = [job for job in waiting_jobs if slack(job, now) <= 0]
    must_ids = {job.job_id for job in must_run}
    wants = [
        job
        for job in waiting_jobs
        if job.job_id not in must_ids and _wants_to_run(job, now, forecast)
    ]
    indexed = {id(job): index for index, job in enumerate(waiting_jobs)}
    must_run.sort(key=lambda job: (slack(job, now), indexed[id(job)]))
    wants.sort(key=lambda job: (slack(job, now), indexed[id(job)]))
    return _fitting(must_run + wants, free_gpus)


def _block_cost(job: Job, forecast: np.ndarray, first_hour: int, resume_minutes: float) -> float:
    """Predicted CI-weighted work for a contiguous execution block."""
    remaining = job.hours_left
    cost = 0.0
    for offset in range(math.ceil(remaining)):
        run_hours = min(1.0, remaining)
        remaining -= run_hours
        resume = resume_minutes / 60.0 if offset == 0 and job.has_started and (
            first_hour > 0 or not job.ran_last_hour
        ) else 0.0
        cost += (run_hours + resume) * float(forecast[first_hour + offset])
    return cost


def make_guarded_greedy(error_margin_ci: float = 0.0, resume_minutes: float = 5.0) -> Decision:
    """Prefer feasible continuous runs; require a clear forecast benefit to wait.

    ``error_margin_ci`` is a per-hour forecast-error scale in gCO2/kWh,
    estimated from validation data. It is a heuristic margin, not a guarantee.
    """
    if (not math.isfinite(error_margin_ci) or error_margin_ci < 0
            or not math.isfinite(resume_minutes) or resume_minutes < 0):
        raise ValueError("forecast error margin and resume minutes must be finite and nonnegative")

    def decide(now: pd.Timestamp, waiting_jobs: list[Job], forecast: np.ndarray,
               free_gpus: int | None) -> list[str]:
        urgent: list[Job] = []
        ready: list[Job] = []
        for job in waiting_jobs:
            if not job.deferrable:
                urgent.append(job)
                continue
            remaining_slots = math.ceil(job.hours_left)
            wait_hours = min(math.floor(slack(job, now)), len(forecast) - remaining_slots)
            if wait_hours <= 0:
                urgent.append(job)
                continue
            cost_now = _block_cost(job, forecast, 0, resume_minutes)
            best_later = min(_block_cost(job, forecast, hour, resume_minutes)
                             for hour in range(1, wait_hours + 1))
            uncertainty = 2.0 * error_margin_ci * job.hours_left
            if cost_now <= best_later + uncertainty:
                ready.append(job)
        order = {id(job): index for index, job in enumerate(waiting_jobs)}
        urgent.sort(key=lambda job: (slack(job, now) if job.deferrable else float("-inf"), order[id(job)]))
        ready.sort(key=lambda job: (slack(job, now), order[id(job)]))
        return _fitting(urgent + ready, free_gpus)

    return decide


# The oracle differs only in which CI series the simulator supplies.
oracle = forecast_greedy

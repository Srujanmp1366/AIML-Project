"""Hourly trace-driven simulator for carbon-aware ML training jobs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict, cast
import warnings

import numpy as np
import pandas as pd

from scheduler import Decision, Job, forecast_greedy, make_guarded_greedy, make_threshold, run_now


GPUS = 64
KW_PER_GPU = 0.4
PUE = 1.5
RESUME_MIN = 5
K_VALUES = [6, 12, 24, 48]
ISSUE_LAG_HOURS = 1
HOUR = pd.Timedelta(hours=1)
DATA_DIR = Path(__file__).resolve().parent / "data"


class _JobRecord(TypedDict):
    job: Job
    duration: float
    arrival: pd.Timestamp
    join: pd.Timestamp
    order: int
    started: bool
    previous_running: bool
    finished: bool
    start_time: pd.Timestamp | None
    finish_time: pd.Timestamp | None
    delay_hours: float | None
    deadline_missed: bool
    busy_gpu_hours: float


def _truth_frame(grid_truth: pd.DataFrame) -> pd.DataFrame:
    truth = grid_truth.copy()
    truth["time"] = pd.to_datetime(truth["time"], utc=True)
    truth["actual_ci"] = pd.to_numeric(truth["actual_ci"], errors="coerce")
    truth["mef"] = pd.to_numeric(truth["mef"], errors="coerce") if "mef" in truth else np.nan
    if truth["time"].duplicated().any():
        raise ValueError("grid_truth.time must be unique")
    return truth.sort_values("time").reset_index(drop=True)


def _forecast_frame(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = forecasts.copy()
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame["predicted_ci"] = pd.to_numeric(frame["predicted_ci"], errors="coerce")
    return frame


def _forecast_at(now: pd.Timestamp, forecasts: pd.DataFrame, truth: pd.DataFrame, source: str) -> np.ndarray:
    if source not in {"forecast", "actual"}:
        raise ValueError("source must be 'forecast' or 'actual'")
    targets = pd.date_range(now, periods=24, freq="h")
    if source == "actual":
        actual = truth.set_index("time")["actual_ci"].reindex(targets)
        assert len(actual) == 24 and actual.notna().all(), "oracle needs 24 actual CI hours"
        return actual.to_numpy(dtype=float)
    cutoff = now - ISSUE_LAG_HOURS * HOUR
    rows = forecasts.loc[forecasts["issue_time"] == cutoff].sort_values("target_time")
    assert len(rows) == 24, f"expected 24 forecast rows issued at {cutoff}"
    assert pd.DatetimeIndex(rows["target_time"]).equals(targets), f"wrong forecast targets at {cutoff}"
    assert (rows["issue_time"] <= cutoff).all(), "forecast leakage detected"
    values = rows["predicted_ci"].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "forecast contains missing or non-finite values"
    return values


def forecast_at(now: pd.Timestamp, forecasts: pd.DataFrame, grid_truth: pd.DataFrame, source: str = "forecast") -> np.ndarray:
    """Return the eligible 24-hour forecast, or actual CI for the oracle."""
    current = pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return _forecast_at(current.tz_convert("UTC"), _forecast_frame(forecasts), _truth_frame(grid_truth), source)


def make_naive_forecast(grid_truth: pd.DataFrame) -> Path:
    """Write a 24-hour naive forecast, using CI from 24 hours before each target."""
    truth = _truth_frame(grid_truth).set_index("time")
    values = truth["actual_ci"]
    rows: list[dict[str, object]] = []
    for now in truth.index:
        targets = pd.date_range(now, periods=24, freq="h")
        lagged = targets - pd.Timedelta(hours=24)
        if not targets.isin(truth.index).all() or not lagged.isin(truth.index).all(): continue
        predictions = values.reindex(lagged).to_numpy(dtype=float)
        if not np.isfinite(predictions).all(): continue
        issue = now - HOUR
        rows.extend({"issue_time": issue, "target_time": t, "predicted_ci": p} for t, p in zip(targets, predictions))
    output = DATA_DIR / "forecasts_naive24.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["issue_time", "target_time", "predicted_ci"]).to_csv(output, index=False)
    return output


def _job_rows(jobs: pd.DataFrame, k: int) -> list[_JobRecord]:
    frame = jobs.copy()
    frame["arrival"] = pd.to_datetime(frame["arrival"], utc=True)
    frame["_order"] = np.arange(len(frame))
    if frame["job_id"].duplicated().any(): raise ValueError("jobs.job_id must be unique")
    result: list[_JobRecord] = []
    for row in frame.sort_values(["arrival", "_order"]).to_dict("records"):
        deferrable = str(row["deferrable"]).strip().lower() in {"true", "1", "yes"}
        arrival = pd.Timestamp(row["arrival"])
        duration, gpu_count = float(row["duration_hours"]), math.ceil(float(row["gpus"]))
        if duration <= 0 or gpu_count <= 0: raise ValueError("job durations and GPU counts must be positive")
        deadline = arrival + pd.Timedelta(hours=k) if deferrable else arrival
        result.append({"job": Job(str(row["job_id"]), gpu_count, duration, deadline, deferrable),
                       "duration": duration, "arrival": arrival, "join": arrival,
                       "order": int(row["_order"]),
                       "started": False, "previous_running": False, "finished": False,
                       "start_time": None, "finish_time": None, "delay_hours": None,
                       "deadline_missed": False, "busy_gpu_hours": 0.0})


    return result


def _median_series(truth: pd.DataFrame) -> pd.Series:
    return truth.set_index("time")["actual_ci"].sort_index().rolling("30D", closed="left").median()


def _hours_to_simulate(forecasts: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[pd.Timestamp]:
    hours = pd.DatetimeIndex(forecasts["issue_time"].drop_duplicates() + HOUR).sort_values()
    if start is not None:
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        hours = hours[hours >= start]
    if end is not None:
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        hours = hours[hours < end]
    if len(hours) == 0: raise ValueError("no forecast hours in the requested period")
    if not hours.equals(pd.date_range(hours[0], hours[-1], freq="h")):
        raise ValueError("forecast issue times must cover every simulated hour")
    return list(hours)


def _simulate(policy: Decision, k: int, gpus: int | None, jobs: pd.DataFrame, truth: pd.DataFrame,
              forecasts: pd.DataFrame, hours: list[pd.Timestamp], admission_end: pd.Timestamp, source: str,
              warn_marginal: bool) -> tuple[dict[str, float], pd.DataFrame]:
    records = _job_rows(jobs, k)
    period_end = hours[-1] + HOUR
    records = [record for record in records if record["arrival"] < admission_end]
    capacity = gpus
    truth_values = truth.set_index("time")
    actual_ci_by_hour = truth_values["actual_ci"].to_dict()
    mef_by_hour = truth_values["mef"].to_dict()
    forecast_groups = {
        issue_time: group.sort_values("target_time").reset_index(drop=True)
        for issue_time, group in forecasts.groupby("issue_time", sort=False)
    }
    forecast_cache: dict[pd.Timestamp, np.ndarray] = {}
    if not pd.DatetimeIndex(hours).isin(truth_values.index).all(): raise ValueError("simulated hours must exist in grid_truth")
    if truth_values["actual_ci"].reindex(hours).isna().any(): raise ValueError("actual_ci is required for every simulated hour")
    available_mef = truth_values["mef"].reindex(hours).notna().all()
    if not available_mef and warn_marginal:
        warnings.warn("MEF missing in test period; marginal scoring is skipped", stacklevel=2)
    total_kwh = avg_kg = marg_kg = busy_gpu_hours = 0.0
    max_hourly_gpus = 0

    now = hours[0]
    next_arrival_index = 0
    active_records: list[_JobRecord] = []
    while now < period_end:
        while (next_arrival_index < len(records)
               and records[next_arrival_index]["arrival"] <= now):
            active_records.append(records[next_arrival_index])
            next_arrival_index += 1
        active = active_records
        inference = sorted((r for r in active if not r["job"].deferrable), key=lambda r: (r["arrival"], r["order"]))
        inference_ids: list[str] = []
        remaining = capacity
        for record in inference:
            job = record["job"]
            if remaining is None or job.gpus <= remaining:
                inference_ids.append(job.job_id)
                if remaining is not None:
                    remaining -= job.gpus
        free = remaining
        hour = now.floor("h")
        if hour not in forecast_cache:
            targets = pd.date_range(hour, periods=24, freq="h")
            if source == "actual":
                actual = truth_values["actual_ci"].reindex(targets)
                assert len(actual) == 24 and actual.notna().all(), "oracle needs 24 actual CI hours"
                forecast_cache[hour] = actual.to_numpy(dtype=float)
            elif source == "forecast":
                cutoff = hour - ISSUE_LAG_HOURS * HOUR
                rows = forecast_groups.get(cutoff)
                assert rows is not None and len(rows) == 24, f"expected 24 forecast rows issued at {cutoff}"
                assert pd.DatetimeIndex(rows["target_time"]).equals(targets), f"wrong forecast targets at {cutoff}"
                assert (rows["issue_time"] <= cutoff).all(), "forecast leakage detected"
                values = rows["predicted_ci"].to_numpy(dtype=float)
                assert np.isfinite(values).all(), "forecast contains missing or non-finite values"
                forecast_cache[hour] = values
            else:
                raise ValueError("source must be 'forecast' or 'actual'")
        forecast = forecast_cache[hour]
        training: list[Job] = []
        for record in active:
            if record["job"].deferrable:
                record["job"].has_started = record["started"]
                record["job"].ran_last_hour = record["previous_running"]
                training.append(record["job"])
        training_ids = policy(now, training, forecast, free)
        training_map = {job.job_id: job for job in training}
        if len(training_ids) != len(set(training_ids)) or any(j not in training_map for j in training_ids):
            raise ValueError("policy returned duplicate or non-waiting job id")
        requested = sum(training_map[job_id].gpus for job_id in training_ids)
        if free is not None and requested > free:
            raise ValueError("policy exceeded the free GPU capacity")
        selected_ids = set(inference_ids + training_ids)
        scheduled_gpus = sum(r["job"].gpus for r in active if r["job"].job_id in selected_ids)
        max_hourly_gpus = max(max_hourly_gpus, scheduled_gpus)
        actual_ci = float(actual_ci_by_hour[hour])
        mef_value = mef_by_hour[hour]
        mef = float(mef_value) if pd.notna(mef_value) else float("nan")

        next_event = min(period_end, hour + HOUR)
        if next_arrival_index < len(records):
            next_event = min(next_event, records[next_arrival_index]["arrival"])
        for record in active:
            job = record["job"]
            if job.job_id in selected_ids:
                finish = now + pd.Timedelta(hours=job.hours_left)
                next_event = min(next_event, finish)

        if next_event <= now:
            raise RuntimeError("simulation event queue did not advance")
        segment_hours = (next_event - now).total_seconds() / 3600.0
        for record in active:
            job = record["job"]
            if job.job_id not in selected_ids:
                record["previous_running"] = False
                continue
            if record["start_time"] is None:
                record["start_time"] = now
            resume = RESUME_MIN / 60.0 if record["started"] and not record["previous_running"] else 0.0
            run_hours = min(segment_hours, job.hours_left)
            job.hours_left = max(0.0, job.hours_left - run_hours)
            energy = job.gpus * KW_PER_GPU * PUE * (run_hours + resume)
            total_kwh += energy
            avg_kg += energy * actual_ci / 1000.0
            if available_mef:
                marg_kg += energy * mef / 1000.0
            record["busy_gpu_hours"] += job.gpus * run_hours
            busy_gpu_hours += job.gpus * run_hours
            record["started"] = True
            record["previous_running"] = True
            if job.hours_left <= 1e-9:
                finish_time = now + pd.Timedelta(hours=run_hours)
                record["finish_time"] = finish_time
                record["delay_hours"] = (
                    finish_time - (record["arrival"] + pd.Timedelta(hours=record["duration"]))
                ).total_seconds() / 3600.0
                record["deadline_missed"] = bool(job.deferrable and finish_time > job.deadline)
                record["previous_running"] = False
                record["finished"] = True
        active_records = [record for record in active_records if not record["finished"]]
        now = next_event

    deferrals = [record for record in records if record["job"].deferrable]
    delays: list[float] = []
    for record in records:
        delay = record["delay_hours"]
        if delay is not None:
            delays.append(delay)
    def missed(record: _JobRecord) -> bool:
        return record["deadline_missed"] or (
            record["finish_time"] is None and record["job"].deadline <= period_end
        )
    def observed(record: _JobRecord) -> bool:
        return record["finish_time"] is not None or record["job"].deadline <= period_end
    def infeasible(record: _JobRecord) -> bool:
        return record["job"].deferrable and (
            record["join"] + pd.Timedelta(hours=record["duration"]) > record["job"].deadline
            or (capacity is not None and record["job"].gpus > capacity)
        )
    def start_wait_hours(record: _JobRecord) -> float:
        started = record["start_time"]
        return (started - record["arrival"]).total_seconds() / 3600.0 if started is not None else float("nan")
    observed_deadlines = sum(observed(record) for record in deferrals)
    deadline_misses = sum(missed(record) for record in deferrals)
    completed_jobs = sum(record["finish_time"] is not None for record in records)
    denominator = capacity * len(hours) if capacity is not None else float("nan")
    metrics = {
        "total_kwh": total_kwh,
        "emissions_avg_kg": avg_kg,
        "emissions_marg_kg": marg_kg if available_mef else float("nan"),
        "total_jobs": len(records),
        "completed_jobs": completed_jobs,
        "unfinished_jobs": len(records) - completed_jobs,
        "infeasible_on_arrival": sum(infeasible(record) for record in deferrals),
        "deadline_misses": deadline_misses,
        "deadline_observed_jobs": observed_deadlines,
        "deadline_miss_rate": deadline_misses / len(deferrals)
        if deferrals and observed_deadlines == len(deferrals) else float("nan"),
        "mean_added_delay_hours": float(np.mean(delays)) if delays else float("nan"),
        "p95_added_delay_hours": float(np.percentile(delays, 95)) if delays else float("nan"),
        "gpu_utilisation": busy_gpu_hours / denominator if denominator else float("nan"),
        "max_hourly_gpus": max_hourly_gpus,
    }
    per_job = pd.DataFrame([{"job_id": r["job"].job_id, "arrival": r["arrival"], "start_time": r["start_time"],
        "finish_time": r["finish_time"], "deadline": r["job"].deadline if r["job"].deferrable else pd.NaT,
        "deferrable": r["job"].deferrable,
        "deadline_observed": observed(r) if r["job"].deferrable else False,
        "deadline_missed": missed(r) if r["job"].deferrable else False,
        "infeasible_on_arrival": infeasible(r),
        "start_wait_hours": start_wait_hours(r),
        "added_delay_hours": r["delay_hours"], "busy_gpu_hours": r["busy_gpu_hours"]} for r in records])
    return metrics, per_job


def run_sim(policy: str | Decision, k: int, gpus: int | None, jobs: pd.DataFrame,
            grid_truth: pd.DataFrame, forecasts: pd.DataFrame,
            start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
            admission_end: pd.Timestamp | None = None,
            forecast_error_margin_ci: float = 0.0) -> tuple[dict[str, float], pd.DataFrame]:
    """Run one policy; admit jobs before ``admission_end`` and simulate to ``end``.

    Carbon comparisons are undefined when either compared run leaves work unfinished.
    An earlier admission cutoff permits a common completion buffer for all policies.
    The legacy oracle is perfect-CI forecast_greedy, not an optimal schedule.
    """
    if k <= 0 or (gpus is not None and gpus <= 0): raise ValueError("k and finite capacity must be positive")
    truth, forecast_rows = _truth_frame(grid_truth), _forecast_frame(forecasts)
    hours = _hours_to_simulate(forecast_rows, start, end)
    if admission_end is None:
        admission_end = hours[-1] + HOUR
    else:
        admission_end = pd.Timestamp(admission_end)
        if admission_end.tzinfo is None:
            raise ValueError("admission_end must be timezone-aware")
        admission_end = admission_end.tz_convert("UTC")
        if not hours[0] < admission_end <= hours[-1] + HOUR:
            raise ValueError("admission_end must fall within the simulation period")
    source = "forecast"
    if isinstance(policy, str):
        names: dict[str, Decision] = {"run_now": run_now, "forecast_greedy": forecast_greedy}
        if policy == "threshold":
            decision = make_threshold(_median_series(truth))
        elif policy == "oracle":
            decision, source = forecast_greedy, "actual"
        elif policy == "forecast_guarded":
            decision = make_guarded_greedy(forecast_error_margin_ci, RESUME_MIN)
        elif policy in names:
            decision = names[policy]
        else:
            raise ValueError(f"unknown policy: {policy}")
    else:
        decision = policy
    metrics, per_job = _simulate(decision, k, gpus, jobs, truth, forecast_rows, hours, admission_end, source, True)
    baseline = metrics if policy == "run_now" or policy is run_now else _simulate(run_now, k, gpus, jobs, truth, forecast_rows, hours, admission_end, "forecast", False)[0]
    oracle_metrics = metrics if policy == "oracle" else _simulate(forecast_greedy, k, gpus, jobs, truth, forecast_rows, hours, admission_end, "actual", False)[0]
    guarded_perfect = (_simulate(make_guarded_greedy(0.0, RESUME_MIN), k, gpus, jobs, truth,
                                forecast_rows, hours, admission_end, "actual", False)[0]
                       if policy == "forecast_guarded" else None)
    for kind, key in (("avg", "emissions_avg_kg"), ("marg", "emissions_marg_kg")):
        base, value = baseline[key], metrics[key]
        metrics[f"run_now_{kind}_kg"] = base
        comparable = metrics["unfinished_jobs"] == baseline["unfinished_jobs"] == 0
        metrics[f"percent_saved_vs_run_now_{kind}_pct"] = 100.0 * (base - value) / base if comparable and np.isfinite(base) and base != 0 else float("nan")
        oracle_value = oracle_metrics[key]
        metrics[f"greedy_perfect_ci_{kind}_kg"] = oracle_value
        comparable_oracle = metrics["unfinished_jobs"] == oracle_metrics["unfinished_jobs"] == 0
        metrics[f"gap_to_oracle_{kind}_kg"] = value - oracle_value if comparable_oracle and np.isfinite(value) and np.isfinite(oracle_value) else float("nan")
        if guarded_perfect is not None:
            perfect_value = guarded_perfect[key]
            comparable_guarded = metrics["unfinished_jobs"] == guarded_perfect["unfinished_jobs"] == 0
            metrics[f"guarded_perfect_ci_{kind}_kg"] = perfect_value
            metrics[f"gap_to_guarded_perfect_ci_{kind}_kg"] = (
                value - perfect_value if comparable_guarded and np.isfinite(value)
                and np.isfinite(perfect_value) else float("nan")
            )
    return metrics, per_job

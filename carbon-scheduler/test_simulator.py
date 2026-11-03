"""Inline trace fixtures for simulator behavior."""

import numpy as np
import pandas as pd
import pytest

from scheduler import Job
from simulator import HOUR, RESUME_MIN, forecast_at, run_sim


def _frames(sim_hours: int = 4, ci: float = 500.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    times = pd.date_range("2025-01-01T00:00:00Z", periods=48, freq="h")
    truth = pd.DataFrame({"time": times, "actual_ci": ci, "mef": ci + 20})
    forecasts = pd.DataFrame([
        {"issue_time": now - HOUR, "target_time": target, "predicted_ci": ci}
        for now in times[:sim_hours]
        for target in pd.date_range(now, periods=24, freq="h")
    ])
    return truth, forecasts, times


def _jobs(*rows: tuple[str, pd.Timestamp, float, int, bool]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["job_id", "arrival", "duration_hours", "gpus", "deferrable"])


def test_capacity_is_never_exceeded() -> None:
    truth, forecasts, times = _frames(3)
    jobs = _jobs(
        ("a", times[0], 2.0, 2, False),
        ("b", times[0], 1.0, 1, False),
        ("c", times[1], 1.0, 1, True),
    )
    metrics, _ = run_sim("run_now", 6, 2, jobs, truth, forecasts, times[0], times[3])
    assert metrics["max_hourly_gpus"] <= 2


def test_mid_hour_finish_uses_partial_hour_energy() -> None:
    truth, forecasts, times = _frames(1)
    jobs = _jobs(("half", times[0], 0.5, 1, False))
    metrics, per_job = run_sim("run_now", 6, 2, jobs, truth, forecasts, times[0], times[1])
    assert metrics["total_kwh"] == pytest.approx(1 * 0.4 * 1.5 * 0.5)
    assert per_job.loc[0, "finish_time"] == times[0] + pd.Timedelta(minutes=30)


def test_resume_adds_exactly_five_minutes_of_energy() -> None:
    truth, forecasts, times = _frames(3)
    jobs = _jobs(("resume", times[0], 2.0, 1, True))

    def pause_one_hour(now: pd.Timestamp, waiting: list[Job], forecast: np.ndarray, free: int | None) -> list[str]:
        del forecast, free
        return [job.job_id for job in waiting] if now != times[1] else []

    metrics, _ = run_sim(pause_one_hour, 6, 1, jobs, truth, forecasts, times[0], times[3])
    expected = 0.4 * 1.5 * (2.0 + RESUME_MIN / 60.0)
    assert metrics["total_kwh"] == pytest.approx(expected)


def test_forecast_rows_do_not_leak_from_later_issue_time() -> None:
    truth, forecasts, times = _frames(1)
    later = pd.DataFrame([
        {"issue_time": times[0], "target_time": target, "predicted_ci": -1.0}
        for target in pd.date_range(times[0], periods=24, freq="h")
    ])
    observed = forecast_at(times[0], pd.concat([forecasts, later]), truth)
    assert np.all(observed == 500.0)
    used = forecasts.loc[forecasts["issue_time"] == times[0] - HOUR, "issue_time"]
    assert (used <= times[0] - HOUR).all()


def test_flat_ci_gives_equal_emissions_for_all_policies() -> None:
    truth, forecasts, times = _frames(4, 500.0)
    jobs = _jobs(("train", times[0], 1.0, 1, True))
    results = [
        run_sim(name, 3, 2, jobs, truth, forecasts, times[0], times[4])[0]
        for name in ("run_now", "threshold", "forecast_greedy", "oracle")
    ]
    assert all(result["emissions_avg_kg"] == pytest.approx(results[0]["emissions_avg_kg"]) for result in results)
    assert all(result["emissions_marg_kg"] == pytest.approx(results[0]["emissions_marg_kg"]) for result in results)


def test_run_twice_is_deterministic() -> None:
    truth, forecasts, times = _frames(3)
    jobs = _jobs(("a", times[0], 1.5, 1, True))
    first = run_sim("forecast_greedy", 6, 2, jobs, truth, forecasts, times[0], times[3])
    second = run_sim("forecast_greedy", 6, 2, jobs, truth, forecasts, times[0], times[3])
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])

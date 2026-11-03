"""Inline tests for the scheduler policies."""

import numpy as np
import pandas as pd

from scheduler import Job, forecast_greedy, make_threshold, oracle, run_now


def test_worked_example() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    forecast = np.array([750, 740, 700, 690, 720, 760] + [800] * 18)
    jobs = [
        Job("A", 1, 2, pd.Timestamp("2025-01-01T16:00:00Z"), True),
        Job("B", 2, 3, pd.Timestamp("2025-01-01T13:00:00Z"), True),
    ]
    assert forecast_greedy(now, jobs, forecast, 2) == ["B"]


def test_must_run_and_capacity_skip() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    jobs = [
        Job("urgent", 1, 2, now + pd.Timedelta(hours=2), True),
        Job("too-large", 3, 1, now + pd.Timedelta(hours=8), True),
        Job("small", 1, 1, now + pd.Timedelta(hours=8), True),
    ]
    assert forecast_greedy(now, jobs, np.zeros(24), 2) == ["urgent", "small"]


def test_oracle_is_same_decision_as_greedy() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    jobs = [Job("a", 1, 1.5, now + pd.Timedelta(hours=8), True)]
    forecast = np.arange(24, dtype=float)
    assert oracle is forecast_greedy
    assert oracle(now, jobs, forecast, 2) == forecast_greedy(now, jobs, forecast, 2)


def test_run_now_respects_order_and_capacity() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    jobs = [Job("large", 3, 1, now, False), Job("small", 1, 1, now, False)]
    assert run_now(now, jobs, np.zeros(24), 2) == ["small"]


def test_threshold_uses_only_supplied_median() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    jobs = [Job("defer", 1, 1, now + pd.Timedelta(hours=4), True)]
    policy = make_threshold(500.0)
    assert policy(now, jobs, np.array([499.0]), 1) == ["defer"]
    assert policy(now, jobs, np.array([500.0]), 1) == []


def test_threshold_still_runs_must_run_job() -> None:
    now = pd.Timestamp("2025-01-01T10:00:00Z")
    urgent = Job("urgent", 1, 1, now + pd.Timedelta(hours=1), True)
    assert make_threshold(500.0)(now, [urgent], np.array([900.0]), 1) == ["urgent"]

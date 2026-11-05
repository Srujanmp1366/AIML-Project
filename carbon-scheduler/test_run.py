import pandas as pd
import pytest

import run


def _zone_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = pd.DataFrame({
        "job_id": ["job-a"],
        "arrival": ["2025-01-01T00:00:00Z"],
        "duration_hours": [1.0],
        "gpus": [1],
        "deferrable": [True],
    })
    truth = pd.DataFrame({
        "zone": ["IN-EA", "IN-WE"],
        "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "actual_ci": [400.0, 500.0],
        "mef": [420.0, 520.0],
    })
    forecasts = pd.DataFrame({
        "zone": ["IN-EA", "IN-WE"],
        "issue_time": ["2024-12-31T23:00:00Z", "2024-12-31T23:00:00Z"],
        "target_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "predicted_ci": [390.0, 490.0],
    })
    return jobs, truth, forecasts


def test_zone_frames_pairs_matching_data_and_reuses_shared_jobs() -> None:
    jobs, truth, forecasts = _zone_inputs()

    frames = run._zone_frames(jobs, truth, forecasts)

    assert [frame[0] for frame in frames] == ["IN-EA", "IN-WE"]
    assert all(frame[1].equals(jobs) for frame in frames)
    assert [frame[2]["actual_ci"].iloc[0] for frame in frames] == [400.0, 500.0]
    assert all("zone" not in frame[2].columns for frame in frames)
    assert all("zone" not in frame[3].columns for frame in frames)


def test_zone_frames_reject_mismatched_zone_sets() -> None:
    jobs, truth, forecasts = _zone_inputs()
    forecasts.loc[forecasts["zone"] == "IN-WE", "zone"] = "IN-SO"

    with pytest.raises(ValueError, match="zone sets must match"):
        run._zone_frames(jobs, truth, forecasts)


def test_sweep_writes_combined_and_per_zone_outputs(tmp_path, monkeypatch) -> None:
    jobs, _, _ = _zone_inputs()
    times = pd.date_range("2025-01-01T00:00:00Z", periods=48, freq="h")
    issue_time = times[0] - pd.Timedelta(hours=1)
    truth_frames = []
    forecast_frames = []
    for zone, ci in (("IN-EA", 400.0), ("IN-WE", 500.0)):
        truth_frames.append(pd.DataFrame({
            "zone": zone,
            "time": times,
            "actual_ci": ci,
            "mef": ci + 20,
        }))
        forecast_frames.append(pd.DataFrame({
            "zone": zone,
            "issue_time": issue_time,
            "target_time": times[:24],
            "predicted_ci": ci,
        }))
    truth = pd.concat(truth_frames, ignore_index=True)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jobs.to_csv(data_dir / "jobs.csv", index=False)
    truth_path = data_dir / "grid_truth_all_zones.csv"
    truth.to_csv(truth_path, index=False)
    forecast_path = data_dir / "forecasts_all_zones.csv"
    forecasts.to_csv(forecast_path, index=False)

    monkeypatch.setattr(run, "BASE_DIR", tmp_path)
    monkeypatch.setattr(run, "DATA_DIR", data_dir)
    monkeypatch.setattr(run, "POLICIES", ("run_now",))
    monkeypatch.setattr(run, "K_VALUES", [6])
    monkeypatch.setattr(run, "CAPACITIES", (2,))

    run._sweep(forecast_path, figures=False, truth_path=truth_path)

    summary = pd.read_csv(tmp_path / "results" / "summary.csv")
    assert summary[["zone", "emissions_avg_kg"]].to_dict("records") == [
        {"zone": "IN-EA", "emissions_avg_kg": pytest.approx(0.24)},
        {"zone": "IN-WE", "emissions_avg_kg": pytest.approx(0.30)},
    ]
    for zone in ("IN-EA", "IN-WE"):
        zone_dir = tmp_path / "results" / "zones" / zone
        assert (zone_dir / "summary.csv").is_file()
        assert (zone_dir / "per_job" / f"{zone}_run_now_k6_g2.csv").is_file()

"""Command-line entry point for naive forecast generation and policy sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from simulator import DATA_DIR, K_VALUES, make_naive_forecast, run_sim


BASE_DIR = Path(__file__).resolve().parent
POLICIES = ("run_now", "threshold", "forecast_greedy", "oracle")
CAPACITIES: tuple[int | None, ...] = (64, 32, 128, None)


def _input_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() or path.exists() else BASE_DIR / path


def _draw_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    at_64 = summary.loc[summary["gpus"].astype(str) == "64"]
    fig, axis = plt.subplots(figsize=(8, 5))
    for policy in ("forecast_greedy", "oracle"):
        subset = at_64.loc[at_64["policy"] == policy].sort_values("k")
        for accounting, column in (
            ("average", "percent_saved_vs_run_now_avg_pct"),
            ("marginal", "percent_saved_vs_run_now_marg_pct"),
        ):
            axis.plot(subset["k"], subset[column], marker="o", label=f"{policy}, {accounting}")
    axis.set(xlabel="Deferral window k (hours)", ylabel="CO2 saved vs run_now (%)", title="Carbon savings by deferral window at 64 GPUs")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "co2_saved_vs_k.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    capacities = ("64", "unlimited")
    for axis, accounting, column in zip(
        axes,
        ("average", "marginal"),
        ("percent_saved_vs_run_now_avg_pct", "percent_saved_vs_run_now_marg_pct"),
    ):
        for policy in ("forecast_greedy", "oracle"):
            for capacity in capacities:
                subset = summary.loc[
                    (summary["policy"] == policy) & (summary["gpus"].astype(str) == capacity)
                ].sort_values("k")
                axis.plot(subset["k"], subset[column], marker="o", label=f"{policy}, {capacity} GPUs")
        axis.set(xlabel="Deferral window k (hours)", title=f"{accounting.title()} emissions")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("CO2 saved vs run_now (%)")
    axes[1].legend(fontsize="small")
    fig.suptitle("Carbon savings at 64 GPUs and unlimited capacity")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_savings.png", dpi=160)
    plt.close(fig)


def _zone_frames(
    jobs: pd.DataFrame,
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> list[tuple[str | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    has_truth_zone = "zone" in truth.columns
    has_forecast_zone = "zone" in forecasts.columns
    if has_truth_zone != has_forecast_zone:
        raise ValueError("grid_truth and forecast CSVs must either both include a zone column or neither include one")
    if not has_truth_zone:
        return [(None, jobs, truth, forecasts)]

    frames: list[tuple[str | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    truth = truth.copy()
    forecasts = forecasts.copy()
    truth["zone"] = truth["zone"].astype("string").str.strip()
    forecasts["zone"] = forecasts["zone"].astype("string").str.strip()
    if truth["zone"].isna().any() or truth["zone"].eq("").any():
        raise ValueError("grid_truth.zone must not be empty")
    if forecasts["zone"].isna().any() or forecasts["zone"].eq("").any():
        raise ValueError("forecast.zone must not be empty")

    truth_zones = set(truth["zone"].unique())
    forecast_zones = set(forecasts["zone"].unique())
    if truth_zones != forecast_zones:
        missing_truth = sorted(forecast_zones - truth_zones)
        missing_forecasts = sorted(truth_zones - forecast_zones)
        raise ValueError(
            "grid_truth and forecast zone sets must match "
            f"(missing truth: {missing_truth}; missing forecasts: {missing_forecasts})"
        )
    slugs = [_zone_slug(str(zone)) for zone in sorted(truth_zones)]
    if len(slugs) != len(set(slugs)):
        raise ValueError("zone names collide after conversion to output folder names")

    if "zone" in jobs.columns:
        jobs = jobs.copy()
        jobs["zone"] = jobs["zone"].astype("string").str.strip()
        if jobs["zone"].isna().any() or jobs["zone"].eq("").any():
            raise ValueError("jobs.zone must not be empty")
        unknown_job_zones = set(jobs["zone"].unique()) - truth_zones
        if unknown_job_zones:
            raise ValueError(f"jobs contain zones with no matching grid data: {sorted(unknown_job_zones)}")

    for zone in sorted(truth_zones):
        zone_jobs = jobs.loc[jobs["zone"] == zone].drop(columns="zone") if "zone" in jobs.columns else jobs
        frames.append((
            str(zone),
            zone_jobs,
            truth.loc[truth["zone"] == zone].drop(columns="zone"),
            forecasts.loc[forecasts["zone"] == zone].drop(columns="zone"),
        ))
    return frames


def _zone_slug(zone: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", zone).strip("._") or "zone"


def _sweep(forecast_path: Path, figures: bool, truth_path: Path | None = None) -> None:
    jobs = pd.read_csv(DATA_DIR / "jobs.csv")
    truth = pd.read_csv(truth_path or DATA_DIR / "grid_truth.csv")
    forecasts = pd.read_csv(forecast_path)
    results_dir = BASE_DIR / "results"
    per_job_dir = results_dir / "per_job"
    per_job_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    zone_runs = _zone_frames(jobs, truth, forecasts)
    total_runs = len(zone_runs) * len(POLICIES) * len(K_VALUES) * len(CAPACITIES)
    run_number = 0
    for zone, zone_jobs, zone_truth, zone_forecasts in zone_runs:
        zone_results = results_dir / "zones" / _zone_slug(zone) if zone is not None else results_dir
        zone_per_job_dir = zone_results / "per_job" if zone is not None else per_job_dir
        zone_per_job_dir.mkdir(parents=True, exist_ok=True)
        zone_summaries: list[dict[str, object]] = []
        for policy in POLICIES:
            for k in K_VALUES:
                for capacity in CAPACITIES:
                    run_number += 1
                    gpu_label = "unlimited" if capacity is None else str(capacity)
                    print(
                        f"[{run_number}/{total_runs}]"
                        f"{f' zone={zone},' if zone is not None else ''}"
                        f" policy={policy}, k={k}, GPUs={gpu_label}",
                        flush=True,
                    )
                    metrics, per_job = run_sim(
                        policy, k, capacity, zone_jobs, zone_truth, zone_forecasts
                    )
                    row: dict[str, object] = {
                        "policy": policy,
                        "k": k,
                        "gpus": gpu_label,
                        **metrics,
                    }
                    if zone is not None:
                        row["zone"] = zone
                    zone_summaries.append(row)
                    filename = f"{policy}_k{k}_g{gpu_label}.csv"
                    if zone is not None:
                        filename = f"{_zone_slug(zone)}_{filename}"
                    per_job.to_csv(zone_per_job_dir / filename, index=False)
        summaries.extend(zone_summaries)
        if zone is not None:
            zone_summary = pd.DataFrame(zone_summaries)
            zone_results.mkdir(parents=True, exist_ok=True)
            zone_summary.to_csv(zone_results / "summary.csv", index=False)
            if figures:
                _draw_figures(zone_summary, zone_results)
    summary = pd.DataFrame(summaries)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary.csv", index=False)
    if figures and len(zone_runs) == 1 and zone_runs[0][0] is None:
        _draw_figures(summary, results_dir)
    print(f"Wrote {len(summary)} runs to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-driven carbon-aware ML job simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("naive", help="build data/forecasts_naive24.csv")
    sweep = commands.add_parser("sweep", help="run the policy, k, and capacity sweep")
    sweep.add_argument("--forecast", required=True, help="forecast CSV path")
    sweep.add_argument("--truth", help="grid truth CSV path (defaults to data/grid_truth.csv)")
    sweep.add_argument("--figures", action="store_true", help="write CO2 comparison plots")
    args = parser.parse_args()
    if args.command == "naive":
        output = make_naive_forecast(pd.read_csv(DATA_DIR / "grid_truth.csv"))
        print(f"Wrote {output}")
    else:
        truth_path = _input_path(args.truth) if args.truth else None
        _sweep(_input_path(args.forecast), args.figures, truth_path)


if __name__ == "__main__":
    main()

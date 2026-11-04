"""Command-line entry point for naive forecast generation and policy sweeps."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from simulator import DATA_DIR, K_VALUES, make_naive_forecast, run_sim


BASE_DIR = Path(__file__).resolve().parent
POLICIES = ("run_now", "threshold", "forecast_greedy", "oracle")
CAPACITIES: tuple[int | None, ...] = (64, 32, 128, None)


def iter_runs(
    jobs: pd.DataFrame,
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
    policies: Sequence[str] = POLICIES,
    k_values: Sequence[int] = K_VALUES,
    capacities: Sequence[int | None] = CAPACITIES,
    forecast_error_margin_ci: float = 0.0,
) -> Iterator[tuple[dict[str, object], pd.DataFrame]]:
    """Yield in-memory sweep results for the CLI and web interface."""
    for policy in policies:
        for k in k_values:
            for capacity in capacities:
                metrics, per_job = run_sim(
                    policy, k, capacity, jobs, truth, forecasts,
                    forecast_error_margin_ci=forecast_error_margin_ci,
                )
                yield {
                    "policy": policy,
                    "k": k,
                    "gpus": "unlimited" if capacity is None else str(capacity),
                    **metrics,
                }, per_job


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


def _sweep(forecast_path: Path, figures: bool) -> None:
    jobs = pd.read_csv(DATA_DIR / "jobs.csv")
    truth = pd.read_csv(DATA_DIR / "grid_truth.csv")
    forecasts = pd.read_csv(forecast_path)
    results_dir = BASE_DIR / "results"
    per_job_dir = results_dir / "per_job"
    per_job_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    total_runs = len(POLICIES) * len(K_VALUES) * len(CAPACITIES)
    run_number = 0
    for summary, per_job in iter_runs(jobs, truth, forecasts):
        run_number += 1
        print(
            f"[{run_number}/{total_runs}] policy={summary['policy']}, "
            f"k={summary['k']}, GPUs={summary['gpus']}",
            flush=True,
        )
        summaries.append(summary)
        per_job.to_csv(
            per_job_dir / f"{summary['policy']}_k{summary['k']}_g{summary['gpus']}.csv",
            index=False,
        )
    summary = pd.DataFrame(summaries)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary.csv", index=False)
    if figures:
        _draw_figures(summary, results_dir)
    print(f"Wrote {len(summary)} runs to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-driven carbon-aware ML job simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("naive", help="build data/forecasts_naive24.csv")
    sweep = commands.add_parser("sweep", help="run the policy, k, and capacity sweep")
    sweep.add_argument("--forecast", required=True, help="forecast CSV path")
    sweep.add_argument("--figures", action="store_true", help="write CO2 comparison plots")
    args = parser.parse_args()
    if args.command == "naive":
        output = make_naive_forecast(pd.read_csv(DATA_DIR / "grid_truth.csv"))
        print(f"Wrote {output}")
    else:
        _sweep(_input_path(args.forecast), args.figures)


if __name__ == "__main__":
    main()

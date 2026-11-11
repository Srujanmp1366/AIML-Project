"""Write simulation tables and static figures into one results directory."""

from __future__ import annotations

from pathlib import Path
import re
from threading import Lock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
_WRITE_LOCK = Lock()


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "single-grid"


def scenario_stem(zone: str | None, policy: str, k: int, gpus: int | None) -> str:
    capacity = "unlimited" if gpus is None else str(gpus)
    return f"{_slug(zone or 'single-grid')}_{_slug(policy)}_k{k}_g{capacity}"


def trace_series(truth: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Pair actual CI with the forecast available one hour before each hour."""
    issues = pd.to_datetime(forecasts["issue_time"], utc=True)
    targets = pd.to_datetime(forecasts["target_time"], utc=True)
    one_hour = forecasts.loc[targets == issues + pd.Timedelta(hours=1), ["target_time", "predicted_ci"]].copy()
    one_hour["time"] = pd.to_datetime(one_hour.pop("target_time"), utc=True)
    actual = truth[["time", "actual_ci"]].copy()
    actual["time"] = pd.to_datetime(actual["time"], utc=True)
    paired = one_hour.merge(actual, on="time", validate="one_to_one").sort_values("time")
    if len(paired) > 240:
        indices = np.linspace(0, len(paired) - 1, 240).round().astype(int)
        paired = paired.iloc[indices]
    return paired[["time", "actual_ci", "predicted_ci"]]


def _write_summary(rows: pd.DataFrame) -> None:
    """Keep one latest summary row per zone, policy, window, and capacity."""
    path = RESULTS_DIR / "summary.csv"
    if path.is_file():
        existing = pd.read_csv(path)
        if "zone" not in existing:
            existing["zone"] = "single-grid"
        rows = pd.concat([existing, rows], ignore_index=True)
    rows["zone"] = rows["zone"].astype(str).str.strip()
    rows["policy"] = rows["policy"].astype(str).str.strip()
    rows["gpus"] = rows["gpus"].astype(str).str.strip()
    rows["k"] = pd.to_numeric(rows["k"], errors="raise").astype(int)
    rows = rows.drop_duplicates(["zone", "policy", "k", "gpus"], keep="last")
    temporary = RESULTS_DIR / "summary.csv.tmp"
    rows.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_emissions(metrics: dict[str, float], path: Path, policy: str) -> None:
    labels = ["Run now", policy.replace("_", " ").title(), "Perfect CI"]
    values = [metrics.get("run_now_avg_kg"), metrics.get("emissions_avg_kg"), metrics.get("greedy_perfect_ci_avg_kg")]
    heights = [float(value) if value is not None and np.isfinite(value) else 0.0 for value in values]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, heights, color=["#a5b6a8", "#279b5e", "#7c9fd0"])
    axis.bar_label(bars, labels=[f"{value:,.1f}" if value is not None and np.isfinite(value) else "N/A" for value in values], padding=4)
    axis.set(ylabel="Average emissions (kg CO₂)", title="Carbon emissions comparison")
    axis.set_ylim(0, max(heights + [1]) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_carbon_trace(trace: pd.DataFrame, path: Path) -> None:
    frame = trace.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    for column in ("actual_ci", "predicted_ci"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time", "actual_ci", "predicted_ci"]).sort_values("time")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    if not frame.empty:
        axis.plot(frame["time"], frame["actual_ci"], color="#279b5e", label="Actual CI")
        axis.plot(frame["time"], frame["predicted_ci"], color="#7195cf", linestyle="--", label="One-hour-ahead forecast")
        axis.legend()
    axis.set(xlabel="UTC time", ylabel="Carbon intensity (gCO₂/kWh)", title="Grid carbon intensity over time")
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_waits(per_job: pd.DataFrame, path: Path) -> None:
    waits = pd.to_numeric(per_job["start_wait_hours"], errors="coerce").dropna()
    waits = waits.loc[waits >= 0]
    labels = ["<1h", "1–6h", "6–12h", "12–24h", "24h+"]
    counts = [
        int((waits < 1).sum()),
        int(((waits >= 1) & (waits < 6)).sum()),
        int(((waits >= 6) & (waits < 12)).sum()),
        int(((waits >= 12) & (waits < 24)).sum()),
        int((waits >= 24).sum()),
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, counts, color="#36a66a")
    axis.bar_label(bars, padding=4)
    axis.set(xlabel="Arrival to first start", ylabel="Jobs", title="Job start wait distribution")
    axis.set_ylim(0, max(counts + [1]) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_scenario(
    metrics: dict[str, float],
    per_job: pd.DataFrame,
    trace: pd.DataFrame,
    policy: str,
    k: int,
    gpus: int | None,
    zone: str | None,
) -> dict[str, str]:
    """Persist one scenario and return filenames relative to results/."""
    stem = scenario_stem(zone, policy, k, gpus)
    names = {
        "summary": "summary.csv",
        "jobs": f"{stem}_jobs.csv",
        "emissions": f"{stem}_emissions.png",
        "carbon_intensity": f"{stem}_carbon_intensity.png",
        "start_waits": f"{stem}_start_waits.png",
    }
    row: dict[str, object] = {
        "zone": zone or "single-grid",
        "policy": policy,
        "k": k,
        "gpus": "unlimited" if gpus is None else str(gpus),
        **metrics,
    }
    with _WRITE_LOCK:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _write_summary(pd.DataFrame([row]))
        per_job.to_csv(RESULTS_DIR / names["jobs"], index=False)
        _save_emissions(metrics, RESULTS_DIR / names["emissions"], policy)
        _save_carbon_trace(trace, RESULTS_DIR / names["carbon_intensity"])
        _save_waits(per_job, RESULTS_DIR / names["start_waits"])
    return names


def save_comparison(rows: list[dict[str, object]], zone: str | None, k: int, gpus: int | None) -> dict[str, str]:
    """Persist a policy comparison in the same directory as scenario outputs."""
    capacity = "unlimited" if gpus is None else str(gpus)
    name = f"{_slug(zone or 'single-grid')}_k{k}_g{capacity}_policy_comparison.png"
    frame = pd.DataFrame([{"zone": zone or "single-grid", **row} for row in rows])
    values = pd.to_numeric(frame["emissions_avg_kg"], errors="coerce")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(frame["policy"].astype(str).str.replace("_", " "), values.fillna(0), color="#279b5e")
    axis.bar_label(bars, labels=[f"{value:,.1f}" if pd.notna(value) else "N/A" for value in values], padding=4)
    axis.set(ylabel="Average emissions (kg CO₂)", title="Scheduling policy comparison")
    axis.set_ylim(0, max(values.fillna(0).tolist() + [1]) * 1.18)
    fig.tight_layout()
    with _WRITE_LOCK:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _write_summary(frame)
        fig.savefig(RESULTS_DIR / name, dpi=160)
    plt.close(fig)
    return {"summary": "summary.csv", "policy_comparison": name}


def save_dashboard_charts(
    comparison: pd.DataFrame,
    trend: pd.DataFrame,
    table: pd.DataFrame,
    zone: str,
    k: int,
    capacity: str,
    accounting: str,
) -> dict[str, str]:
    """Save the Streamlit dashboard's filtered table and three chart views."""
    stem = f"dashboard_{_slug(zone)}_k{k}_g{_slug(capacity)}_{accounting}"
    names = {
        "rows": f"{stem}_rows.csv",
        "policy_emissions": f"{stem}_policy_emissions.png",
        "savings_by_k": f"{stem}_savings_by_k.png",
        "deadline_tradeoff": f"{stem}_deadline_tradeoff.png",
    }
    emissions_column = f"emissions_{accounting}_kg"
    savings_column = f"percent_saved_vs_run_now_{accounting}_pct"
    with _WRITE_LOCK:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS_DIR / names["rows"], index=False)

        fig, axis = plt.subplots(figsize=(8, 4.5))
        values = pd.to_numeric(comparison[emissions_column], errors="coerce")
        bars = axis.bar(comparison["policy"].astype(str), values.fillna(0), color="#279b5e")
        axis.bar_label(bars, labels=[f"{value:,.1f}" if pd.notna(value) else "N/A" for value in values], padding=4)
        axis.set(ylabel="Emissions (kg CO₂)", title=f"{accounting.title()} emissions by policy")
        axis.set_ylim(0, max(values.fillna(0).tolist() + [1]) * 1.18)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / names["policy_emissions"], dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 4.5))
        for policy, group in trend.groupby("policy", observed=True):
            values = pd.to_numeric(group[savings_column], errors="coerce")
            axis.plot(group["k"], values, marker="o", label=str(policy))
        axis.set(xlabel="Deferral window (hours)", ylabel="Saved vs run now (%)", title="Carbon savings by deferral window")
        axis.grid(alpha=0.2)
        if not trend.empty:
            axis.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / names["savings_by_k"], dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 4.5))
        for _, row in comparison.iterrows():
            x = pd.to_numeric(row["deadline_misses"], errors="coerce")
            y = pd.to_numeric(row[savings_column], errors="coerce")
            if pd.notna(x) and pd.notna(y):
                axis.scatter(x, y, s=65, label=str(row["policy"]))
        axis.set(xlabel="Deadline misses (jobs)", ylabel="Saved vs run now (%)", title="Carbon and deadline trade-off")
        axis.grid(alpha=0.2)
        if axis.collections:
            axis.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / names["deadline_tradeoff"], dpi=160)
        plt.close(fig)
    return names

"""Streamlit dashboard for carbon-aware scheduler summary results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results" / "summary.csv"
REQUIRED_COLUMNS = {
    "policy",
    "k",
    "gpus",
    "total_kwh",
    "emissions_avg_kg",
    "emissions_marg_kg",
    "deadline_misses",
    "mean_added_delay_hours",
    "gpu_utilisation",
    "percent_saved_vs_run_now_avg_pct",
    "percent_saved_vs_run_now_marg_pct",
}
POLICY_ORDER = ("run_now", "threshold", "forecast_greedy", "oracle")


def _read_summary(source: object) -> tuple[pd.DataFrame | None, str | None]:
    try:
        summary = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        return None, f"Could not read the summary CSV: {error}"

    missing_columns = sorted(REQUIRED_COLUMNS - set(summary.columns))
    if missing_columns:
        return None, f"Summary CSV is missing required columns: {', '.join(missing_columns)}."

    summary = summary.copy()
    if "zone" not in summary.columns:
        summary["zone"] = "single-grid"
    elif summary["zone"].isna().any():
        return None, "Summary CSV contains a row with an empty grid zone."
    summary["zone"] = summary["zone"].astype(str).str.strip()
    summary["policy"] = summary["policy"].astype(str).str.strip()
    summary["gpus"] = summary["gpus"].astype(str).str.strip()
    numeric_columns = REQUIRED_COLUMNS - {"policy", "gpus"}
    for column in numeric_columns:
        original_values = summary[column]
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
        invalid_values = original_values.notna() & original_values.astype(str).str.strip().ne("") & summary[column].isna()
        if invalid_values.any():
            return None, f"Summary CSV contains non-numeric values in `{column}`."

    keys = ["zone", "policy", "k", "gpus"]
    if (
        summary[keys].isna().any().any()
        or summary["zone"].eq("").any()
        or summary["policy"].eq("").any()
        or summary["gpus"].eq("").any()
    ):
        return None, "Summary CSV contains a row with an empty zone, policy, k, or GPU capacity."
    for column in ("k", "total_kwh", "deadline_misses"):
        if summary[column].isna().any():
            return None, f"Summary CSV contains missing values in `{column}`."
    if summary.duplicated(keys).any():
        return None, "Summary CSV must contain at most one row per policy, k, and GPU capacity."
    return summary, None


def _format_number(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    number = float(value)
    if abs(number) < 0.5 * 10 ** -decimals:
        number = 0.0
    return f"{number:,.{decimals}f}"


st.set_page_config(
    page_title="Carbon-Aware Scheduler",
    page_icon="🌱",
    layout="wide",
)

st.title("Carbon-aware scheduling simulation")
st.markdown(
    "Compare how scheduling policy, deferral window, and GPU capacity affect "
    "energy, emissions, and job deadlines."
)

with st.sidebar:
    st.header("Results")
    uploaded_summary = st.file_uploader("Upload a summary.csv", type="csv")
    if uploaded_summary is None:
        st.caption(f"Using `{SUMMARY_PATH.relative_to(BASE_DIR)}`")
        summary, load_error = _read_summary(SUMMARY_PATH)
    else:
        summary, load_error = _read_summary(uploaded_summary)

if load_error:
    st.error(load_error)
    st.info("Run the simulator sweep to create `results/summary.csv`, or upload a compatible summary CSV.")
    st.stop()
if summary is None or summary.empty:
    st.error("The summary CSV has no result rows.")
    st.stop()

available_policies = summary["policy"].unique().tolist()
ordered_policies = [policy for policy in POLICY_ORDER if policy in available_policies]
ordered_policies.extend(policy for policy in available_policies if policy not in ordered_policies)
available_zones = sorted(summary["zone"].unique().tolist())

with st.sidebar:
    st.header("Scenario")
    selected_zone = st.selectbox("Grid zone", available_zones)
    selected_policy = st.selectbox("Policy", ordered_policies, index=ordered_policies.index(
        "forecast_greedy" if "forecast_greedy" in ordered_policies else ordered_policies[0]
    ))
    zone_summary = summary.loc[summary["zone"] == selected_zone]
    available_k = sorted(zone_summary["k"].dropna().unique().tolist())
    selected_k = st.selectbox("Deferral window (hours)", available_k, index=(
        available_k.index(24) if 24 in available_k else len(available_k) - 1
    ))
    capacities = zone_summary["gpus"].dropna().unique().tolist()
    capacities = sorted(capacities, key=lambda value: (value == "unlimited", int(value) if value.isdigit() else 0))
    selected_capacity = st.selectbox(
        "GPU capacity",
        capacities,
        index=capacities.index("64") if "64" in capacities else 0,
    )
    accounting = st.radio("Emissions accounting", ("Average grid intensity", "Marginal grid intensity"))

accounting_key = "avg" if accounting.startswith("Average") else "marg"
emissions_column = f"emissions_{accounting_key}_kg"
savings_column = f"percent_saved_vs_run_now_{accounting_key}_pct"
selected_rows = summary.loc[
    (summary["zone"] == selected_zone)
    & (summary["policy"] == selected_policy)
    & (summary["k"] == selected_k)
    & (summary["gpus"] == selected_capacity)
]
baseline_rows = summary.loc[
    (summary["zone"] == selected_zone)
    & (summary["policy"] == "run_now")
    & (summary["k"] == selected_k)
    & (summary["gpus"] == selected_capacity)
]

st.caption(
    f"Selected scenario: **{selected_zone}** · **{selected_policy}** · **k = {selected_k:g} hours** · "
    f"**{selected_capacity} GPUs** · {accounting.lower()}."
)

if selected_rows.empty:
    st.warning("No result exists for this policy, deferral window, and GPU capacity.")
else:
    result = selected_rows.iloc[0]
    emissions = result[emissions_column]
    savings = result[savings_column]
    baseline_emissions = baseline_rows.iloc[0][emissions_column] if not baseline_rows.empty else float("nan")

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        f"{accounting_key.title()} emissions",
        f"{_format_number(emissions)} kg CO₂" if pd.notna(emissions) else "Unavailable",
    )
    metric_columns[1].metric(
        "Saved vs run-now baseline",
        f"{_format_number(savings, 2)}%" if pd.notna(savings) else "Unavailable",
        help="Savings are only reported when the simulator considers the compared runs complete and comparable.",
    )
    metric_columns[2].metric("Energy used", f"{_format_number(result['total_kwh'])} kWh")
    metric_columns[3].metric(
        "Deadline misses",
        _format_number(result["deadline_misses"], 0),
        help="Absolute number of deferrable jobs that missed their deadline.",
    )
    st.caption(
        f"Run-now baseline: {_format_number(baseline_emissions)} kg CO₂ · "
        f"Mean added delay: {_format_number(result['mean_added_delay_hours'], 1)} hours · "
        + (
            f"GPU utilization: {_format_number(result['gpu_utilisation'] * 100, 1)}%"
            if pd.notna(result["gpu_utilisation"])
            else "GPU utilization: not defined for unlimited capacity"
        )
    )

if accounting_key == "marg" and summary[emissions_column].isna().all():
    st.warning("Marginal emissions are unavailable in these results; choose average grid intensity to compare runs.")

st.subheader("Compare policies")
comparison = summary.loc[
    (summary["zone"] == selected_zone)
    & (summary["k"] == selected_k)
    & (summary["gpus"] == selected_capacity)
].copy()
comparison["policy"] = pd.Categorical(comparison["policy"], ordered_policies, ordered=True)
comparison = comparison.sort_values("policy")

left, right = st.columns(2)
with left:
    st.markdown(f"**{accounting} by policy**")
    emissions_chart = px.bar(
        comparison,
        x="policy",
        y=emissions_column,
        color="policy",
        category_orders={"policy": ordered_policies},
        labels={"policy": "Policy", emissions_column: "Emissions (kg CO₂)"},
        hover_data={"policy": True, emissions_column: ":,.2f", "deadline_misses": True},
    )
    emissions_chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="kg CO₂")
    st.plotly_chart(emissions_chart, width="stretch")
with right:
    st.markdown("**Carbon savings across deferral windows**")
    trend = summary.loc[
        (summary["zone"] == selected_zone) & (summary["gpus"] == selected_capacity)
    ].copy()
    trend["policy"] = pd.Categorical(trend["policy"], ordered_policies, ordered=True)
    savings_chart = px.line(
        trend,
        x="k",
        y=savings_column,
        color="policy",
        markers=True,
        category_orders={"policy": ordered_policies},
        labels={"k": "Deferral window (hours)", savings_column: "Saved vs run-now (%)"},
        hover_data={"policy": True, savings_column: ":.2f", "deadline_misses": True},
    )
    savings_chart.update_layout(xaxis_title="Deferral window (hours)", yaxis_title="Saved vs run-now (%)")
    st.plotly_chart(savings_chart, width="stretch")

st.subheader("Carbon and deadline trade-off")
if comparison[savings_column].notna().any():
    tradeoff = px.scatter(
        comparison,
        x="deadline_misses",
        y=savings_column,
        color="policy",
        size="total_kwh",
        hover_name="policy",
        category_orders={"policy": ordered_policies},
        labels={
            "deadline_misses": "Deadline misses (jobs)",
            savings_column: "Saved vs run-now (%)",
            "total_kwh": "Energy (kWh)",
        },
    )
    tradeoff.update_layout(xaxis_title="Deadline misses (jobs)", yaxis_title="Saved vs run-now (%)")
    st.plotly_chart(tradeoff, width="stretch")
else:
    st.info("This summary does not contain comparable savings values for the selected accounting method.")

with st.expander("How to interpret these results: assumptions and caveats"):
    st.markdown(
        """
        - **Baseline:** `run_now` is the reference for carbon-savings percentages.
          A percentage is shown only when the simulator considers both runs complete and comparable.
        - **Policies:** `threshold` uses a past-only 30-day median; `forecast_greedy`
          uses the available 24-hour forecast. `oracle` receives actual carbon intensity
          as a perfect signal, but still uses the greedy policy—it is **not** a globally
          optimal schedule.
        - **Average vs marginal:** emissions use actual grid intensity or marginal
          emissions factors (MEF), respectively. Marginal results are unavailable when
          MEF data is missing.
        - **Energy model:** GPU energy assumes 0.4 kW per GPU and a PUE of 1.5.
          Resuming a paused job adds five minutes of GPU energy without additional
          wall-clock time; these are modeling assumptions, not measurements.
        - **Utilization:** finite-capacity GPU utilization is busy GPU-hours divided by
          available GPU-hours. It is undefined for unlimited capacity.
        - **Scope:** this view summarizes completed simulation runs. It is not a
          live scheduler, and the summary CSV does not include an hourly GPU or
          emissions timeline.
        """
    )

st.subheader("Result rows")
table = summary.loc[
    (summary["zone"] == selected_zone) & (summary["gpus"] == selected_capacity)
].sort_values(["k", "policy"]).copy()
display_columns = [
    "zone",
    "policy",
    "k",
    "gpus",
    emissions_column,
    savings_column,
    "total_kwh",
    "deadline_misses",
    "mean_added_delay_hours",
    "gpu_utilisation",
]
st.dataframe(
    table[display_columns],
    hide_index=True,
    width="stretch",
    column_config={
        "zone": st.column_config.TextColumn("Grid zone"),
        "policy": st.column_config.TextColumn("Policy"),
        "k": st.column_config.NumberColumn("Deferral window (hours)", format="%d"),
        "gpus": st.column_config.TextColumn("GPU capacity"),
        emissions_column: st.column_config.NumberColumn(f"{accounting} emissions (kg CO₂)", format="%.2f"),
        savings_column: st.column_config.NumberColumn("Saved vs run-now (%)", format="%.2f%%"),
        "total_kwh": st.column_config.NumberColumn("Energy (kWh)", format="%.1f"),
        "deadline_misses": st.column_config.NumberColumn("Deadline misses", format="%d"),
        "mean_added_delay_hours": st.column_config.NumberColumn("Mean added delay (hours)", format="%.2f"),
        "gpu_utilisation": st.column_config.NumberColumn("GPU utilization", format="percent"),
    },
)

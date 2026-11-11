# Carbon aware scheduler

This small Python codebase compares hourly scheduling policies for ML jobs against Indian-grid carbon-intensity traces. It consumes job, actual-grid, and forecast CSV files; it does not train a forecaster or fit an MEF regression.

Requires Python 3.10+, pandas, numpy, and matplotlib. Install pytest to run the inline-fixture test suite.

## React interface

The frontend lives in `frontend/`. It calls the local Python API in `api.py`, which uses
`simulator.run_sim` for a single scenario and `run.iter_runs` for policy comparisons.
The interface shows emissions, completion and delay metrics, and per-job outcomes.
After a simulation, the Overview also graphs actual versus one-hour-ahead forecast
carbon intensity and the distribution of job start waits. The API saves the summary
and per-job tables as CSV and the emissions, carbon-intensity, and wait graphs as PNG,
all directly in `results/`. The dashboard provides download links to these files.

If you do not already have `.venv`, create it with `python -m venv .venv`.
From `carbon-scheduler`, start the API in one PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python api.py
```

In a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the URL Vite prints (normally `http://127.0.0.1:5173`). The API runs at
`http://127.0.0.1:8000`; Vite forwards `/api` requests to it. The interface can read
`data/jobs.csv`, a `grid_truth*.csv`, and a `forecasts*.csv` in `data/`, or you can
upload all three CSV files in the browser. Select one grid zone for multi-zone traces.
Uploaded input files are not saved to disk; simulation outputs are saved in `results/`.
The optional year-alignment checkbox shifts job arrival dates in memory to the
forecast year when the files cover different years.

For a single-server build, run `npm run build` from `frontend/`, then run
`python api.py` from `carbon-scheduler/` and open `http://127.0.0.1:8000`.
The CLI commands below remain available separately and still write output files.

## Inputs

The current `data/` directory contains:

- `jobs.csv`: `job_id`, `arrival`, `duration_hours`, `gpus`, `deferrable`
- `grid_truth_all_zones.csv`: `zone`, `time`, `actual_ci`, `mef`
- `forecasts_lightgbm_all_zones (1).csv`: `zone`, `issue_time`, `target_time`, `predicted_ci`

The supplied jobs arrive in January 2026, while both grid traces cover 2025. Enable
year alignment in the React interface, or pass `--align-jobs-year` to the CLI, to
simulate those jobs on the corresponding January dates in 2025. This does not edit
`jobs.csv`. Alternatively, supply grid truth and forecasts covering January 2026.
The current `mef` column is empty, so marginal-emissions results are unavailable.

For multi-grid runs, add a `zone` column to **both** grid truth and forecast CSVs. The same zone value must identify the same grid in both files. Each zone must have a complete actual carbon-intensity trace matching its forecasts. The simulator runs each zone independently and writes a combined `results/summary.csv` with a `zone` column, plus per-zone summaries and per-job files under `results/zones/<zone>/`. If `jobs.csv` has no `zone` column, the same workload is simulated for every zone; if it does have one, each zone uses only its matching jobs.

Grid and forecast timestamps must identify UTC hours; job arrival timestamps may be at any UTC time. Each forecast issue time needs exactly 24 target rows, from the first target hour through the following 23 hours. `actual_ci` and `predicted_ci` are in gCO₂/kWh; MEF uses the same units. Leave `mef` empty when marginal intensity is unavailable. In that case the simulator warns and reports marginal scores as missing.

The simulator treats `deferrable` values `true`, `1`, and `yes` as true (case insensitive). GPU demand is rounded up to a whole GPU.

## Run

From this directory, use an environment with the dependencies above, then:

```bash
python run.py scenario --zone IN-EA --align-jobs-year
python run.py sweep --align-jobs-year
```

`scenario` runs the default forecast-greedy policy with a 24-hour deferral window
and 64 GPUs for one zone. It creates `results/summary.csv`, a per-job CSV, and three
PNG graphs in the same `results/` folder. Use `--policy`, `--k`, and `--gpus` to
change the scenario. The React interface writes the same output format.

`sweep` auto-detects the truth and forecast files when each has one matching file in
`data/`. It runs `run_now`, `threshold`, `forecast_greedy`, and `oracle` for
`k = 6, 12, 24, 48` and capacities of 64, 32, 128, and unlimited GPUs. It writes
`results/summary.csv`, per-job CSV files, per-zone summary CSV files, and comparison
PNG plots directly in `results/`. The older per-zone subfolders are also kept for
compatibility. Use `--truth` and `--forecast` to select files when there is more than
one match. Use `--no-figures` to skip the PNG plots.

The CLI and React API limit the simulation to job arrivals plus enough hours for
completion. The `run_sim` API also accepts an optional UTC `start` and exclusive
`end` timestamp.

## Dashboard

Install the project and dashboard dependencies, then run the dashboard from this directory:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads `results/summary.csv` by default. Generate it first with the
scenario or sweep command above if it is missing. You can also upload a compatible
summary CSV. Use the sidebar to select a policy, deferral window, GPU capacity, and
average or marginal emissions; the dashboard shows headline metrics, policy
comparisons, savings trends, and deadline trade-offs. It also saves the filtered
dashboard table as CSV and all three dashboard graphs as PNG in `results/`, with
download buttons in the **Saved CSV and PNG outputs** section.

## Model assumptions

- Scheduling decisions happen at job arrivals, job completions, and hour boundaries. Arriving jobs are eligible immediately. At each event, running jobs may be preempted and the policy chooses the jobs to run again; inference jobs get capacity first in arrival order.
- Carbon intensity and MEF are constant within each hour: energy used between 10:00 and 11:00 is scored with the 10:00 values. Jobs may start or finish mid-hour, and energy/utilisation account for partial-hour work.
- A resumed job adds five minutes of GPU energy use, with no added wall time. That overhead affects kWh and emissions, but not GPU utilisation.
- Energy is `GPU count × 0.4 kW × 1.5 PUE × (run hours + resume overhead)`. The 0.4 kW figure is anchored to the NVIDIA A100 SXM maximum power specification; it is a modeling reference, not a measurement of a particular workload. The PUE of 1.5 is a rounded scenario value near Uptime Institute's reported 2026 industry average PUE of 1.52. See the [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) and [Uptime Institute's 2026 PUE survey update](https://intelligence.uptimeinstitute.com/resource/growing-pue-advantage-larger-data-centers).
- Average and marginal emissions are always scored with actual CI/MEF for the hour energy was used. Missing MEF disables marginal scoring. A trailing 30-day median for `threshold` uses actual CI strictly before the current hour.
- `oracle` is the same greedy policy as `forecast_greedy`, with actual CI supplied as its 24-hour signal. It is a perfect-forecast comparison, not a globally optimal schedule.
- GPU utilisation is busy GPU-hours divided by finite capacity times simulated hours. It is undefined (`NaN`) for unlimited capacity.

## Limitations

- The simulator needs complete hourly truth and a complete 24-hour forecast at each simulated decision time; it does not fill gaps or infer timezone semantics from local time.
- There is no idle GPU power, CPU, memory, networking, checkpoint storage, startup cost, or grid-side response model.
- The fixed resume cost does not vary by model or checkpoint state.
- Forecast uncertainty and MEF regression error are outside the simulator; the oracle uses a heuristic policy and is not a mathematical optimum.
- Mean added delay is averaged over jobs that finish in the test period. Unfinished deferrable jobs count as deadline misses.

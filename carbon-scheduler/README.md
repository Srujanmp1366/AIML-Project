# Carbon aware scheduler

This small Python codebase compares hourly scheduling policies for ML jobs against an Indian-grid carbon-intensity trace. It consumes job, actual-grid, and forecast CSV files; it does not train a forecaster or fit an MEF regression. No sample data is included.

Requires Python 3.10+, pandas, numpy, and matplotlib. Install pytest to run the inline-fixture test suite.

## React interface

The frontend lives in `frontend/`. It calls the local Python API in `api.py`, which uses
`simulator.run_sim` for a single scenario and `run.iter_runs` for policy comparisons.
The interface shows emissions, completion and delay metrics, and per-job outcomes.
Web requests keep uploaded data and results in memory; they do not write CSV files.

If you do not already have `.venv`, create it with `python -m venv .venv`.
From `carbon-scheduler`, start the API in one PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
pip install numpy pandas
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
`data/jobs.csv`, `data/grid_truth.csv`, and a forecast CSV in `data/`, or you can
upload all three CSV files in the browser. Uploaded files are not saved to disk.

For a single-server build, run `npm run build` from `frontend/`, then run
`python api.py` from `carbon-scheduler/` and open `http://127.0.0.1:8000`.
The CLI commands below remain available separately and still write output files.

## Inputs

Put these files in `data/`:

- `jobs.csv`: `job_id`, `arrival`, `duration_hours`, `gpus`, `deferrable`
- `grid_truth.csv`: `time`, `actual_ci`, `mef`
- `forecasts_<model>.csv`: `issue_time`, `target_time`, `predicted_ci`

Grid and forecast timestamps must identify UTC hours; job arrival timestamps may be at any UTC time. Each forecast issue time needs exactly 24 target rows, from the first target hour through the following 23 hours. `actual_ci` and `predicted_ci` are in gCO₂/kWh; MEF uses the same units. Leave `mef` empty when marginal intensity is unavailable. In that case the simulator warns and reports marginal scores as missing.

The simulator treats `deferrable` values `true`, `1`, and `yes` as true (case insensitive). GPU demand is rounded up to a whole GPU.

## Run

From this directory, use an environment with the dependencies above, then:

```bash
python run.py naive
python run.py sweep --forecast data/forecasts_naive24.csv
python run.py sweep --forecast data/forecasts_naive24.csv --figures
pytest
```

`naive` writes `data/forecasts_naive24.csv`, using the actual CI from 24 hours before each target hour. `sweep` runs `run_now`, `threshold`, `forecast_greedy`, and `oracle` for `k = 6, 12, 24, 48` and capacities of 64, 32, 128, and unlimited GPUs. It writes `results/summary.csv` and one CSV per run under `results/per_job/`. `--figures` also writes `results/co2_saved_vs_k.png` and `results/capacity_savings.png`.

The test period defaults to the hourly starts represented by the forecast file. The `run_sim` API also accepts an optional UTC `start` and exclusive `end` timestamp.

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

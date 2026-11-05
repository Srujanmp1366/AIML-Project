"""Local JSON API for the React interface; simulation results stay in memory."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import math
import mimetypes
from pathlib import Path
import traceback
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

from run import POLICIES, iter_runs
from simulator import DATA_DIR, K_VALUES, run_sim


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
MAX_REQUEST_BYTES = 25_000_000


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _forecast_names() -> list[str]:
    return sorted(
        path.name for path in DATA_DIR.glob("*.csv")
        if path.name not in {"jobs.csv", "grid_truth.csv"}
    )


def _config() -> dict[str, object]:
    forecasts = _forecast_names()
    return {
        "policies": list(POLICIES),
        "k_values": K_VALUES,
        "capacities": [64, 32, 128, None],
        "forecasts": forecasts,
        "local_ready": (
            (DATA_DIR / "jobs.csv").is_file()
            and (DATA_DIR / "grid_truth.csv").is_file()
            and bool(forecasts)
        ),
    }


def _frames(payload: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = payload.get("source", "local")
    if source == "upload":
        uploads = payload.get("uploads")
        if not isinstance(uploads, dict):
            raise ValueError("Upload jobs, grid truth, and forecast CSV files")
        csv_texts = []
        for name in ("jobs", "grid_truth", "forecasts"):
            value = uploads.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {name} CSV")
            csv_texts.append(value)
        jobs_csv, truth_csv, forecast_csv = csv_texts
        return (
            pd.read_csv(StringIO(jobs_csv)),
            pd.read_csv(StringIO(truth_csv)),
            pd.read_csv(StringIO(forecast_csv)),
        )
    if source != "local":
        raise ValueError("source must be local or upload")

    forecast_name = payload.get("forecast_file")
    if not isinstance(forecast_name, str) or forecast_name not in _forecast_names():
        raise ValueError("Select an available forecast CSV")
    return (
        pd.read_csv(DATA_DIR / "jobs.csv"),
        pd.read_csv(DATA_DIR / "grid_truth.csv"),
        pd.read_csv(DATA_DIR / forecast_name),
    )


def _options(payload: dict[str, object]) -> tuple[str, int, int | None]:
    policy = payload.get("policy", "forecast_greedy")
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError("Select a valid policy")
    k = payload.get("k", 24)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k > 168:
        raise ValueError("k must be an integer between 1 and 168 hours")
    gpus = payload.get("gpus", 64)
    if gpus is not None and (isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0):
        raise ValueError("GPU capacity must be a positive integer or unlimited")
    return policy, k, gpus


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: object) -> None:
        encoded = json.dumps(_json_safe(body), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/config":
            self._send_json(200, _config())
            return
        if path.startswith("/api/"):
            self._send_json(404, {"error": "Unknown API endpoint"})
            return
        self._serve_frontend(path)

    def _serve_frontend(self, path: str) -> None:
        if not FRONTEND_DIST.is_dir():
            self._send_json(404, {"error": "Frontend not built; start Vite with npm run dev"})
            return
        root = FRONTEND_DIST.resolve()
        requested = (root / unquote(path).lstrip("/")).resolve()
        if not requested.is_relative_to(root):
            self._send_json(403, {"error": "Forbidden path"})
            return
        if not requested.is_file():
            if requested.suffix:
                self._send_json(404, {"error": "File not found"})
                return
            requested = root / "index.html"
        content = requested.read_bytes()
        mime_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/simulate", "/api/compare"}:
            self._send_json(404, {"error": "Unknown API endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "Request must be between 1 byte and 25 MB"})
                return
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be an object")
            policy, k, gpus = _options(payload)
            jobs, truth, forecasts = _frames(payload)
            if path == "/api/simulate":
                metrics, per_job = run_sim(
                    policy, k, gpus, jobs, truth, forecasts,
                )
                self._send_json(200, {
                    "metrics": metrics,
                    "jobs": per_job.to_dict("records"),
                })
            else:
                rows = [summary for summary, _ in iter_runs(
                    jobs, truth, forecasts,
                    policies=POLICIES,
                    k_values=(k,),
                    capacities=(gpus,),
                )]
                self._send_json(200, {"rows": rows})
        except (AssertionError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc) or "Invalid simulation input"})
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"error": "Simulation failed; check the API terminal for details"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local API for the carbon scheduler frontend")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"API running at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

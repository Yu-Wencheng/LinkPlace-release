#!/usr/bin/env python3
"""Persistent CPU queue for the six formal all-greedy seed-1000 runs."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("LINKPLACE_PYTHON", sys.executable))
RESULT_ROOT = Path(os.environ.get("LINKPLACE_RESULT_ROOT", PROJECT_ROOT / "outputs" / "formal"))
QUEUE_ROOT = RESULT_ROOT / "queue" / "all-greedy-seed1000"
BENCHMARKS = ("bigblue3", "adaptec4", "adaptec3", "adaptec2", "adaptec1", "bigblue1")
WORKERS = ("worker-0", "worker-1", "worker-2")
SEED = 1000
MAX_ATTEMPTS = 2


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def result_path(benchmark: str) -> Path:
    return RESULT_ROOT / "ablation" / "all-greedy" / benchmark / "seed-1000" / "result.json"


def terminal_status(benchmark: str):
    result = read_json(result_path(benchmark))
    if result and result.get("status") in {"complete", "failed"}:
        return result["status"]
    return None


def write_state(statuses, active, attempts, started_at, finished_at=None) -> None:
    atomic_json(
        QUEUE_ROOT / "queue-state.json",
        {
            "protocol": "all-greedy-seed1000",
            "seed": SEED,
            "benchmarks": list(BENCHMARKS),
            "started_at": started_at,
            "updated_at": now(),
            "finished_at": finished_at,
            "supervisor_pid": os.getpid(),
            "statuses": statuses,
            "attempts": attempts,
            "active": {
                benchmark: {"worker": item["worker"], "pid": item["process"].pid}
                for benchmark, item in active.items()
            },
        },
    )


def main() -> int:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    (QUEUE_ROOT / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started_at = now()
    statuses = {benchmark: (terminal_status(benchmark) or "pending") for benchmark in BENCHMARKS}
    attempts = {benchmark: 0 for benchmark in BENCHMARKS}
    pending = [benchmark for benchmark in BENCHMARKS if statuses[benchmark] == "pending"]
    active = {}
    write_state(statuses, active, attempts, started_at)

    while pending or active:
        for benchmark, item in list(active.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            del active[benchmark]
            outcome = terminal_status(benchmark)
            if outcome is not None:
                statuses[benchmark] = outcome
            elif attempts[benchmark] < MAX_ATTEMPTS:
                statuses[benchmark] = "pending"
                pending.append(benchmark)
            else:
                statuses[benchmark] = "process-failed"

        used_workers = {item["worker"] for item in active.values()}
        for worker in [worker for worker in WORKERS if worker not in used_workers]:
            if not pending:
                break
            benchmark = pending.pop(0)
            attempts[benchmark] += 1
            attempt = attempts[benchmark]
            log_path = QUEUE_ROOT / f"{benchmark}-attempt-{attempt:02d}.log"
            log_stream = log_path.open("ab", buffering=0)
            command = [
                str(PYTHON), "-m", "linkplace.runner",
                "--cache-root", "datasets/cache",
                "--result-root", str(RESULT_ROOT),
                "run-ablation", "all-greedy", benchmark,
                "--seed", str(SEED), "--episodes", "1000", "--device", "cpu",
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONHASHSEED": str(SEED),
                    "PYTHONPATH": ".",
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "8",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[benchmark] = {"worker": worker, "process": process, "log": log_stream}
            statuses[benchmark] = "running"

        write_state(statuses, active, attempts, started_at)
        time.sleep(5)

    write_state(statuses, active, attempts, started_at, finished_at=now())
    return 0 if all(status in {"complete", "failed"} for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

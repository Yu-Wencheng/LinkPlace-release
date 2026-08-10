#!/usr/bin/env python3
"""Persistent three-GPU queue for the six formal LinkPlace-M seed-1000 runs."""

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
AGGREGATE_PYTHON = Path(os.environ.get("LINKPLACE_AGGREGATE_PYTHON", str(PYTHON)))
RESULT_ROOT = Path(os.environ.get("LINKPLACE_RESULT_ROOT", PROJECT_ROOT / "outputs" / "formal"))
QUEUE_ROOT = RESULT_ROOT / "queue" / "linkplace-m-seed1000"
BENCHMARKS = ("bigblue3", "adaptec4", "adaptec3", "adaptec2", "adaptec1", "bigblue1")
GPUS = ("0", "1", "2")
SEED = 1000
EPISODES = 1000
MAX_ATTEMPTS = 3


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
    return RESULT_ROOT / "ablation" / "linkplace-m" / benchmark / "seed-1000" / "result.json"


def completed(benchmark: str) -> bool:
    result = read_json(result_path(benchmark))
    return bool(result and result.get("status") == "complete")


def write_state(statuses, active, attempts, started_at, finished_at=None) -> None:
    payload = {
        "protocol": "linkplace-m-seed1000",
        "seed": SEED,
        "episodes": EPISODES,
        "benchmarks": list(BENCHMARKS),
        "started_at": started_at,
        "updated_at": now(),
        "finished_at": finished_at,
        "supervisor_pid": os.getpid(),
        "statuses": statuses,
        "attempts": attempts,
        "active": {
            benchmark: {"gpu": item["gpu"], "pid": item["process"].pid}
            for benchmark, item in active.items()
        },
    }
    atomic_json(QUEUE_ROOT / "queue-state.json", payload)


def main() -> int:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    (QUEUE_ROOT / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started_at = now()
    statuses = {benchmark: ("complete" if completed(benchmark) else "pending") for benchmark in BENCHMARKS}
    attempts = {benchmark: 0 for benchmark in BENCHMARKS}
    pending = [benchmark for benchmark in BENCHMARKS if statuses[benchmark] != "complete"]
    active = {}
    write_state(statuses, active, attempts, started_at)

    while pending or active:
        for benchmark, item in list(active.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            del active[benchmark]
            if return_code == 0 and completed(benchmark):
                statuses[benchmark] = "complete"
            elif attempts[benchmark] < MAX_ATTEMPTS:
                statuses[benchmark] = "pending"
                pending.append(benchmark)
            else:
                statuses[benchmark] = "failed"

        used_gpus = {item["gpu"] for item in active.values()}
        for gpu in [gpu for gpu in GPUS if gpu not in used_gpus]:
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
                "run-ablation", "linkplace-m", benchmark,
                "--seed", str(SEED), "--episodes", str(EPISODES), "--device", "cuda",
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONHASHSEED": str(SEED),
                    "PYTHONPATH": ".",
                    "PYTHONUNBUFFERED": "1",
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
            active[benchmark] = {"gpu": gpu, "process": process, "log": log_stream}
            statuses[benchmark] = "running"

        write_state(statuses, active, attempts, started_at)
        time.sleep(10)

    finished_at = now()
    write_state(statuses, active, attempts, started_at, finished_at=finished_at)
    if all(status == "complete" for status in statuses.values()):
        aggregate_log = (QUEUE_ROOT / "aggregate.log").open("ab", buffering=0)
        return_code = subprocess.call(
            [str(AGGREGATE_PYTHON), "tools/aggregate_paper_results.py", "--result-root", str(RESULT_ROOT)],
            cwd=str(PROJECT_ROOT),
            stdout=aggregate_log,
            stderr=subprocess.STDOUT,
        )
        aggregate_log.close()
        atomic_json(
            QUEUE_ROOT / "aggregate-state.json",
            {"completed_at": now(), "return_code": return_code},
        )
        return return_code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

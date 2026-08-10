#!/usr/bin/env python3
"""Persistent queue for the seed-1000 ISPD2005 224/448 grid ablation."""

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
GRID224_ROOT = RESULT_ROOT / "grid-ablation" / "grid-224"
QUEUE_ROOT = RESULT_ROOT / "queue" / "grid-ablation-seed1000"
LINKPLACE_M_STATE = RESULT_ROOT / "queue" / "linkplace-m-seed1000" / "queue-state.json"
BENCHMARKS = (
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue2",
    "bigblue3",
    "bigblue4",
)
GPU_ORDER = ("0", "1", "2")
CPU_WORKERS = ("cpu-0", "cpu-1", "cpu-2")
SEED = 1000
EPISODES = 1000
MAX_GPU_ATTEMPTS = 3
MAX_CPU_ATTEMPTS = 2


def task(grid: int, variant: str, benchmark: str, resource: str):
    return {
        "id": "g{}-{}-{}".format(grid, variant, benchmark),
        "grid": grid,
        "variant": variant,
        "benchmark": benchmark,
        "resource": resource,
    }


CPU_TASKS = tuple(
    [task(448, "all-greedy", benchmark, "cpu") for benchmark in ("bigblue4", "bigblue2")]
    + [task(224, "all-greedy", benchmark, "cpu") for benchmark in reversed(BENCHMARKS)]
)

GPU_TASKS = tuple(
    [
        task(448, "linkplace-m", "bigblue4", "gpu"),
        task(448, "linkplace-m", "bigblue2", "gpu"),
    ]
    + [
        task(224, variant, benchmark, "gpu")
        for benchmark in reversed(BENCHMARKS)
        for variant in ("linkplace-c", "linkplace-m")
    ]
)
TASKS = CPU_TASKS + GPU_TASKS
TASK_BY_ID = {item["id"]: item for item in TASKS}


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


def task_root(item) -> Path:
    return RESULT_ROOT if item["grid"] == 448 else GRID224_ROOT


def result_path(item) -> Path:
    root = task_root(item)
    if item["variant"] == "linkplace-c":
        return root / "main" / item["benchmark"] / "seed-1000" / "result.json"
    return root / "ablation" / item["variant"] / item["benchmark"] / "seed-1000" / "result.json"


def terminal_status(item):
    result = read_json(result_path(item))
    if not result:
        return None
    status = result.get("status")
    if item["variant"] == "all-greedy" and status in {"complete", "failed"}:
        return status
    return "complete" if status == "complete" else None


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def externally_reserved_gpus():
    state = read_json(LINKPLACE_M_STATE) or {}
    reserved = {}
    for benchmark, item in state.get("active", {}).items():
        if pid_alive(item.get("pid")):
            reserved[str(item.get("gpu"))] = {
                "protocol": state.get("protocol"),
                "benchmark": benchmark,
                "pid": item.get("pid"),
            }
    return reserved


def command_for(item):
    common = [
        str(PYTHON),
        "-m",
        "linkplace.runner",
        "--cache-root",
        "datasets/cache",
        "--result-root",
        str(task_root(item)),
    ]
    if item["variant"] == "linkplace-c":
        return common + [
            "run-main",
            item["benchmark"],
            "--seed",
            str(SEED),
            "--episodes",
            str(EPISODES),
            "--grid",
            str(item["grid"]),
            "--device",
            "cuda",
            "--max-flow-restarts",
            "0",
        ]
    return common + [
        "run-ablation",
        item["variant"],
        item["benchmark"],
        "--seed",
        str(SEED),
        "--episodes",
        str(EPISODES),
        "--grid",
        str(item["grid"]),
        "--device",
        "cpu" if item["resource"] == "cpu" else "cuda",
    ]


def write_state(statuses, attempts, active_cpu, active_gpu, started_at, finished_at=None):
    external = externally_reserved_gpus()
    atomic_json(
        QUEUE_ROOT / "queue-state.json",
        {
            "protocol": "ispd2005-grid-ablation-seed1000",
            "seed": SEED,
            "episodes": EPISODES,
            "tasks": list(TASKS),
            "started_at": started_at,
            "updated_at": now(),
            "finished_at": finished_at,
            "supervisor_pid": os.getpid(),
            "statuses": statuses,
            "attempts": attempts,
            "external_reserved_gpus": external,
            "active_cpu": {
                task_id: {"worker": item["slot"], "pid": item["process"].pid}
                for task_id, item in active_cpu.items()
            },
            "active_gpu": {
                task_id: {"gpu": item["slot"], "pid": item["process"].pid}
                for task_id, item in active_gpu.items()
            },
        },
    )


def launch(item, slot, attempt):
    log_path = QUEUE_ROOT / "{}-attempt-{:02d}.log".format(item["id"], attempt)
    log_stream = log_path.open("ab", buffering=0)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if item["resource"] == "cpu" else slot,
            "PYTHONHASHSEED": str(SEED),
            "PYTHONPATH": ".",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if item["resource"] == "cpu":
        environment["OMP_NUM_THREADS"] = "8"
    process = subprocess.Popen(
        command_for(item),
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"slot": slot, "process": process, "log": log_stream}


def reap(active, pending, statuses, attempts, maximum_attempts):
    for task_id, running in list(active.items()):
        return_code = running["process"].poll()
        if return_code is None:
            continue
        running["log"].close()
        del active[task_id]
        item = TASK_BY_ID[task_id]
        outcome = terminal_status(item)
        if outcome is not None:
            statuses[task_id] = outcome
        elif attempts[task_id] < maximum_attempts:
            statuses[task_id] = "pending"
            pending.append(task_id)
        else:
            statuses[task_id] = "process-failed"


def main() -> int:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    (QUEUE_ROOT / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started_at = now()
    statuses = {item["id"]: (terminal_status(item) or "pending") for item in TASKS}
    attempts = {item["id"]: 0 for item in TASKS}
    pending_cpu = [item["id"] for item in CPU_TASKS if statuses[item["id"]] == "pending"]
    pending_gpu = [item["id"] for item in GPU_TASKS if statuses[item["id"]] == "pending"]
    active_cpu = {}
    active_gpu = {}
    write_state(statuses, attempts, active_cpu, active_gpu, started_at)

    while pending_cpu or pending_gpu or active_cpu or active_gpu:
        reap(active_cpu, pending_cpu, statuses, attempts, MAX_CPU_ATTEMPTS)
        reap(active_gpu, pending_gpu, statuses, attempts, MAX_GPU_ATTEMPTS)

        used_workers = {item["slot"] for item in active_cpu.values()}
        for worker in [value for value in CPU_WORKERS if value not in used_workers]:
            if not pending_cpu:
                break
            task_id = pending_cpu.pop(0)
            attempts[task_id] += 1
            active_cpu[task_id] = launch(TASK_BY_ID[task_id], worker, attempts[task_id])
            statuses[task_id] = "running"

        external = externally_reserved_gpus()
        used_gpus = {item["slot"] for item in active_gpu.values()}
        available_gpus = [gpu for gpu in GPU_ORDER if gpu not in used_gpus and gpu not in external]
        for gpu in available_gpus:
            if not pending_gpu:
                break
            task_id = pending_gpu.pop(0)
            attempts[task_id] += 1
            active_gpu[task_id] = launch(TASK_BY_ID[task_id], gpu, attempts[task_id])
            statuses[task_id] = "running"

        write_state(statuses, attempts, active_cpu, active_gpu, started_at)
        time.sleep(10)

    finished_at = now()
    write_state(statuses, attempts, active_cpu, active_gpu, started_at, finished_at=finished_at)
    aggregate_log = (QUEUE_ROOT / "aggregate.log").open("ab", buffering=0)
    return_code = subprocess.call(
        [
            str(AGGREGATE_PYTHON),
            "tools/aggregate_grid_ablation.py",
            "--result-root",
            str(RESULT_ROOT),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=aggregate_log,
        stderr=subprocess.STDOUT,
    )
    aggregate_log.close()
    atomic_json(
        QUEUE_ROOT / "aggregate-state.json",
        {"completed_at": now(), "return_code": return_code},
    )
    acceptable = {"complete", "failed"}
    return return_code if all(status in acceptable for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

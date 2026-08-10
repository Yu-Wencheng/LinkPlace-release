#!/usr/bin/env python3
"""Persistent five-seed queue for dual-grid LinkPlace variants and ICCAD2015."""

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
QUEUE_ROOT = RESULT_ROOT / "queue" / "paper-extension-multiseed-v2"
SEEDS = (999, 1000, 1001, 1002, 1003)
ISPD2005 = (
    "adaptec1", "adaptec2", "adaptec3", "adaptec4",
    "bigblue1", "bigblue2", "bigblue3", "bigblue4",
)
ICCAD2015 = (
    "superblue1", "superblue3", "superblue4", "superblue5",
    "superblue7", "superblue10", "superblue16", "superblue18",
)
GPU_ORDER = ("0", "1", "2")
CPU_WORKERS = ("cpu-0", "cpu-1", "cpu-2", "cpu-3")
EPISODES = 1000
MAX_GPU_ATTEMPTS = 3
MAX_CPU_ATTEMPTS = 2


def task(grid, variant, benchmark, seed, resource, suite="ispd2005"):
    return {
        "id": f"{suite}-g{grid}-{variant}-{benchmark}-seed-{seed}",
        "suite": suite,
        "grid": grid,
        "variant": variant,
        "benchmark": benchmark,
        "seed": seed,
        "resource": resource,
    }


CPU_TASKS = tuple(
    task(grid, "all-greedy", benchmark, seed, "cpu")
    for seed in SEEDS
    for grid in (448, 224)
    for benchmark in reversed(ISPD2005)
)
GPU_TASKS = tuple(
    [
        task(grid, variant, benchmark, seed, "gpu")
        for seed in SEEDS
        for grid, variant in ((224, "linkplace-c"), (224, "linkplace-m"), (448, "linkplace-m"))
        for benchmark in reversed(ISPD2005)
    ]
    + [
        task(448, "linkplace-m", benchmark, seed, "gpu", suite="iccad2015")
        for seed in SEEDS
        for benchmark in reversed(ICCAD2015)
    ]
)
TASKS = CPU_TASKS + GPU_TASKS
TASK_BY_ID = {item["id"]: item for item in TASKS}


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def task_root(item):
    return RESULT_ROOT if item["grid"] == 448 else GRID224_ROOT


def result_path(item):
    root = task_root(item)
    category = "main" if item["variant"] == "linkplace-c" else "ablation/" + item["variant"]
    return root / category / item["benchmark"] / f"seed-{item['seed']}" / "result.json"


def terminal_status(item):
    result = read_json(result_path(item))
    if not result:
        return None
    status = result.get("status")
    if item["variant"] == "all-greedy" and status in {"complete", "failed"}:
        return status
    if status != "complete":
        return None
    if result.get("seed") != item["seed"] or result.get("grid") != item["grid"]:
        return None
    metrics = result.get("metrics", {})
    return "complete" if metrics.get("legal") is True else None


def command_for(item):
    common = [
        str(PYTHON), "-m", "linkplace.runner",
        "--cache-root", "datasets/cache",
        "--result-root", str(task_root(item)),
    ]
    if item["variant"] == "linkplace-c":
        return common + [
            "run-main", item["benchmark"],
            "--seed", str(item["seed"]),
            "--episodes", str(EPISODES),
            "--grid", str(item["grid"]),
            "--device", "cuda",
            "--max-flow-restarts", "0",
        ]
    return common + [
        "run-ablation", item["variant"], item["benchmark"],
        "--seed", str(item["seed"]),
        "--episodes", str(EPISODES),
        "--grid", str(item["grid"]),
        "--device", "cpu" if item["resource"] == "cpu" else "cuda",
    ]


def write_state(statuses, attempts, active_cpu, active_gpu, started_at, finished_at=None):
    atomic_json(
        QUEUE_ROOT / "queue-state.json",
        {
            "protocol": "paper-extension-multiseed-v2",
            "seeds": list(SEEDS),
            "episodes": EPISODES,
            "tasks": list(TASKS),
            "started_at": started_at,
            "updated_at": now(),
            "finished_at": finished_at,
            "supervisor_pid": os.getpid(),
            "statuses": statuses,
            "attempts": attempts,
            "active_cpu": {
                task_id: {"worker": value["slot"], "pid": value["process"].pid}
                for task_id, value in active_cpu.items()
            },
            "active_gpu": {
                task_id: {"gpu": value["slot"], "pid": value["process"].pid}
                for task_id, value in active_gpu.items()
            },
        },
    )


def launch(item, slot, attempt):
    log_path = QUEUE_ROOT / f"{item['id']}-attempt-{attempt:02d}.log"
    log_stream = log_path.open("ab", buffering=0)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if item["resource"] == "cpu" else slot,
            "PYTHONHASHSEED": str(item["seed"]),
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
        if running["process"].poll() is None:
            continue
        running["log"].close()
        del active[task_id]
        outcome = terminal_status(TASK_BY_ID[task_id])
        if outcome is not None:
            statuses[task_id] = outcome
        elif attempts[task_id] < maximum_attempts:
            statuses[task_id] = "pending"
            pending.append(task_id)
        else:
            statuses[task_id] = "process-failed"


def main():
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    (QUEUE_ROOT / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started_at = now()
    statuses = {item["id"]: (terminal_status(item) or "pending") for item in TASKS}
    attempts = {item["id"]: 0 for item in TASKS}
    pending_cpu = [item["id"] for item in CPU_TASKS if statuses[item["id"]] == "pending"]
    pending_gpu = [item["id"] for item in GPU_TASKS if statuses[item["id"]] == "pending"]
    active_cpu, active_gpu = {}, {}
    write_state(statuses, attempts, active_cpu, active_gpu, started_at)

    while pending_cpu or pending_gpu or active_cpu or active_gpu:
        reap(active_cpu, pending_cpu, statuses, attempts, MAX_CPU_ATTEMPTS)
        reap(active_gpu, pending_gpu, statuses, attempts, MAX_GPU_ATTEMPTS)

        used_cpu = {value["slot"] for value in active_cpu.values()}
        for worker in [value for value in CPU_WORKERS if value not in used_cpu]:
            if not pending_cpu:
                break
            task_id = pending_cpu.pop(0)
            attempts[task_id] += 1
            active_cpu[task_id] = launch(TASK_BY_ID[task_id], worker, attempts[task_id])
            statuses[task_id] = "running"

        used_gpu = {value["slot"] for value in active_gpu.values()}
        for gpu in [value for value in GPU_ORDER if value not in used_gpu]:
            if not pending_gpu:
                break
            task_id = pending_gpu.pop(0)
            attempts[task_id] += 1
            active_gpu[task_id] = launch(TASK_BY_ID[task_id], gpu, attempts[task_id])
            statuses[task_id] = "running"

        write_state(statuses, attempts, active_cpu, active_gpu, started_at)
        time.sleep(10)

    finished_at = now()
    write_state(statuses, attempts, active_cpu, active_gpu, started_at, finished_at)
    aggregate_log = (QUEUE_ROOT / "aggregate.log").open("ab", buffering=0)
    return_code = subprocess.call(
        [
            str(AGGREGATE_PYTHON), "tools/aggregate_multiseed_extension.py",
            "--result-root", str(RESULT_ROOT),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=aggregate_log,
        stderr=subprocess.STDOUT,
    )
    aggregate_log.close()
    atomic_json(QUEUE_ROOT / "aggregate-state.json", {"completed_at": now(), "return_code": return_code})
    acceptable = {"complete", "failed"}
    return return_code if all(value in acceptable for value in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

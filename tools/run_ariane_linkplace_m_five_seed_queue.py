#!/usr/bin/env python3
"""Persistent formal five-seed LinkPlace-M queue for Ariane."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("LINKPLACE_PYTHON", sys.executable))
RESULT_ROOT = Path(os.environ.get("LINKPLACE_RESULT_ROOT", PROJECT_ROOT / "outputs" / "formal"))
QUEUE_ROOT = RESULT_ROOT / "queue" / "ariane-linkplace-m-five-seed-v2"
SEEDS = (999, 1000, 1001, 1002, 1003)
GPU_ORDER = ("0", "1", "2")
EPISODES = 1000
GRID = 448
MAX_ATTEMPTS = 3


def task(seed):
    return {
        "id": "iccad2015-g448-linkplace-m-ariane-seed-{}".format(seed),
        "suite": "iccad2015",
        "grid": GRID,
        "variant": "linkplace-m",
        "benchmark": "ariane",
        "seed": seed,
        "resource": "gpu",
    }


TASKS = tuple(task(seed) for seed in SEEDS)
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


def result_path(item):
    return (
        RESULT_ROOT
        / "ablation"
        / "linkplace-m"
        / "ariane"
        / "seed-{}".format(item["seed"])
        / "result.json"
    )


def terminal_status(item):
    result = read_json(result_path(item))
    if not result:
        return None
    if result.get("seed") != item["seed"] or result.get("grid") != GRID:
        return None
    status = result.get("status")
    if status == "failed":
        return "failed"
    metrics = result.get("metrics") or {}
    if status == "complete" and metrics.get("legal") is True:
        return "complete"
    return None


def matching_runner_pids():
    matches = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        required = (
            "linkplace.runner",
            "run-ablation",
            "linkplace-m",
            "ariane",
        )
        if all(token in command for token in required):
            matches.append({"pid": int(entry.name), "command": command.strip()})
    return matches


def command_for(item):
    return [
        str(PYTHON),
        "-m",
        "linkplace.runner",
        "--cache-root",
        "datasets/cache",
        "--result-root",
        str(RESULT_ROOT),
        "run-ablation",
        "linkplace-m",
        "ariane",
        "--seed",
        str(item["seed"]),
        "--episodes",
        str(EPISODES),
        "--grid",
        str(GRID),
        "--device",
        "cuda",
    ]


def write_state(statuses, attempts, active_gpu, started_at, finished_at=None, note=None):
    atomic_json(
        QUEUE_ROOT / "queue-state.json",
        {
            "protocol": "ariane-linkplace-m-five-seed-v2",
            "seeds": list(SEEDS),
            "episodes": EPISODES,
            "grid": GRID,
            "tasks": list(TASKS),
            "started_at": started_at,
            "updated_at": now(),
            "finished_at": finished_at,
            "supervisor_pid": os.getpid(),
            "statuses": statuses,
            "attempts": attempts,
            "active_gpu": {
                task_id: {"gpu": value["slot"], "pid": value["process"].pid}
                for task_id, value in active_gpu.items()
            },
            "note": note,
        },
    )


def launch(item, slot, attempt):
    log_path = QUEUE_ROOT / "{}-attempt-{:02d}.log".format(item["id"], attempt)
    log_stream = log_path.open("ab", buffering=0)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": slot,
            "PYTHONHASHSEED": str(item["seed"]),
            "PYTHONPATH": ".",
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        command_for(item),
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"slot": slot, "process": process, "log": log_stream}


def reap(active, pending, statuses, attempts):
    for task_id, running in list(active.items()):
        if running["process"].poll() is None:
            continue
        running["log"].close()
        del active[task_id]
        outcome = terminal_status(TASK_BY_ID[task_id])
        if outcome is not None:
            statuses[task_id] = outcome
        elif attempts[task_id] < MAX_ATTEMPTS:
            statuses[task_id] = "pending"
            pending.append(task_id)
        else:
            statuses[task_id] = "process-failed"


def main():
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (QUEUE_ROOT / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 2

    started_at = now()
    statuses = {item["id"]: (terminal_status(item) or "pending") for item in TASKS}
    attempts = {item["id"]: 0 for item in TASKS}
    active_gpu = {}
    existing = matching_runner_pids()
    if existing:
        write_state(
            statuses,
            attempts,
            active_gpu,
            started_at,
            note="blocked-existing-matching-runner: {}".format(existing),
        )
        return 3

    pending = [item["id"] for item in TASKS if statuses[item["id"]] == "pending"]
    (QUEUE_ROOT / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    write_state(statuses, attempts, active_gpu, started_at)

    while pending or active_gpu:
        reap(active_gpu, pending, statuses, attempts)
        used_gpu = {value["slot"] for value in active_gpu.values()}
        for gpu in [value for value in GPU_ORDER if value not in used_gpu]:
            if not pending:
                break
            task_id = pending.pop(0)
            attempts[task_id] += 1
            active_gpu[task_id] = launch(TASK_BY_ID[task_id], gpu, attempts[task_id])
            statuses[task_id] = "running"
        write_state(statuses, attempts, active_gpu, started_at)
        time.sleep(10)

    finished_at = now()
    write_state(statuses, attempts, active_gpu, started_at, finished_at=finished_at)
    acceptable = {"complete", "failed"}
    return 0 if all(value in acceptable for value in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

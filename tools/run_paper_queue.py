#!/usr/bin/env python3
"""Persistent three-GPU supervisor for the approved LinkPlace paper runs.

The supervisor is safe to restart.  It adopts still-running children recorded in
queue-state.json, skips completed result.json files, and retains a separate log
for every process attempt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (999, 1000, 1001, 1002, 1003)
MAIN_BENCHMARKS = (
    "adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3",
    "bigblue2", "bigblue4", "ariane", "superblue1", "superblue3", "superblue4",
    "superblue5", "superblue7", "superblue10", "superblue16", "superblue18",
)
ABLATION_BENCHMARKS = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3")


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def append_event(path: Path, event, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(project_root: Path, result_root: Path, args):
    files = []
    for relative_root in ("linkplace", "tools", "tests"):
        root = project_root / relative_root
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".sh"}:
                continue
            files.append(
                {
                    "path": str(path.relative_to(project_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    versions = {}
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception as error:
        versions["numpy_error"] = repr(error)
    try:
        import torch

        versions.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as error:
        versions["torch_error"] = repr(error)
    atomic_json(
        result_root / "provenance" / "code-and-runtime.json",
        {
            "created_at": now(),
            "project_root": str(project_root),
            "result_root": str(result_root),
            "python": sys.executable,
            "arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
            "versions": versions,
            "files": files,
        },
    )


def cache_key(name):
    if name in {"bigblue2", "bigblue4"}:
        return name + "-1024"
    if name.startswith("superblue"):
        return name + "-512"
    return name


def build_jobs(result_root: Path, episodes: int, include_ablation: bool):
    jobs = [
            {
                "id": "regression-gate__run-blank-large__adaptec1__seed-999",
                "kind": "regression-gate",
                "variant": "a1-448-historical-range",
                "benchmark": "adaptec1",
                "seed": 999,
                "required_cache": "adaptec1.json.gz",
                "result_path": str(
                    result_root
                    / "supplementary"
                    / "blank-canvas"
                    / "adaptec1"
                    / "largest-component"
                    / "seed-999"
                    / "result.json"
                ),
                "arguments": [
                    "-m", "linkplace.runner", "--cache-root", "datasets/cache",
                    "--result-root", str(result_root), "run-blank-large", "adaptec1",
                    "--seed", "999", "--episodes", str(episodes), "--device", "cuda",
                ],
            }
        ]
    # A seed-major order puts different designs on the three GPUs immediately.
    for seed in SEEDS:
        for benchmark in MAIN_BENCHMARKS:
            jobs.append(
                {
                    "id": "main__{}__seed-{}".format(benchmark, seed),
                    "kind": "main",
                    "benchmark": benchmark,
                    "seed": seed,
                    "required_cache": cache_key(benchmark) + ".json.gz",
                    "result_path": str(result_root / "main" / benchmark / "seed-{}".format(seed) / "result.json"),
                    "arguments": [
                        "-m", "linkplace.runner", "--cache-root", "datasets/cache",
                        "--result-root", str(result_root), "run-main", benchmark,
                        "--seed", str(seed), "--episodes", str(episodes), "--device", "cuda",
                    ],
                }
            )
    if include_ablation:
        for variant in ("linkplace-m", "all-greedy"):
            for benchmark in ABLATION_BENCHMARKS:
                jobs.append(
                    {
                        "id": "ablation__{}__{}__seed-999".format(variant, benchmark),
                        "kind": "ablation",
                        "variant": variant,
                        "benchmark": benchmark,
                        "seed": 999,
                        "required_cache": cache_key(benchmark) + ".json.gz",
                        "result_path": str(
                            result_root / "ablation" / variant / benchmark / "seed-999" / "result.json"
                        ),
                        "arguments": [
                            "-m", "linkplace.runner", "--cache-root", "datasets/cache",
                            "--result-root", str(result_root), "run-ablation", variant, benchmark,
                            "--seed", "999", "--episodes", str(episodes), "--device", "cuda",
                        ],
                    }
                )
    return jobs


def result_complete(job):
    result = read_json(Path(job["result_path"]), {})
    return result.get("status") == "complete"


def ablation_finished(job):
    if job.get("kind") != "ablation":
        return False
    result = read_json(Path(job["result_path"]), {})
    return result.get("status") in {"complete", "failed"}


def process_alive(pid, job_id):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        command = Path("/proc") / str(pid) / "cmdline"
        if command.exists():
            text = command.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
            identifying_parts = job_id.split("__")
            return all(part.replace("seed-", "") in text for part in identifying_parts[1:])
        return True
    except (OSError, ValueError):
        return False


def summarize(records):
    counts = {}
    for record in records.values():
        state = record.get("status", "pending")
        counts[state] = counts.get(state, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Run all approved LinkPlace paper experiments")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--result-root", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--cache-root", type=Path, default=Path("datasets/cache"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-process-restarts", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument(
        "--external-regression-pid",
        type=int,
        default=None,
        help="adopt an already-running regression gate and use the other GPUs immediately",
    )
    parser.add_argument("--external-regression-gpu", default="0")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    result_root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    queue_root = result_root / "queue"
    queue_root.mkdir(parents=True, exist_ok=True)
    write_provenance(project_root, result_root, args)
    lock_stream = (queue_root / "queue.lock").open("w")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another queue supervisor already holds {}".format(queue_root / "queue.lock"), file=sys.stderr)
        return 2

    state_path = queue_root / "queue-state.json"
    event_path = queue_root / "events.jsonl"
    prior = read_json(state_path, {})
    prior_records = prior.get("jobs", {}) if isinstance(prior, dict) else {}
    jobs = build_jobs(
        result_root,
        args.episodes,
        not args.main_only,
    )
    records = {}
    for sequence, job in enumerate(jobs):
        previous = prior_records.get(job["id"], {})
        record = {**job, **previous, "sequence": sequence}
        if result_complete(job):
            record["status"] = "complete"
        elif ablation_finished(job):
            record["status"] = "failed"
            record["experiment_status"] = read_json(Path(job["result_path"]), {}).get("status")
        elif (
            job["kind"] == "regression-gate"
            and args.external_regression_pid is not None
            and process_alive(args.external_regression_pid, job["id"])
        ):
            record.update(
                {
                    "status": "running",
                    "pid": int(args.external_regression_pid),
                    "gpu": str(args.external_regression_gpu),
                    "adopted": True,
                    "external": True,
                    "attempts": max(1, int(record.get("attempts", 0))),
                }
            )
        elif record.get("status") == "running" and process_alive(record.get("pid"), job["id"]):
            record["status"] = "running"
            record["adopted"] = True
        else:
            record["status"] = "pending"
            record.pop("pid", None)
            record.pop("gpu", None)
        records[job["id"]] = record

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    active = {}
    stopping = {"value": False}

    def request_stop(signum, frame):
        del frame
        stopping["value"] = True
        append_event(event_path, "supervisor-stop-requested", signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    for record in records.values():
        if record.get("status") == "running":
            active[record["id"]] = {"process": None, "gpu": str(record["gpu"]), "log": None}
            append_event(event_path, "job-adopted", job_id=record["id"], pid=record.get("pid"), gpu=record.get("gpu"))

    append_event(event_path, "supervisor-started", pid=os.getpid(), jobs=len(records), gpus=gpus)

    def persist():
        atomic_json(
            state_path,
            {
                "updated_at": now(),
                "supervisor_pid": os.getpid(),
                "result_root": str(result_root),
                "counts": summarize(records),
                "jobs": records,
            },
        )

    persist()
    while not stopping["value"]:
        changed = False
        for job_id, runtime in list(active.items()):
            record = records[job_id]
            process = runtime["process"]
            if process is None:
                finished = not process_alive(record.get("pid"), job_id)
                return_code = None
            else:
                return_code = process.poll()
                finished = return_code is not None
            if not finished:
                continue
            if runtime["log"] is not None:
                runtime["log"].close()
            del active[job_id]
            if record.get("kind") == "regression-gate":
                subprocess.run(
                    [
                        args.python,
                        "tools/finalize_regression_gate.py",
                        "--result-root",
                        str(result_root),
                    ],
                    cwd=str(project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if result_complete(record):
                record.update({"status": "complete", "finished_at": now(), "return_code": return_code})
                append_event(event_path, "job-complete", job_id=job_id, gpu=runtime["gpu"], return_code=return_code)
            elif ablation_finished(record):
                record.update(
                    {
                        "status": "failed",
                        "experiment_status": "failed",
                        "finished_at": now(),
                        "return_code": return_code,
                    }
                )
                append_event(
                    event_path,
                    "ablation-finished-with-failure",
                    job_id=job_id,
                    gpu=runtime["gpu"],
                    return_code=return_code,
                )
            else:
                attempts = int(record.get("attempts", 0))
                exhausted = args.max_process_restarts > 0 and attempts >= args.max_process_restarts + 1
                record.update(
                    {
                        "status": "failed" if exhausted else "pending",
                        "last_failure_at": now(),
                        "return_code": return_code,
                    }
                )
                append_event(
                    event_path,
                    "job-failed" if exhausted else "job-requeued",
                    job_id=job_id,
                    gpu=runtime["gpu"],
                    return_code=return_code,
                    attempts=attempts,
                )
            record.pop("pid", None)
            record.pop("gpu", None)
            changed = True

        used_gpus = {str(item["gpu"]) for item in active.values()}
        available_gpus = [gpu for gpu in gpus if gpu not in used_gpus]
        gate = records["regression-gate__run-blank-large__adaptec1__seed-999"]
        gate_ready = gate.get("status") in {"running", "complete"}
        for record in sorted(records.values(), key=lambda item: item["sequence"]):
            if not available_gpus:
                break
            if record.get("status") != "pending":
                continue
            if record["kind"] != "regression-gate" and not gate_ready:
                continue
            if not (cache_root / record["required_cache"]).exists():
                continue
            gpu = available_gpus.pop(0)
            attempt = int(record.get("attempts", 0)) + 1
            log_path = queue_root / "logs" / record["id"] / "attempt-{:03d}.log".format(attempt)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("a", encoding="utf-8")
            command = [args.python] + record["arguments"]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONHASHSEED"] = str(record["seed"])
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            record.update(
                {
                    "status": "running",
                    "attempts": attempt,
                    "pid": process.pid,
                    "gpu": gpu,
                    "started_at": now(),
                    "log_path": str(log_path),
                    "command": command,
                }
            )
            active[record["id"]] = {"process": process, "gpu": gpu, "log": log_stream}
            append_event(event_path, "job-started", job_id=record["id"], attempt=attempt, pid=process.pid, gpu=gpu)
            changed = True

        if all(record.get("status") in {"complete", "failed"} for record in records.values()):
            persist()
            append_event(event_path, "supervisor-complete", counts=summarize(records))
            return 0 if all(record.get("status") == "complete" for record in records.values()) else 1
        if changed:
            persist()
        time.sleep(max(1.0, args.poll_seconds))

    persist()
    append_event(event_path, "supervisor-stopped", active_jobs=sorted(active))
    # Children intentionally keep running; a restarted supervisor adopts them.
    return 130


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent watcher for baseline archival, DREAMPlace, and final tables."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SEEDS = (999, 1000, 1001, 1002, 1003)
BENCHMARKS = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3", "bigblue2", "bigblue4")


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, event, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "event": event, **fields}, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def alive(pid, identifying_parts=()):
    try:
        os.kill(int(pid), 0)
        command = Path("/proc") / str(pid) / "cmdline"
        if command.exists():
            text = command.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
            return all(str(item) in text for item in identifying_parts)
        return True
    except (OSError, TypeError, ValueError):
        return False


def queue_finished(result_root: Path):
    state = read_json(result_root / "queue" / "queue-state.json", {})
    jobs = state.get("jobs", {})
    return bool(jobs) and all(
        item.get("status") == "complete"
        or (item.get("kind") == "ablation" and item.get("status") == "failed")
        for item in jobs.values()
    )


def dream_result(result_root: Path, benchmark: str, seed: int):
    return result_root / "dreamplace" / benchmark / "seed-{}".format(seed) / "result.json"


def dream_done(path: Path):
    result = read_json(path, {})
    return result.get("status") in {"complete", "unavailable"}


def best_main_seed(result_root: Path, benchmark: str):
    rows = []
    for seed in SEEDS:
        path = result_root / "main" / benchmark / "seed-{}".format(seed) / "result.json"
        result = read_json(path, {})
        if result.get("status") != "complete" or not result.get("metrics", {}).get("legal"):
            return None
        hpwl = result["metrics"].get("comp_res_hpwl", result["metrics"].get("macro_hpwl"))
        rows.append((float(hpwl), seed, path))
    return min(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=None,
        help="Optional official-baseline run directory to archive.",
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    result_root = args.result_root.resolve()
    post_root = result_root / "postprocess"
    post_root.mkdir(parents=True, exist_ok=True)
    lock = (post_root / "postprocess.lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another postprocess watcher is active", file=sys.stderr)
        return 2

    state_path = post_root / "state.json"
    events = post_root / "events.jsonl"
    stopped = {"value": False}

    def stop(signum, frame):
        del frame
        stopped["value"] = True
        append_event(events, "stop-requested", signal=signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    baseline_index = result_root / "baselines" / "available_official_reproductions" / "index.json"
    if args.baseline_source is not None and not baseline_index.exists():
        log_path = post_root / "baseline-archive.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    args.python,
                    "tools/archive_available_baselines.py",
                    "--source",
                    str(args.baseline_source.resolve()),
                    "--destination",
                    str(baseline_index.parent),
                ],
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        append_event(events, "baseline-archive-finished", return_code=completed.returncode, log=str(log_path))

    prior = read_json(state_path, {})
    prior_jobs = prior.get("jobs", {}) if isinstance(prior, dict) else {}
    records = {}
    for sequence, benchmark in enumerate(BENCHMARKS):
        job_id = benchmark
        selected = best_main_seed(result_root, benchmark)
        seed = None if selected is None else selected[1]
        result_path = None if seed is None else str(dream_result(result_root, benchmark, seed))
        record = {
            "id": job_id,
            "benchmark": benchmark,
            "seed": seed,
            "sequence": sequence,
            "selection_metric": "minimum comp_res_hpwl among five complete seeds",
            "result_path": result_path,
            **prior_jobs.get(job_id, {}),
        }
        if selected is not None:
            record["seed"] = selected[1]
            record["selected_comp_res_hpwl"] = selected[0]
            record["result_path"] = str(dream_result(result_root, benchmark, selected[1]))
        if record["result_path"] and dream_done(Path(record["result_path"])):
            record["status"] = read_json(Path(record["result_path"]), {}).get("status")
        elif (
            record.get("status") == "running"
            and record.get("seed") is not None
            and alive(record.get("pid"), ("run_dreamplace_mixed.py", benchmark, str(record["seed"])))
        ):
            record["status"] = "running"
            record["adopted"] = True
        else:
            record["status"] = "pending"
            record.pop("pid", None)
            record.pop("gpu", None)
        records[job_id] = record

    active = {}
    for record in records.values():
        if record["status"] == "running":
            active[record["id"]] = {"process": None, "gpu": str(record["gpu"]), "log": None}
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    bootstrap = None
    bootstrap_log = None

    def persist(stage):
        counts = {}
        for record in records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        atomic_json(
            state_path,
            {
                "status": "running",
                "stage": stage,
                "updated_at": now(),
                "pid": os.getpid(),
                "counts": counts,
                "jobs": records,
            },
        )

    append_event(events, "watcher-started", pid=os.getpid(), jobs=len(records))
    persist("waiting-for-training")
    while not stopped["value"]:
        if not queue_finished(result_root):
            persist("waiting-for-training")
            time.sleep(max(5.0, args.poll_seconds))
            continue

        for record in records.values():
            selected = best_main_seed(result_root, record["benchmark"])
            if selected is None:
                raise RuntimeError(
                    "five complete legal seeds are required before DREAMPlace: {}".format(
                        record["benchmark"]
                    )
                )
            record["seed"] = selected[1]
            record["selected_comp_res_hpwl"] = selected[0]
            record["result_path"] = str(
                dream_result(result_root, record["benchmark"], selected[1])
            )
            if record["status"] == "pending" and dream_done(Path(record["result_path"])):
                record["status"] = read_json(Path(record["result_path"]), {}).get("status")

        ready = result_root / "bootstrap" / "dreamplace-ready"
        if not ready.exists():
            if bootstrap is None:
                existing_pid = read_json(result_root / "bootstrap" / "dreamplace-bootstrap-state.json", {}).get("pid")
                pid_file = result_root / "bootstrap" / "dreamplace-bootstrap.pid"
                if pid_file.exists():
                    try:
                        existing_pid = int(pid_file.read_text(encoding="utf-8").strip())
                    except ValueError:
                        existing_pid = None
                if alive(existing_pid, ("bootstrap_dreamplace.sh",)):
                    persist("waiting-for-dreamplace-build")
                    time.sleep(max(5.0, args.poll_seconds))
                    continue
                bootstrap_log = (post_root / "dreamplace-bootstrap-retry.log").open("a", encoding="utf-8")
                bootstrap_environment = os.environ.copy()
                bootstrap_environment["LINKPLACE_RESULT_ROOT"] = str(result_root)
                bootstrap = subprocess.Popen(
                    ["bash", "tools/bootstrap_dreamplace.sh"],
                    cwd=str(project_root),
                    env=bootstrap_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=bootstrap_log,
                    stderr=subprocess.STDOUT,
                )
                append_event(events, "dreamplace-bootstrap-started", pid=bootstrap.pid)
            elif bootstrap.poll() is not None:
                code = bootstrap.returncode
                bootstrap_log.close()
                bootstrap = None
                bootstrap_log = None
                append_event(events, "dreamplace-bootstrap-retry", return_code=code)
                time.sleep(60.0)
            persist("waiting-for-dreamplace-build")
            time.sleep(max(5.0, args.poll_seconds))
            continue

        changed = False
        for job_id, runtime in list(active.items()):
            record = records[job_id]
            process = runtime["process"]
            if process is None:
                finished = not alive(record.get("pid"), ("run_dreamplace_mixed.py", record["benchmark"], str(record["seed"])))
                return_code = None
            else:
                return_code = process.poll()
                finished = return_code is not None
            if not finished:
                continue
            if runtime["log"] is not None:
                runtime["log"].close()
            del active[job_id]
            result = read_json(Path(record["result_path"]), {})
            if result.get("status") in {"complete", "unavailable"}:
                record["status"] = result["status"]
                record["finished_at"] = now()
                append_event(events, "dreamplace-job-finished", job_id=job_id, status=result["status"], return_code=return_code)
            else:
                record["status"] = "pending"
                record["last_return_code"] = return_code
                append_event(events, "dreamplace-job-requeued", job_id=job_id, return_code=return_code)
            record.pop("pid", None)
            record.pop("gpu", None)
            changed = True

        used = {runtime["gpu"] for runtime in active.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        for record in sorted(records.values(), key=lambda item: item["sequence"]):
            if not free:
                break
            if record["status"] != "pending":
                continue
            gpu = free.pop(0)
            attempt = int(record.get("attempts", 0)) + 1
            log_path = post_root / "logs" / record["id"] / "attempt-{:03d}.log".format(attempt)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a", encoding="utf-8")
            command = [
                args.python,
                "tools/run_dreamplace_mixed.py",
                record["benchmark"],
                "--seed",
                str(record["seed"]),
                "--main-root",
                str(result_root / "main"),
                "--result-root",
                str(result_root / "dreamplace"),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
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
                }
            )
            active[record["id"]] = {"process": process, "gpu": gpu, "log": log}
            append_event(events, "dreamplace-job-started", job_id=record["id"], pid=process.pid, gpu=gpu, attempt=attempt)
            changed = True

        if all(record["status"] in {"complete", "unavailable"} for record in records.values()):
            aggregate_log = post_root / "aggregate.log"
            with aggregate_log.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [args.python, "tools/aggregate_paper_results.py", "--result-root", str(result_root)],
                    cwd=str(project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            final = {
                "status": "complete" if completed.returncode == 0 else "failed",
                "stage": "complete",
                "finished_at": now(),
                "aggregate_return_code": completed.returncode,
                "aggregate_log": str(aggregate_log),
                "jobs": records,
            }
            atomic_json(state_path, final)
            append_event(events, "postprocess-complete", return_code=completed.returncode)
            return completed.returncode
        if changed:
            persist("dreamplace")
        time.sleep(max(5.0, args.poll_seconds))

    persist("stopped")
    append_event(events, "watcher-stopped", active=sorted(active))
    return 130


if __name__ == "__main__":
    raise SystemExit(main())

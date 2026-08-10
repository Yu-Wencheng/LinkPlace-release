#!/usr/bin/env python3
"""Archive already-produced official-code baseline evidence without rerunning it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SKIP_DIRECTORIES = {"models", "model", "tb_log", "tensorboard", "sandbox", "__pycache__"}
KEEP_SUFFIXES = {".json", ".log", ".csv", ".pl", ".txt", ".yaml", ".yml", ".png", ".pdf"}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def retained_files(run: Path):
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if any(part in SKIP_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if path.suffix.lower() in KEEP_SUFFIXES or path.name in {"events.out.tfevents"}:
            yield path, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing baseline runs to archive.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("outputs/formal/baselines/available_official_reproductions"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for evaluation_path in sorted(source.glob("*/*/evaluation.json")):
        run = evaluation_path.parent
        method = run.parent.name
        status = read_json(run / "status.json")
        evaluation = read_json(evaluation_path)
        target = destination / method / run.name
        files = []
        for path, relative in retained_files(run):
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            files.append({"path": str(relative), "sha256": sha256(output), "bytes": output.stat().st_size})
        # Every artifact currently available under baselines-smoke used a deliberately
        # reduced budget.  Preserve it as implementation evidence, never as a paper
        # result or convergence curve.
        record = {
            "method": method,
            "run": run.name,
            "source": str(run),
            "destination": str(target),
            "status": status.get("status"),
            "legal": status.get("legal", evaluation.get("legal")),
            "macro_hpwl": status.get("macro_hpwl", evaluation.get("macro_hpwl")),
            "official_commit": status.get("official_commit"),
            "evidence_level": "official-code-smoke-only",
            "eligible_for_paper_comparison": False,
            "exclusion_reason": "reduced smoke-test budget and incomplete benchmark/seed coverage",
            "files": files,
        }
        atomic_json(target / "archive-provenance.json", record)
        records.append(record)
    index = {
        "status": "complete",
        "source": str(source),
        "archived_runs": len(records),
        "paper_eligible_runs": sum(bool(item["eligible_for_paper_comparison"]) for item in records),
        "policy": "No baseline was rerun. Reduced-budget smoke evidence is retained but excluded from paper tables.",
        "runs": records,
    }
    atomic_json(destination / "index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

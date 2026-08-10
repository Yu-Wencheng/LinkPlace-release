#!/usr/bin/env python3
"""Aggregate the seed-1000 ISPD2005 grid-resolution ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
VARIANTS = ("linkplace-c", "linkplace-m", "all-greedy")
GRIDS = (448, 224)
SEED = 1000


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def result_path(result_root: Path, grid: int, variant: str, benchmark: str) -> Path:
    root = result_root if grid == 448 else result_root / "grid-ablation" / "grid-224"
    if variant == "linkplace-c":
        return root / "main" / benchmark / "seed-1000" / "result.json"
    candidates = [root / "ablation" / variant / benchmark / "seed-1000" / "result.json"]
    if variant == "linkplace-m":
        candidates.append(root / "ablation" / "monolithic" / benchmark / "seed-1000" / "result.json")
    return next((path for path in candidates if path.exists()), candidates[0])


def failure_text(result) -> str:
    if not result:
        return "missing result.json"
    failure = result.get("failure")
    if failure is None:
        failure = result.get("component", {}).get("failure")
    if failure is None:
        failure = result.get("reason")
    if failure is None:
        return ""
    return json.dumps(failure, ensure_ascii=False, sort_keys=True) if not isinstance(failure, str) else failure


def row_for(result_root: Path, grid: int, variant: str, benchmark: str):
    path = result_path(result_root, grid, variant, benchmark)
    result = read_json(path)
    metrics = (result or {}).get("metrics", {})
    component = (result or {}).get("component", {})
    return {
        "grid": grid,
        "benchmark": benchmark,
        "variant": variant,
        "seed": (result or {}).get("seed", SEED),
        "status": (result or {}).get("status", "missing"),
        "legal": metrics.get("legal", ""),
        "comp_res_hpwl": metrics.get("comp_res_hpwl", ""),
        "rudy_peak": metrics.get("rudy_peak", ""),
        "rudy_top5_mean": metrics.get("rudy_top5_mean", ""),
        "wall_seconds": (result or {}).get("wall_seconds", component.get("wall_seconds", "")),
        "episodes": component.get("episodes", ""),
        "best_epoch": component.get("best_epoch", ""),
        "stage": (result or {}).get("stage", ""),
        "failure": failure_text(result),
        "result_path": str(path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/formal"),
    )
    args = parser.parse_args()
    rows = [
        row_for(args.result_root, grid, variant, benchmark)
        for grid in GRIDS
        for benchmark in BENCHMARKS
        for variant in VARIANTS
    ]
    output = args.result_root / "paper_outputs" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "grid_ablation_seed1000.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "ispd2005-grid-ablation-seed1000",
        "seed": SEED,
        "grids": list(GRIDS),
        "variants": list(VARIANTS),
        "benchmarks": list(BENCHMARKS),
        "expected_results": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "csv": str(csv_path.resolve()),
    }
    (output / "grid_ablation_seed1000.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

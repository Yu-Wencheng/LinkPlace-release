#!/usr/bin/env python3
"""Aggregate five-seed dual-grid LinkPlace runs and ICCAD2015 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (999, 1000, 1001, 1002, 1003)
ISPD2005 = (
    "adaptec1", "adaptec2", "adaptec3", "adaptec4",
    "bigblue1", "bigblue2", "bigblue3", "bigblue4",
)
ICCAD2015 = (
    "superblue1", "superblue3", "superblue4", "superblue5",
    "superblue7", "superblue10", "superblue16", "superblue18",
)
GRIDS = (448, 224)
VARIANTS = ("linkplace-c", "linkplace-m", "all-greedy")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def result_path(root: Path, grid: int, variant: str, benchmark: str, seed: int):
    base = root if grid == 448 else root / "grid-ablation" / "grid-224"
    if variant == "linkplace-c":
        return base / "main" / benchmark / f"seed-{seed}" / "result.json"
    candidates = [base / "ablation" / variant / benchmark / f"seed-{seed}" / "result.json"]
    if variant == "linkplace-m":
        candidates.append(base / "ablation" / "monolithic" / benchmark / f"seed-{seed}" / "result.json")
    return next((path for path in candidates if path.exists()), candidates[0])


def seed_row(root: Path, grid: int, variant: str, benchmark: str, seed: int):
    path = result_path(root, grid, variant, benchmark, seed)
    result = read_json(path)
    metrics = (result or {}).get("metrics", {})
    component = (result or {}).get("component", {})
    status = (result or {}).get("status", "missing")
    return {
        "grid": grid,
        "benchmark": benchmark,
        "variant": variant,
        "seed": seed,
        "status": status,
        "legal": metrics.get("legal", ""),
        "comp_res_hpwl": metrics.get("comp_res_hpwl", ""),
        "rudy_peak": metrics.get("rudy_peak", ""),
        "rudy_top5_mean": metrics.get("rudy_top5_mean", ""),
        "wall_seconds": (result or {}).get("wall_seconds", component.get("wall_seconds", "")),
        "episodes": component.get("episodes", ""),
        "best_epoch": component.get("best_epoch", ""),
        "stage": (result or {}).get("stage", ""),
        "failure": (result or {}).get("failure", component.get("failure", "")),
        "result_path": str(path),
    }


def finite_values(rows, key):
    values = []
    for row in rows:
        if row["status"] != "complete" or row["legal"] is not True:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summary_row(grid, benchmark, variant, rows):
    hpwl = finite_values(rows, "comp_res_hpwl")
    rudy = finite_values(rows, "rudy_peak")
    runtime = finite_values(rows, "wall_seconds")
    return {
        "grid": grid,
        "benchmark": benchmark,
        "variant": variant,
        "successful_seeds": len(hpwl),
        "required_seeds": len(SEEDS),
        "failed_seeds": sum(row["status"] == "failed" for row in rows),
        "missing_seeds": sum(row["status"] == "missing" for row in rows),
        "comp_res_hpwl_mean": statistics.fmean(hpwl) if hpwl else "",
        "comp_res_hpwl_std": statistics.pstdev(hpwl) if len(hpwl) > 1 else (0.0 if hpwl else ""),
        "rudy_peak_mean": statistics.fmean(rudy) if rudy else "",
        "rudy_peak_std": statistics.pstdev(rudy) if len(rudy) > 1 else (0.0 if rudy else ""),
        "wall_seconds_mean": statistics.fmean(runtime) if runtime else "",
        "complete": len(hpwl) == len(SEEDS),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/formal"),
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = root / "paper_outputs" / "tables"

    grid_seed_rows = [
        seed_row(root, grid, variant, benchmark, seed)
        for grid in GRIDS
        for benchmark in ISPD2005
        for variant in VARIANTS
        for seed in SEEDS
    ]
    write_csv(output / "grid_ablation_five_seed_results.csv", grid_seed_rows)
    grid_summary = []
    for grid in GRIDS:
        for benchmark in ISPD2005:
            for variant in VARIANTS:
                rows = [
                    row for row in grid_seed_rows
                    if row["grid"] == grid
                    and row["benchmark"] == benchmark
                    and row["variant"] == variant
                ]
                grid_summary.append(summary_row(grid, benchmark, variant, rows))
    write_csv(output / "grid_ablation_five_seed_mean_std.csv", grid_summary)

    iccad_seed_rows = [
        seed_row(root, 448, "linkplace-m", benchmark, seed)
        for benchmark in ICCAD2015
        for seed in SEEDS
    ]
    write_csv(output / "linkplace_m_iccad2015_seed_results.csv", iccad_seed_rows)
    iccad_summary = [
        summary_row(
            448,
            benchmark,
            "linkplace-m",
            [row for row in iccad_seed_rows if row["benchmark"] == benchmark],
        )
        for benchmark in ICCAD2015
    ]
    write_csv(output / "linkplace_m_iccad2015_mean_std.csv", iccad_summary)

    state = {
        "protocol": "paper-extension-multiseed-v2",
        "seeds": list(SEEDS),
        "grid_seed_rows": len(grid_seed_rows),
        "grid_summary_rows": len(grid_summary),
        "iccad_seed_rows": len(iccad_seed_rows),
        "iccad_summary_rows": len(iccad_summary),
        "grid_terminal": sum(row["status"] in {"complete", "failed"} for row in grid_seed_rows),
        "iccad_complete": sum(row["status"] == "complete" for row in iccad_seed_rows),
    }
    (output / "paper_extension_multiseed_v2.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

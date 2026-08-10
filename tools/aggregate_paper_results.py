#!/usr/bin/env python3
"""Build paper-ready progress reports and tables from completed real artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (999, 1000, 1001, 1002, 1003)
MAIN_BENCHMARKS = (
    "adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3",
    "bigblue2", "bigblue4", "ariane", "superblue1", "superblue3", "superblue4",
    "superblue5", "superblue7", "superblue10", "superblue16", "superblue18",
)
ABLATION_BENCHMARKS = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3")
DREAMPLACE_BENCHMARKS = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3", "bigblue2", "bigblue4")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) >= 2 else 0.0


def format_number(value):
    return "--" if value is None else "{:.6g}".format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("outputs/formal"))
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = root / "paper_outputs"

    seed_rows = []
    incomplete = []
    for benchmark in MAIN_BENCHMARKS:
        for seed in SEEDS:
            path = root / "main" / benchmark / "seed-{}".format(seed) / "result.json"
            result = read_json(path)
            if not result or result.get("status") != "complete" or not result.get("metrics", {}).get("legal"):
                incomplete.append({"kind": "main", "benchmark": benchmark, "seed": seed, "path": str(path)})
                continue
            metrics = result["metrics"]
            seed_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "comp_res_hpwl": metrics["comp_res_hpwl"],
                    "rudy_peak": metrics["rudy_peak"],
                    "rudy_top5_mean": metrics["rudy_top5_mean"],
                    "flow_restart": result.get("flow_restart", 0),
                    "wall_seconds": result.get("wall_seconds"),
                    "result_path": str(path),
                    "final_placement": result.get("final_placement"),
                }
            )
    write_csv(
        output / "tables" / "main_seed_results.csv",
        seed_rows,
        (
            "benchmark", "seed", "comp_res_hpwl", "rudy_peak", "rudy_top5_mean",
            "flow_restart", "wall_seconds", "result_path", "final_placement",
        ),
    )

    aggregate_rows = []
    representative = {}
    for benchmark in MAIN_BENCHMARKS:
        rows = [row for row in seed_rows if row["benchmark"] == benchmark]
        hpwls = [float(row["comp_res_hpwl"]) for row in rows]
        peaks = [float(row["rudy_peak"]) for row in rows]
        tails = [float(row["rudy_top5_mean"]) for row in rows]
        hpwl_mean, hpwl_std = mean_std(hpwls)
        peak_mean, peak_std = mean_std(peaks)
        tail_mean, tail_std = mean_std(tails)
        best = min(rows, key=lambda row: float(row["comp_res_hpwl"])) if rows else None
        aggregate_rows.append(
            {
                "benchmark": benchmark,
                "completed_seeds": len(rows),
                "required_seeds": len(SEEDS),
                "comp_res_hpwl_mean": hpwl_mean,
                "comp_res_hpwl_std": hpwl_std,
                "rudy_peak_mean": peak_mean,
                "rudy_peak_std": peak_std,
                "rudy_top5_mean_mean": tail_mean,
                "rudy_top5_mean_std": tail_std,
                "best_seed": None if best is None else best["seed"],
                "best_comp_res_hpwl": None if best is None else best["comp_res_hpwl"],
                "complete": len(rows) == len(SEEDS),
            }
        )
        if best is not None:
            representative[benchmark] = best
    write_csv(
        output / "tables" / "main_mean_std.csv",
        aggregate_rows,
        (
            "benchmark", "completed_seeds", "required_seeds", "comp_res_hpwl_mean", "comp_res_hpwl_std",
            "rudy_peak_mean", "rudy_peak_std", "rudy_top5_mean_mean", "rudy_top5_mean_std",
            "best_seed", "best_comp_res_hpwl", "complete",
        ),
    )
    atomic_json(output / "representative_results.json", representative)

    latex = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Benchmark & Runs & CompRes HPWL & RUDY top-5\% mean \\",
        r"\hline",
    ]
    for row in aggregate_rows:
        hpwl = "--" if row["comp_res_hpwl_mean"] is None else "{} $\\pm$ {}".format(
            format_number(row["comp_res_hpwl_mean"]), format_number(row["comp_res_hpwl_std"])
        )
        rudy = "--" if row["rudy_top5_mean_mean"] is None else "{} $\\pm$ {}".format(
            format_number(row["rudy_top5_mean_mean"]), format_number(row["rudy_top5_mean_std"])
        )
        latex.append("{} & {}/{} & {} & {} \\\\".format(
            row["benchmark"], row["completed_seeds"], row["required_seeds"], hpwl, rudy
        ))
    latex.extend((r"\hline", r"\end{tabular}"))
    latex_path = output / "tables" / "main_mean_std.tex"
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")

    ablation_rows = []
    for benchmark in ABLATION_BENCHMARKS:
        current = representative.get(benchmark)
        if current is not None:
            ablation_rows.append(
                {
                    "benchmark": benchmark,
                    "variant": "linkplace-c-best-of-5",
                    "seed": current["seed"],
                    "comp_res_hpwl": current["comp_res_hpwl"],
                    "rudy_peak": current["rudy_peak"],
                    "rudy_top5_mean": current["rudy_top5_mean"],
                    "status": "complete",
                    "result_path": current["result_path"],
                }
            )
        for variant in ("linkplace-m", "all-greedy"):
            seed = 1000
            candidates = [root / "ablation" / variant / benchmark / "seed-{}".format(seed) / "result.json"]
            if variant == "linkplace-m":
                candidates.append(root / "ablation" / "monolithic" / benchmark / "seed-{}".format(seed) / "result.json")
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            result = read_json(path)
            if not result or result.get("status") != "complete":
                incomplete.append({"kind": "ablation", "variant": variant, "benchmark": benchmark, "path": str(path)})
                continue
            metrics = result["metrics"]
            ablation_rows.append(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "seed": seed,
                    "comp_res_hpwl": metrics["comp_res_hpwl"],
                    "rudy_peak": metrics["rudy_peak"],
                    "rudy_top5_mean": metrics["rudy_top5_mean"],
                    "status": result["status"],
                    "result_path": str(path),
                }
            )
    write_csv(
        output / "tables" / "ablation.csv",
        ablation_rows,
        ("benchmark", "variant", "seed", "comp_res_hpwl", "rudy_peak", "rudy_top5_mean", "status", "result_path"),
    )

    dream_seed_rows = []
    dream_aggregate_rows = []
    for benchmark in DREAMPLACE_BENCHMARKS:
        complete_values = []
        unavailable = 0
        selected = representative.get(benchmark)
        selected_seeds = () if selected is None else (int(selected["seed"]),)
        for seed in selected_seeds:
            path = root / "dreamplace" / benchmark / "seed-{}".format(seed) / "result.json"
            result = read_json(path)
            status = None if result is None else result.get("status")
            if status == "complete":
                value = float(result["full_design_hpwl"])
                complete_values.append(value)
                dream_seed_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "status": status,
                        "full_design_hpwl": value,
                        "before_full_design_hpwl": result.get("before_full_design_hpwl"),
                        "after_full_design_hpwl_recomputed": result.get("after_full_design_hpwl_recomputed"),
                        "before_rudy_peak": result.get("before_rudy_peak"),
                        "before_rudy_top5_mean": result.get("before_rudy_top5_mean"),
                        "after_rudy_peak": result.get("after_rudy_peak"),
                        "after_rudy_top5_mean": result.get("after_rudy_top5_mean"),
                        "fixed_macro_count": result.get("fixed_macro_count"),
                        "fixed_macro_max_coordinate_drift": result.get("fixed_macro_max_coordinate_drift"),
                        "wall_seconds": result.get("wall_seconds"),
                        "reason": "",
                        "result_path": str(path),
                    }
                )
            elif status == "unavailable":
                unavailable += 1
                dream_seed_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "status": status,
                        "full_design_hpwl": None,
                        "before_full_design_hpwl": None,
                        "after_full_design_hpwl_recomputed": None,
                        "before_rudy_peak": None,
                        "before_rudy_top5_mean": None,
                        "after_rudy_peak": None,
                        "after_rudy_top5_mean": None,
                        "fixed_macro_count": None,
                        "fixed_macro_max_coordinate_drift": None,
                        "wall_seconds": result.get("wall_seconds"),
                        "reason": result.get("reason"),
                        "result_path": str(path),
                    }
                )
            else:
                incomplete.append({"kind": "dreamplace", "benchmark": benchmark, "seed": seed, "path": str(path)})
        value_mean, value_std = mean_std(complete_values)
        dream_aggregate_rows.append(
            {
                "benchmark": benchmark,
                "completed_seeds": len(complete_values),
                "unavailable_seeds": unavailable,
                "required_seeds": 1,
                "full_design_hpwl_mean": value_mean,
                "full_design_hpwl_std": value_std,
                "best_full_design_hpwl": min(complete_values) if complete_values else None,
                "complete": len(complete_values) == 1 or unavailable == 1,
            }
        )
    write_csv(
        output / "tables" / "dreamplace_seed_results.csv",
        dream_seed_rows,
        (
            "benchmark", "seed", "status", "full_design_hpwl", "fixed_macro_count",
            "before_full_design_hpwl", "after_full_design_hpwl_recomputed",
            "before_rudy_peak", "before_rudy_top5_mean", "after_rudy_peak", "after_rudy_top5_mean",
            "fixed_macro_max_coordinate_drift", "wall_seconds", "reason", "result_path",
        ),
    )
    write_csv(
        output / "tables" / "dreamplace_mean_std.csv",
        dream_aggregate_rows,
        (
            "benchmark", "completed_seeds", "unavailable_seeds", "required_seeds",
            "full_design_hpwl_mean", "full_design_hpwl_std", "best_full_design_hpwl", "complete",
        ),
    )

    queue_state = read_json(root / "queue" / "queue-state.json")
    baseline_index_path = root / "baselines" / "available_official_reproductions" / "index.json"
    baseline_index = read_json(baseline_index_path)
    progress = {
        "main_completed": len(seed_rows),
        "main_required": len(MAIN_BENCHMARKS) * len(SEEDS),
        "main_benchmarks_fully_complete": sum(row["complete"] for row in aggregate_rows),
        "main_benchmarks_required": len(MAIN_BENCHMARKS),
        "ablation_completed": sum(row["variant"] in {"linkplace-m", "all-greedy"} for row in ablation_rows),
        "ablation_required": 2 * len(ABLATION_BENCHMARKS),
        "dreamplace_completed": sum(row["status"] == "complete" for row in dream_seed_rows),
        "dreamplace_unavailable": sum(row["status"] == "unavailable" for row in dream_seed_rows),
        "dreamplace_required": len(DREAMPLACE_BENCHMARKS),
        "baseline_archive": None
        if baseline_index is None
        else {
            "status": baseline_index.get("status"),
            "archived_runs": baseline_index.get("archived_runs"),
            "paper_eligible_runs": baseline_index.get("paper_eligible_runs"),
            "index": str(baseline_index_path),
        },
        "incomplete": incomplete,
        "queue": None if queue_state is None else queue_state.get("counts"),
    }
    atomic_json(output / "progress.json", progress)
    print(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

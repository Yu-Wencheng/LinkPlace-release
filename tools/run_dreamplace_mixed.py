#!/usr/bin/env python3
"""Run official DREAMPlace 4.1.0 with a LinkPlace-C layout fixed in place."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


BENCHMARKS = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3", "bigblue2", "bigblue4")
HPWL_PATTERN = re.compile(r"\bHPWL\s+([-+0-9.eE]+)", re.IGNORECASE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meaningful_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if line and not line.startswith("#") and not line.lower().startswith("ucla "):
                yield line


def aux_files(aux_path: Path):
    for line in meaningful_lines(aux_path):
        if line.lower().startswith("rowbasedplacement"):
            return [(aux_path.parent / token).resolve() for token in line.split(":", 1)[1].split()]
    raise ValueError("invalid Bookshelf aux: {}".format(aux_path))


def parse_terminals(nodes_path: Path):
    terminals = {}
    total = 0
    for line in meaningful_lines(nodes_path):
        lower = line.lower()
        if lower.startswith("numnodes") or lower.startswith("numterminals"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            width, height = float(fields[1]), float(fields[2])
        except ValueError:
            continue
        total += 1
        if any(token.lower().startswith("terminal") for token in fields[3:]):
            terminals[fields[0]] = (width, height)
    return terminals, total


def parse_selected_positions(pl_path: Path, wanted):
    wanted = set(wanted)
    result = {}
    for line in meaningful_lines(pl_path):
        fields = line.replace(":", " : ").split()
        if len(fields) >= 3 and fields[0] in wanted:
            result[fields[0]] = (float(fields[1]), float(fields[2]))
    return result


def parse_all_positions(pl_path: Path):
    result = {}
    for line in meaningful_lines(pl_path):
        fields = line.replace(":", " : ").split()
        if len(fields) < 3:
            continue
        try:
            result[fields[0]] = (float(fields[1]), float(fields[2]))
        except ValueError:
            continue
    return result


def parse_node_dimensions(nodes_path: Path):
    result = {}
    for line in meaningful_lines(nodes_path):
        fields = line.split()
        if len(fields) < 3 or fields[0].lower().startswith("numnodes") or fields[0].lower().startswith("numterminals"):
            continue
        try:
            result[fields[0]] = (float(fields[1]), float(fields[2]))
        except ValueError:
            continue
    return result


def parse_scl_bounds(scl_path: Path, positions, dimensions):
    minimum_x = math.inf
    minimum_y = math.inf
    maximum_x = -math.inf
    maximum_y = -math.inf
    coordinate = None
    height = None
    site_width = None
    for line in meaningful_lines(scl_path):
        fields = line.replace(":", " : ").split()
        lower = [item.lower() for item in fields]
        if not fields:
            continue
        if lower[0] == "coordinate":
            coordinate = float(fields[-1])
        elif lower[0] == "height":
            height = float(fields[-1])
        elif lower[0] == "sitewidth":
            site_width = float(fields[-1])
        elif lower[0] == "subroworigin" and "numsites" in lower:
            origin = float(fields[2] if len(fields) > 2 and fields[1] == ":" else fields[1])
            marker = lower.index("numsites")
            count = float(fields[marker + 2] if marker + 2 < len(fields) and fields[marker + 1] == ":" else fields[marker + 1])
            width = 1.0 if site_width is None else site_width
            minimum_x = min(minimum_x, origin)
            maximum_x = max(maximum_x, origin + count * width)
            if coordinate is not None:
                minimum_y = min(minimum_y, coordinate)
                maximum_y = max(maximum_y, coordinate + (height or 0.0))
    if not all(math.isfinite(value) for value in (minimum_x, minimum_y, maximum_x, maximum_y)):
        minimum_x = minimum_y = math.inf
        maximum_x = maximum_y = -math.inf
    for name, (x, y) in positions.items():
        width, node_height = dimensions.get(name, (0.0, 0.0))
        minimum_x = min(minimum_x, x)
        minimum_y = min(minimum_y, y)
        maximum_x = max(maximum_x, x + width)
        maximum_y = max(maximum_y, y + node_height)
    return minimum_x, minimum_y, maximum_x, maximum_y


def _axis_groups(low, high, origin, cell, grid):
    shifted_low = max(0.0, low - origin)
    shifted_high = max(shifted_low, high - origin)
    start = max(0, min(grid - 1, int(math.floor(shifted_low / cell))))
    end = max(0, min(grid - 1, int(math.ceil(shifted_high / cell) - 1)))
    if start == end:
        return ((start, end, min(1.0, max(0.0, (shifted_high - shifted_low) / cell))),)
    first = ((start + 1) * cell - shifted_low) / cell
    last = (shifted_high - end * cell) / cell
    groups = [(start, start, min(1.0, max(0.0, first)))]
    if start + 1 <= end - 1:
        groups.append((start + 1, end - 1, 1.0))
    groups.append((end, end, min(1.0, max(0.0, last))))
    return tuple(item for item in groups if item[2] > 0.0)


def evaluate_full_design(nodes_path, nets_path, scl_path, pl_path, grid=224):
    dimensions = parse_node_dimensions(nodes_path)
    positions = parse_all_positions(pl_path)
    bounds = parse_scl_bounds(scl_path, positions, dimensions)
    x0, y0, x1, y1 = bounds
    cell_x = max((x1 - x0) / grid, np.finfo(np.float64).eps)
    cell_y = max((y1 - y0) / grid, np.finfo(np.float64).eps)
    difference = np.zeros((grid + 1, grid + 1), dtype=np.float64)
    hpwl = 0.0
    evaluated_nets = 0
    current = None

    def finish(points):
        nonlocal hpwl, evaluated_nets
        if points is None or len(points) < 2:
            return
        xs = np.asarray([item[0] for item in points], dtype=np.float64)
        ys = np.asarray([item[1] for item in points], dtype=np.float64)
        raw_x0, raw_x1 = float(xs.min()), float(xs.max())
        raw_y0, raw_y1 = float(ys.min()), float(ys.max())
        hpwl += raw_x1 - raw_x0 + raw_y1 - raw_y0
        evaluated_nets += 1
        center_x = (raw_x0 + raw_x1) / 2.0
        center_y = (raw_y0 + raw_y1) / 2.0
        span_x = max(raw_x1 - raw_x0, cell_x)
        span_y = max(raw_y1 - raw_y0, cell_y)
        box_x0 = max(x0, center_x - span_x / 2.0)
        box_x1 = min(x1, center_x + span_x / 2.0)
        box_y0 = max(y0, center_y - span_y / 2.0)
        box_y1 = min(y1, center_y + span_y / 2.0)
        span_x = max(box_x1 - box_x0, np.finfo(np.float64).eps)
        span_y = max(box_y1 - box_y0, np.finfo(np.float64).eps)
        density = (span_x + span_y) / (span_x * span_y)
        for ix0, ix1, fraction_x in _axis_groups(box_x0, box_x1, x0, cell_x, grid):
            for iy0, iy1, fraction_y in _axis_groups(box_y0, box_y1, y0, cell_y, grid):
                value = density * fraction_x * fraction_y
                difference[ix0, iy0] += value
                difference[ix1 + 1, iy0] -= value
                difference[ix0, iy1 + 1] -= value
                difference[ix1 + 1, iy1 + 1] += value

    with nets_path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or stripped.lower().startswith("ucla"):
                continue
            fields = stripped.split()
            if fields[0].lower() == "netdegree":
                finish(current)
                current = []
                continue
            if current is None or fields[0].lower() in {"numnets", "numpins"}:
                continue
            name = fields[0]
            if name not in positions or name not in dimensions:
                continue
            try:
                offset_x = float(fields[-2])
                offset_y = float(fields[-1])
            except (ValueError, IndexError):
                continue
            x, y = positions[name]
            width, height = dimensions[name]
            current.append((x + width / 2.0 + offset_x, y + height / 2.0 + offset_y))
    finish(current)
    demand = difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1)
    flat = demand.reshape(-1)
    tail_count = max(1, int(math.ceil(0.05 * flat.size)))
    tail = flat if tail_count == flat.size else np.partition(flat, flat.size - tail_count)[-tail_count:]
    return {
        "hpwl": float(hpwl),
        "rudy_peak": float(flat.max(initial=0.0)),
        "rudy_top5_mean": float(tail.mean()),
        "rudy_grid": grid,
        "evaluated_nets": evaluated_nets,
        "placed_nodes": len(positions),
        "bounds": list(bounds),
    }, demand


def save_rudy_heatmap(path: Path, demand, title):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(np.asarray(demand).T, origin="lower", cmap="magma", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("x bin")
    axis.set_ylabel("y bin")
    figure.colorbar(image, ax=axis, label="RUDY")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_full_design_placement_plot(
    path: Path,
    nodes_path: Path,
    pl_path: Path,
    bounds,
    highlighted,
    title,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import patches

    dimensions = parse_node_dimensions(nodes_path)
    positions = parse_all_positions(pl_path)
    centers = [
        (
            value[0] + dimensions[name][0] / 2.0,
            value[1] + dimensions[name][1] / 2.0,
        )
        for name, value in positions.items()
        if name in dimensions and name not in highlighted
    ]
    x0, y0, x1, y1 = bounds
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    if centers:
        xs = np.asarray([item[0] for item in centers])
        ys = np.asarray([item[1] for item in centers])
        density, x_edges, y_edges = np.histogram2d(
            xs, ys, bins=384, range=((x0, x1), (y0, y1))
        )
        image = axis.imshow(
            np.log1p(density).T,
            origin="lower",
            extent=(x0, x1, y0, y1),
            cmap="Blues",
            aspect="equal",
        )
        figure.colorbar(image, ax=axis, label="log(1 + movable-cell count)")
    for name in highlighted:
        if name not in positions or name not in dimensions:
            continue
        x, y = positions[name]
        width, height = dimensions[name]
        axis.add_patch(
            patches.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor="#E45756",
                linewidth=0.35,
                alpha=0.8,
            )
        )
    axis.set_xlim(x0, x1)
    axis.set_ylim(y0, y1)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def find_fixed_obstacle_overlaps(terminals, original_pl: Path, selected_placement):
    unselected_names = set(terminals) - set(selected_placement)
    unselected_positions = parse_selected_positions(original_pl, unselected_names)
    if len(unselected_positions) != len(unselected_names):
        missing = sorted(unselected_names - set(unselected_positions))
        raise ValueError("terminal positions missing: {}".format(missing[:5]))
    ordered = sorted((x, name, y) for name, (x, y) in unselected_positions.items())
    starts = [item[0] for item in ordered]
    examples = []
    for name, (x, y) in selected_placement.items():
        width, height = terminals[name]
        stop = bisect.bisect_left(starts, x + width)
        for other_x, other, other_y in ordered[:stop]:
            other_width, other_height = terminals[other]
            if other_x + other_width <= x:
                continue
            if other_y < y + height and y < other_y + other_height:
                examples.append({"selected": name, "fixed": other})
                if len(examples) >= 20:
                    return examples
    return examples


def find_selected_overlaps(terminals, selected_placement):
    ordered = sorted((x, name, y) for name, (x, y) in selected_placement.items())
    examples = []
    for index, (left_x, left_name, left_y) in enumerate(ordered):
        left_width, left_height = terminals[left_name]
        for right_x, right_name, right_y in ordered[index + 1 :]:
            if right_x >= left_x + left_width:
                break
            right_width, right_height = terminals[right_name]
            if right_y < left_y + left_height and left_y < right_y + right_height:
                examples.append({"left": left_name, "right": right_name})
                if len(examples) >= 20:
                    return examples
    return examples


def prepare_input(source_dir: Path, input_dir: Path, placement):
    source_aux = source_dir / (source_dir.name + ".aux")
    files = aux_files(source_aux)
    by_suffix = {path.suffix.lower(): path for path in files}
    required = {".nodes", ".nets", ".pl", ".scl"}
    if not required.issubset(by_suffix):
        raise ValueError("Bookshelf files missing: {}".format(sorted(required - set(by_suffix))))
    terminals, total_nodes = parse_terminals(by_suffix[".nodes"])
    missing = sorted(set(placement) - set(terminals))
    if missing:
        raise ValueError("LinkPlace-C macros are not terminals in full design: {}".format(missing[:10]))
    selected_overlaps = find_selected_overlaps(terminals, placement)
    if selected_overlaps:
        return None, {
            "status": "unavailable",
            "reason": "Bookshelf-quantized selected macro layout contains overlaps",
            "overlap_examples": selected_overlaps,
            "selected_macros": len(placement),
            "full_design_terminals": len(terminals),
            "full_design_nodes": total_nodes,
        }
    overlaps = find_fixed_obstacle_overlaps(terminals, by_suffix[".pl"], placement)
    if overlaps:
        return None, {
            "status": "unavailable",
            "reason": "selected macro layout overlaps original fixed unselected terminals",
            "overlap_examples": overlaps,
            "selected_macros": len(placement),
            "full_design_terminals": len(terminals),
            "full_design_nodes": total_nodes,
        }

    input_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = input_dir / path.name
        if path.suffix.lower() != ".pl" and not target.exists():
            target.symlink_to(path)
    output_pl = input_dir / by_suffix[".pl"].name
    found = set()
    with by_suffix[".pl"].open("r", encoding="utf-8", errors="replace") as source, output_pl.open(
        "w", encoding="utf-8"
    ) as output:
        for raw in source:
            fields = raw.strip().replace(":", " : ").split()
            if len(fields) >= 3 and fields[0] in placement:
                name = fields[0]
                x, y = placement[name]
                orientation = "N"
                if ":" in fields:
                    marker = fields.index(":")
                    if marker + 1 < len(fields) and not fields[marker + 1].startswith("/"):
                        orientation = fields[marker + 1]
                output.write("\t{}\t{:.6f}\t{:.6f}\t: {} /FIXED\n".format(name, x, y, orientation))
                found.add(name)
            else:
                output.write(raw)
    if found != set(placement):
        raise ValueError("failed to rewrite all macro positions")
    output_aux = input_dir / source_aux.name
    output_aux.write_text(source_aux.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return output_aux, {
        "selected_macros": len(placement),
        "full_design_terminals": len(terminals),
        "full_design_nodes": total_nodes,
        "unselected_fixed_terminals": len(terminals) - len(placement),
    }


def output_macro_drift(output_pl: Path, expected):
    actual = parse_selected_positions(output_pl, expected)
    if len(actual) != len(expected):
        return float("inf"), sorted(set(expected) - set(actual))[:10]
    drift = max(
        max(abs(actual[name][0] - value[0]), abs(actual[name][1] - value[1]))
        for name, value in expected.items()
    )
    return drift, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--main-root", type=Path, default=Path("outputs/formal/main"))
    parser.add_argument("--result-root", type=Path, default=Path("outputs/formal/dreamplace"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dreamplace-official/ispd2005"))
    parser.add_argument("--dreamplace-source", type=Path, default=Path("third_party/DREAMPlace-4.1.0-source"))
    parser.add_argument("--dreamplace-install", type=Path, default=Path("third_party/DREAMPlace-4.1.0-install"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    output = args.result_root.resolve() / args.benchmark / "seed-{}".format(args.seed)
    status_path = output / "result.json"
    if status_path.exists():
        existing = read_json(status_path)
        if existing.get("status") in {"complete", "unavailable"}:
            print(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    current_result_path = args.main_root.resolve() / args.benchmark / "seed-{}".format(args.seed) / "result.json"
    current_result = read_json(current_result_path)
    if current_result.get("status") != "complete" or not current_result.get("metrics", {}).get("legal"):
        raise ValueError("LinkPlace-C result is not complete and legal: {}".format(current_result_path))
    placement_path = Path(current_result["final_placement"])
    placement = {name: (float(value[0]), float(value[1])) for name, value in read_json(placement_path)["placement"].items()}
    # Bookshelf/DREAMPlace stores placement coordinates as integer database
    # units. Quantize explicitly before writing so fixed-macro drift is measured
    # against the exact optimizer input instead of the higher-precision LinkPlace-C
    # JSON coordinate.
    bookshelf_placement = {
        name: (float(math.floor(value[0] + 0.5)), float(math.floor(value[1] + 0.5)))
        for name, value in placement.items()
    }
    quantization_delta = max(
        max(abs(bookshelf_placement[name][0] - placement[name][0]), abs(bookshelf_placement[name][1] - placement[name][1]))
        for name in placement
    )
    atomic_json(
        output / "fixed-macro-bookshelf-input.json",
        {"placement": {name: list(value) for name, value in sorted(bookshelf_placement.items())}},
    )

    source_dir = args.data_root.resolve() / args.benchmark
    aux_path, input_metadata = prepare_input(source_dir, output / "input", bookshelf_placement)
    input_metadata["source_to_bookshelf_max_quantization"] = quantization_delta
    if aux_path is None:
        result = {
            **input_metadata,
            "benchmark": args.benchmark,
            "seed": args.seed,
            "variant": "LinkPlace-C-fixed-macros-plus-DREAMPlace-4.1.0",
            "current_result": str(current_result_path),
            "wall_seconds": time.time() - started,
        }
        atomic_json(status_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    prepared_files = {path.suffix.lower(): path for path in aux_files(aux_path)}
    before_metrics, before_demand = evaluate_full_design(
        prepared_files[".nodes"],
        prepared_files[".nets"],
        prepared_files[".scl"],
        prepared_files[".pl"],
    )
    np.save(output / "rudy-before-224.npy", before_demand)
    save_rudy_heatmap(
        output / "rudy-before-224.png",
        before_demand,
        "{} before DREAMPlace".format(args.benchmark),
    )
    save_full_design_placement_plot(
        output / "placement-before.png",
        prepared_files[".nodes"],
        prepared_files[".pl"],
        before_metrics["bounds"],
        set(bookshelf_placement),
        "{} before DREAMPlace".format(args.benchmark),
    )
    atomic_json(output / "full-design-before.json", before_metrics)

    template = args.dreamplace_source.resolve() / "test" / "ispd2005" / (args.benchmark + ".json")
    configuration = read_json(template)
    configuration.update(
        {
            "aux_input": str(aux_path),
            "result_dir": str(output / "official-results"),
            "random_seed": int(args.seed),
            "gpu": 1,
            "plot_flag": 0,
            "random_center_init_flag": 1,
            "deterministic_flag": 0,
        }
    )
    config_path = output / "dreamplace-config.json"
    atomic_json(config_path, configuration)
    command = [args.python, str(args.dreamplace_install.resolve() / "dreamplace" / "Placer.py"), str(config_path)]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    atomic_json(
        output / "status.json",
        {
            "status": "running",
            "benchmark": args.benchmark,
            "seed": args.seed,
            "command": command,
            "input": input_metadata,
            "started_at": started,
        },
    )
    log_path = output / "official.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        result = {
            "status": "failed",
            "benchmark": args.benchmark,
            "seed": args.seed,
            "return_code": completed.returncode,
            "log": str(log_path),
            "wall_seconds": time.time() - started,
        }
        atomic_json(output / "status.json", result)
        atomic_json(status_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    placements = sorted((output / "official-results").rglob("*.gp.pl"))
    if len(placements) != 1:
        raise RuntimeError("expected one DREAMPlace output placement, found {}".format(len(placements)))
    final_pl = placements[0]
    drift, missing = output_macro_drift(final_pl, bookshelf_placement)
    values = [float(match.group(1)) for match in HPWL_PATTERN.finditer(log_path.read_text(encoding="utf-8", errors="replace"))]
    if not values:
        raise RuntimeError("DREAMPlace log contains no HPWL metric")
    after_metrics, after_demand = evaluate_full_design(
        prepared_files[".nodes"],
        prepared_files[".nets"],
        prepared_files[".scl"],
        final_pl,
    )
    np.save(output / "rudy-after-224.npy", after_demand)
    save_rudy_heatmap(
        output / "rudy-after-224.png",
        after_demand,
        "{} after DREAMPlace".format(args.benchmark),
    )
    save_full_design_placement_plot(
        output / "placement-after.png",
        prepared_files[".nodes"],
        final_pl,
        after_metrics["bounds"],
        set(bookshelf_placement),
        "{} after DREAMPlace".format(args.benchmark),
    )
    atomic_json(output / "full-design-after.json", after_metrics)
    result = {
        "status": "complete" if drift <= 1e-3 and not missing else "failed",
        "benchmark": args.benchmark,
        "seed": args.seed,
        "variant": "LinkPlace-C-fixed-macros-plus-DREAMPlace-4.1.0",
        "dreamplace_version": "4.1.0",
        "dreamplace_commit": "5d13c9001a3bc900dca1e108e633d5dd45b00701",
        "full_design_hpwl": values[-1],
        "before_full_design_hpwl": before_metrics["hpwl"],
        "after_full_design_hpwl_recomputed": after_metrics["hpwl"],
        "full_design_hpwl_delta": after_metrics["hpwl"] - before_metrics["hpwl"],
        "hpwl_trace": values,
        "before_rudy_peak": before_metrics["rudy_peak"],
        "before_rudy_top5_mean": before_metrics["rudy_top5_mean"],
        "after_rudy_peak": after_metrics["rudy_peak"],
        "after_rudy_top5_mean": after_metrics["rudy_top5_mean"],
        "rudy_before_array": str(output / "rudy-before-224.npy"),
        "rudy_before_heatmap": str(output / "rudy-before-224.png"),
        "rudy_after_array": str(output / "rudy-after-224.npy"),
        "rudy_after_heatmap": str(output / "rudy-after-224.png"),
        "placement_before_plot": str(output / "placement-before.png"),
        "placement_after_plot": str(output / "placement-after.png"),
        "fixed_macro_max_coordinate_drift": drift,
        "fixed_macro_drift_reference": "fixed-macro-bookshelf-input.json",
        "source_to_bookshelf_max_quantization": quantization_delta,
        "missing_fixed_macros": missing,
        "fixed_macro_count": len(placement),
        "input": input_metadata,
        "current_result": str(current_result_path),
        "current_placement": str(placement_path),
        "dreamplace_placement": str(final_pl),
        "dreamplace_placement_sha256": sha256(final_pl),
        "configuration": str(config_path),
        "official_log": str(log_path),
        "return_code": completed.returncode,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output / "status.json", result)
    atomic_json(status_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

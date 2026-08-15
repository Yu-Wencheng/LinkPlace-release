#!/usr/bin/env python3
"""Render the six complete-layout best-of-five LinkPlace galleries.

The script consumes the archived final placements used by the paper.  It is
kept separate from ``render_readme_results.py`` because the full placement
archives and benchmark geometry are intentionally not duplicated in the
lightweight release repository.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


CIRCUITS = (
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
)

BEST_SEEDS = {
    "linkplace-m": {
        "adaptec1": 1003,
        "adaptec2": 999,
        "adaptec3": 999,
        "adaptec4": 1003,
        "bigblue1": 1000,
        "bigblue3": 1002,
    },
    "linkplace-c": {
        "adaptec1": 1001,
        "adaptec2": 999,
        "adaptec3": 999,
        "adaptec4": 1002,
        "bigblue1": 1000,
        "bigblue3": 999,
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ucla_nodes(path: Path) -> dict[str, tuple[float, float]]:
    nodes: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            row = raw.strip().split()
            if len(row) < 3 or row[0].startswith("#") or row[0] == "UCLA":
                continue
            try:
                nodes[row[0]] = (float(row[1]), float(row[2]))
            except ValueError:
                continue
    return nodes


def component_color(index: int) -> tuple[float, float, float]:
    """Match the component palette used by Fig. 5 in the paper."""

    hue = (0.618033988749895 * index + 0.08) % 1.0
    saturation = 0.62 + 0.18 * ((index % 3) / 2.0)
    value = 0.78 + 0.16 * (index % 2)
    return colorsys.hsv_to_rgb(hue, saturation, value)


def c_flow_directory(artifact_root: Path, circuit: str, seed: int) -> Path:
    seed_directory = artifact_root / "main" / circuit / f"seed-{seed}"
    result = load_json(seed_directory / "result.json")
    restart = int(result.get("flow_restart", 0))
    return seed_directory / f"flow-restart-{restart:03d}"


def layout_inputs(
    artifact_root: Path,
    benchmark_root: Path,
    variant: str,
    circuit: str,
):
    seed = BEST_SEEDS[variant][circuit]
    nodes = parse_ucla_nodes(benchmark_root / circuit / f"{circuit}.nodes")

    if variant == "linkplace-m":
        placement_path = (
            artifact_root
            / "ablation"
            / "monolithic"
            / circuit
            / f"seed-{seed}"
            / "final"
            / "placement.json"
        )
        placement = load_json(placement_path)["placement"]
        component_ids = {name: 0 for name in placement}
    else:
        flow = c_flow_directory(artifact_root, circuit, seed)
        placement = load_json(flow / "final" / "placement.json")["placement"]
        ordered_components = load_json(flow / "components.json")["ordered_components"]
        component_ids = {
            macro: color_index
            for color_index, component in enumerate(ordered_components)
            for macro in component["macros"]
        }

    benchmark_path = artifact_root / "main" / circuit / f"seed-{BEST_SEEDS['linkplace-c'][circuit]}" / "benchmark.json"
    canvas = load_json(benchmark_path)["canvas"]

    missing_geometry = sorted(set(placement) - set(nodes))
    missing_components = sorted(set(placement) - set(component_ids))
    if missing_geometry:
        raise ValueError(f"{variant}/{circuit}: missing geometry for {len(missing_geometry)} macros")
    if missing_components:
        raise ValueError(f"{variant}/{circuit}: missing component IDs for {len(missing_components)} macros")
    return placement, nodes, component_ids, canvas


def render_gallery(
    artifact_root: Path,
    benchmark_root: Path,
    output_path: Path,
    variant: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.45))
    for axis, circuit in zip(axes.ravel(), CIRCUITS):
        placement, nodes, component_ids, canvas = layout_inputs(
            artifact_root, benchmark_root, variant, circuit
        )
        for name, coordinates in placement.items():
            width, height = nodes[name]
            if variant == "linkplace-c":
                facecolor = component_color(component_ids[name])
                alpha = 0.88
            else:
                facecolor = "#4C78A8"
                alpha = 0.78
            axis.add_patch(
                Rectangle(
                    (float(coordinates[0]), float(coordinates[1])),
                    width,
                    height,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=0.18,
                    alpha=alpha,
                )
            )

        axis.set_xlim(float(canvas["x"]), float(canvas["x"]) + float(canvas["width"]))
        axis.set_ylim(float(canvas["y"]), float(canvas["y"]) + float(canvas["height"]))
        axis.set_aspect("equal", adjustable="box")
        axis.text(
            0.5,
            0.985,
            circuit,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)

    figure.subplots_adjust(
        left=0.025,
        right=0.99,
        bottom=0.02,
        top=0.99,
        wspace=0.08,
        hspace=0.08,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=workspace / "论文" / "experiment_artifacts",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=workspace / "yu-PPo" / "benchmark" / "ispd05",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "assets" / "paper")
    args = parser.parse_args()

    render_gallery(
        args.artifact_root,
        args.benchmark_root,
        args.output_dir / "best_of_five_linkplace_m_layouts.png",
        "linkplace-m",
    )
    render_gallery(
        args.artifact_root,
        args.benchmark_root,
        args.output_dir / "best_of_five_linkplace_c_layouts.png",
        "linkplace-c",
    )
    print("rendered best-of-five LinkPlace-M and LinkPlace-C layout galleries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

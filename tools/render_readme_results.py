#!/usr/bin/env python3
"""Render deterministic README result tables and dependency-free SVG figures."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "tables"
FIGURES = ROOT / "assets" / "results"
README = ROOT / "README.md"

ISPD = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
ICCAD = ["superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18"]
SEEDS = [999, 1000, 1001, 1002, 1003]

COLORS = {
    "linkplace-c": "#2563EB",
    "linkplace-m": "#F97316",
    "maskplace": "#64748B",
    "wiremask": "#0F766E",
    "efficientplace": "#7C3AED",
    "grid": "#CBD5E1",
    "text": "#0F172A",
    "muted": "#475569",
    "failure": "#DC2626",
}


def rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, title: str, description: str, body: list[str]) -> str:
    style = """
    text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Arial, sans-serif; fill: #0F172A; }
    .title { font-size: 27px; font-weight: 700; }
    .subtitle { font-size: 15px; fill: #475569; }
    .axis { font-size: 13px; fill: #475569; }
    .label { font-size: 13px; font-weight: 600; }
    .small { font-size: 12px; fill: #475569; }
    .legend { font-size: 13px; font-weight: 600; }
    """.strip()
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"  <title id=\"title\">{esc(title)}</title>",
            f"  <desc id=\"desc\">{esc(description)}</desc>",
            f"  <style>{style}</style>",
            f'  <rect width="{width}" height="{height}" fill="#FFFFFF"/>',
            *[f"  {item}" for item in body],
            "</svg>",
            "",
        ]
    )


def write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"stale generated file: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1, dash: str | None = None) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{extra}/>'


def text(x: float, y: float, value: object, *, css: str = "axis", anchor: str = "start", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" text-anchor="{anchor}"{transform}>{esc(value)}</text>'


def rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str | None = None, radius: float = 0) -> str:
    border = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{radius}" fill="{fill}"{border}/>'


def circle(x: float, y: float, radius: float, *, fill: str, stroke: str = "#FFFFFF") -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def indexed(data: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in data}


def require_exact_keys(
    data: list[dict[str, str]],
    fields: tuple[str, ...],
    expected: set[tuple[str, ...]],
    label: str,
) -> None:
    actual = [tuple(row[field] for field in fields) for row in data]
    unique = set(actual)
    if len(actual) != len(unique):
        raise SystemExit(f"duplicate records in {label}: {len(actual) - len(unique)}")
    missing, extra = expected - unique, unique - expected
    if missing or extra:
        raise SystemExit(
            f"unexpected {label} coverage: missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def validate_inputs(
    main_summary: list[dict[str, str]],
    main_seeds: list[dict[str, str]],
    grid_summary: list[dict[str, str]],
    grid_seeds: list[dict[str, str]],
    iccad_m_summary: list[dict[str, str]],
    iccad_m_seeds: list[dict[str, str]],
    baseline_summary: list[dict[str, str]],
    baseline_seeds: list[dict[str, str]],
) -> None:
    c_benchmarks = ISPD + ["ariane"] + ICCAD
    seed_strings = [str(seed) for seed in SEEDS]
    require_exact_keys(
        main_summary,
        ("benchmark",),
        {(benchmark,) for benchmark in c_benchmarks},
        "LinkPlace-C summaries",
    )
    require_exact_keys(
        main_seeds,
        ("benchmark", "seed"),
        {(benchmark, seed) for benchmark in c_benchmarks for seed in seed_strings},
        "LinkPlace-C seeds",
    )
    require_exact_keys(
        grid_summary,
        ("grid", "benchmark", "variant"),
        {
            (grid, benchmark, variant)
            for grid in ("224", "448")
            for benchmark in ISPD
            for variant in ("linkplace-c", "linkplace-m", "all-greedy")
        },
        "dual-grid summaries",
    )
    require_exact_keys(
        grid_seeds,
        ("grid", "benchmark", "variant", "seed"),
        {
            (grid, benchmark, variant, seed)
            for grid in ("224", "448")
            for benchmark in ISPD
            for variant in ("linkplace-c", "linkplace-m", "all-greedy")
            for seed in seed_strings
        },
        "dual-grid seeds",
    )
    require_exact_keys(
        iccad_m_summary,
        ("benchmark",),
        {(benchmark,) for benchmark in ICCAD},
        "LinkPlace-M ICCAD2015 summaries",
    )
    require_exact_keys(
        iccad_m_seeds,
        ("grid", "benchmark", "variant", "seed"),
        {("448", benchmark, "linkplace-m", seed) for benchmark in ICCAD for seed in seed_strings},
        "LinkPlace-M ICCAD2015 seeds",
    )
    baseline_methods = ("maskplace", "wiremask", "efficientplace")
    require_exact_keys(
        baseline_summary,
        ("method", "benchmark"),
        {(method, benchmark) for method in baseline_methods for benchmark in ISPD},
        "official baseline summaries",
    )
    require_exact_keys(
        baseline_seeds,
        ("method", "benchmark", "seed"),
        {(method, benchmark, "1000") for method in baseline_methods for benchmark in ISPD},
        "official baseline seeds",
    )


def render_ispd_comparison(grid_summary: list[dict[str, str]], baselines: list[dict[str, str]]) -> str:
    grid = indexed(grid_summary, "grid", "benchmark", "variant")
    baseline = indexed(baselines, "method", "benchmark")
    methods = ["linkplace-c", "linkplace-m", "maskplace", "wiremask", "efficientplace"]
    labels = ["LinkPlace-C (5 seeds)", "LinkPlace-M (5 seeds)", "MaskPlace (seed 1000)", "WireMask-EA (seed 1000)", "EfficientPlace (seed 1000)"]
    values: dict[tuple[str, str], float | None] = {}
    for benchmark in ISPD:
        reference = number(grid[("448", benchmark, "linkplace-c")]["comp_res_hpwl_mean"])
        assert reference is not None
        for method in methods:
            if method in {"linkplace-c", "linkplace-m"}:
                value = number(grid[("448", benchmark, method)]["comp_res_hpwl_mean"])
            else:
                value = number(baseline[(method, benchmark)]["comp_res_hpwl_mean"])
            values[(method, benchmark)] = None if value is None else value / reference

    width, height = 1500, 650
    left, right, top, bottom = 82, 35, 95, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(value for value in values.values() if value is not None)
    y_max = math.ceil((maximum + 0.1) * 2) / 2
    body = [
        text(left, 38, "ISPD2005 CompRes MacroHPWL relative to LinkPlace-C", css="title"),
        text(left, 66, "Five-seed means for LinkPlace; official baselines are single seed 1000. Lower is better.", css="subtitle"),
    ]
    for tick in range(int(y_max * 2) + 1):
        value = tick / 2
        y = top + plot_h - value / y_max * plot_h
        body += [line(left, y, width - right, y, stroke=COLORS["grid"]), text(left - 12, y + 4, f"{value:.1f}x", anchor="end")]
    body += [line(left, top, left, top + plot_h, stroke=COLORS["text"], width=1.2), line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["text"], width=1.2)]
    group_w = plot_w / len(ISPD)
    bar_w = group_w * 0.13
    for i, benchmark in enumerate(ISPD):
        center = left + group_w * (i + 0.5)
        for j, method in enumerate(methods):
            x = center + (j - 2) * bar_w * 1.08 - bar_w / 2
            value = values[(method, benchmark)]
            if value is None:
                y = top + plot_h - 9
                body += [line(x + 2, y - 7, x + bar_w - 2, y + 7, stroke=COLORS["failure"], width=2), line(x + 2, y + 7, x + bar_w - 2, y - 7, stroke=COLORS["failure"], width=2)]
            else:
                bar_h = value / y_max * plot_h
                body.append(rect(x, top + plot_h - bar_h, bar_w, bar_h, fill=COLORS[method], radius=2))
        body.append(text(center + 18, top + plot_h + 25, benchmark, css="label", anchor="end", rotate=-35))
    legend_x, legend_y = left + 20, height - 31
    for method, label in zip(methods, labels):
        body += [rect(legend_x, legend_y - 13, 18, 12, fill=COLORS[method], radius=2), text(legend_x + 25, legend_y - 2, label, css="legend")]
        legend_x += 260
    body += [line(width - 160, legend_y - 12, width - 146, legend_y + 2, stroke=COLORS["failure"], width=2), line(width - 160, legend_y + 2, width - 146, legend_y - 12, stroke=COLORS["failure"], width=2), text(width - 138, legend_y - 2, "technical failure", css="legend")]
    return svg_document(width, height, "ISPD2005 normalized HPWL comparison", "Grouped bars compare CompRes MacroHPWL ratios against the LinkPlace-C five-seed mean.", body)


def render_grid_ablation(grid_summary: list[dict[str, str]]) -> str:
    grid = indexed(grid_summary, "grid", "benchmark", "variant")
    ratios: dict[tuple[str, str], float] = {}
    success: dict[tuple[str, str], int] = {}
    for benchmark in ISPD:
        for method in ("linkplace-c", "linkplace-m"):
            low = number(grid[("224", benchmark, method)]["comp_res_hpwl_mean"])
            high = number(grid[("448", benchmark, method)]["comp_res_hpwl_mean"])
            assert low is not None and high is not None
            ratios[(method, benchmark)] = low / high
        for resolution in ("224", "448"):
            success[(resolution, benchmark)] = int(grid[(resolution, benchmark, "all-greedy")]["successful_seeds"])

    width, height = 1500, 760
    left, right = 90, 45
    plot_w = width - left - right
    body = [
        text(left, 38, "Grid-resolution ablation and All-greedy legality", css="title"),
        text(left, 66, "Top: 224/448 mean HPWL ratio (above 1 means 224 is worse). Bottom: legal All-greedy runs out of five.", css="subtitle"),
    ]
    top_y, top_h = 110, 330
    y_min, y_max = 0.9, max(1.5, math.ceil(max(ratios.values()) * 10) / 10)
    for tick in range(int(round((y_max - y_min) / 0.1)) + 1):
        value = y_min + tick * 0.1
        y = top_y + top_h - (value - y_min) / (y_max - y_min) * top_h
        body += [line(left, y, width - right, y, stroke=COLORS["grid"], dash="4 4" if abs(value - 1) < 1e-9 else None), text(left - 12, y + 4, f"{value:.1f}x", anchor="end")]
    x_positions = [left + plot_w * (i + 0.5) / len(ISPD) for i in range(len(ISPD))]
    for method in ("linkplace-c", "linkplace-m"):
        points = []
        for x, benchmark in zip(x_positions, ISPD):
            value = ratios[(method, benchmark)]
            y = top_y + top_h - (value - y_min) / (y_max - y_min) * top_h
            points.append((x, y))
        body.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" fill="none" stroke="{COLORS[method]}" stroke-width="3"/>')
        for x, y in points:
            body.append(circle(x, y, 6, fill=COLORS[method]))
    body += [rect(left + 20, 87, 18, 12, fill=COLORS["linkplace-c"], radius=2), text(left + 45, 97, "LinkPlace-C", css="legend"), rect(left + 175, 87, 18, 12, fill=COLORS["linkplace-m"], radius=2), text(left + 200, 97, "LinkPlace-M", css="legend")]

    lower_y, lower_h = 520, 150
    for tick in range(6):
        y = lower_y + lower_h - tick / 5 * lower_h
        body += [line(left, y, width - right, y, stroke=COLORS["grid"]), text(left - 12, y + 4, tick, anchor="end")]
    group_w = plot_w / len(ISPD)
    for i, (x, benchmark) in enumerate(zip(x_positions, ISPD)):
        for offset, resolution, color in [(-14, "448", "#334155"), (14, "224", "#94A3B8")]:
            value = success[(resolution, benchmark)]
            bar_h = value / 5 * lower_h
            body.append(rect(x + offset - 11, lower_y + lower_h - bar_h, 22, bar_h, fill=color, radius=2))
            body.append(text(x + offset, lower_y + lower_h - bar_h - 6, value, css="small", anchor="middle"))
        body.append(text(x + 18, lower_y + lower_h + 27, benchmark, css="label", anchor="end", rotate=-35))
    body += [rect(width - 280, 482, 18, 12, fill="#334155", radius=2), text(width - 255, 492, "448 grid", css="legend"), rect(width - 160, 482, 18, 12, fill="#94A3B8", radius=2), text(width - 135, 492, "224 grid", css="legend")]
    return svg_document(width, height, "Grid ablation and legality", "Line chart compares 224-to-448 HPWL ratios and bars show All-greedy legal counts.", body)


def blend(color: tuple[int, int, int], strength: float) -> str:
    strength = max(0.0, min(1.0, strength))
    rgb = tuple(round(255 + (channel - 255) * strength) for channel in color)
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def render_seed_stability(main_seeds: list[dict[str, str]], grid_seeds: list[dict[str, str]], iccad_m_seeds: list[dict[str, str]]) -> str:
    c_rows = {(row["benchmark"], int(row["seed"])): number(row["comp_res_hpwl"]) for row in main_seeds}
    m_source = [row for row in grid_seeds if row["grid"] == "448" and row["variant"] == "linkplace-m"] + iccad_m_seeds
    m_rows = {(row["benchmark"], int(row["seed"])): number(row["comp_res_hpwl"]) for row in m_source}
    c_benchmarks = ISPD + ["ariane"] + ICCAD
    m_benchmarks = ISPD + ICCAD

    def deviations(source: dict[tuple[str, int], float | None], benchmarks: list[str]) -> dict[tuple[str, int], float]:
        output: dict[tuple[str, int], float] = {}
        for benchmark in benchmarks:
            values = [source[(benchmark, seed)] for seed in SEEDS]
            assert all(value is not None for value in values)
            mean = sum(value for value in values if value is not None) / len(values)
            for seed, value in zip(SEEDS, values):
                assert value is not None
                output[(benchmark, seed)] = (value / mean - 1) * 100
        return output

    c_dev = deviations(c_rows, c_benchmarks)
    m_dev = deviations(m_rows, m_benchmarks)
    scale = max(5.0, math.ceil(max(abs(value) for value in list(c_dev.values()) + list(m_dev.values())) / 5) * 5)
    width, height = 1500, 970
    body = [
        text(55, 38, "Five-seed HPWL stability", css="title"),
        text(55, 66, f"Each cell is the seed's deviation from that method/circuit mean. Common color scale: ±{scale:.0f}%.", css="subtitle"),
    ]

    def panel(x0: float, title_value: str, benchmarks: list[str], values: dict[tuple[str, int], float]) -> None:
        label_w, cell_w, cell_h = 122, 92, 43
        y0 = 128
        body.append(text(x0, 101, title_value, css="label"))
        for col, seed in enumerate(SEEDS):
            body.append(text(x0 + label_w + col * cell_w + cell_w / 2, y0 - 12, seed, css="label", anchor="middle"))
        for row_index, benchmark in enumerate(benchmarks):
            y = y0 + row_index * cell_h
            body.append(text(x0 + label_w - 10, y + 27, benchmark, css="label", anchor="end"))
            for col, seed in enumerate(SEEDS):
                value = values[(benchmark, seed)]
                color = (37, 99, 235) if value < 0 else (249, 115, 22)
                fill = blend(color, 0.18 + 0.75 * min(abs(value) / scale, 1))
                x = x0 + label_w + col * cell_w
                body.append(rect(x, y, cell_w - 4, cell_h - 4, fill=fill, radius=3))
                body.append(text(x + (cell_w - 4) / 2, y + 26, f"{value:+.1f}%", css="small", anchor="middle"))

    panel(40, "LinkPlace-C · 17 circuits · 85 runs", c_benchmarks, c_dev)
    panel(790, "LinkPlace-M · 16 circuits · 80 public seed records", m_benchmarks, m_dev)
    body += [rect(595, height - 48, 25, 14, fill=blend((37, 99, 235), 0.75), radius=2), text(628, height - 36, "below mean (lower HPWL)", css="legend"), rect(840, height - 48, 25, 14, fill=blend((249, 115, 22), 0.75), radius=2), text(873, height - 36, "above mean", css="legend")]
    return svg_document(width, height, "Five-seed stability heatmap", "Heatmap shows seed-level HPWL deviations from each method and circuit mean.", body)


def render_rudy_delta(main_summary: list[dict[str, str]], grid_summary: list[dict[str, str]], iccad_m_summary: list[dict[str, str]]) -> str:
    c_main = indexed(main_summary, "benchmark")
    grid = indexed(grid_summary, "grid", "benchmark", "variant")
    m_iccad = indexed(iccad_m_summary, "benchmark")
    benchmarks = ISPD + ICCAD
    values: dict[tuple[str, str], float] = {}
    for benchmark in benchmarks:
        c = grid[("448", benchmark, "linkplace-c")] if benchmark in ISPD else c_main[(benchmark,)]
        m = grid[("448", benchmark, "linkplace-m")] if benchmark in ISPD else m_iccad[(benchmark,)]
        for metric, column in [("peak", "rudy_peak_mean"), ("top5", "rudy_top5_mean_mean")]:
            c_value, m_value = number(c[column]), number(m[column])
            assert c_value is not None and m_value is not None
            values[(benchmark, metric)] = (m_value / c_value - 1) * 100
    extent = max(abs(value) for value in values.values())
    axis_max = max(20, math.ceil(extent / 10) * 10)
    width, height = 1400, 850
    left, right, top, bottom = 180, 65, 110, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    body = [
        text(left, 38, "LinkPlace-M RUDY change relative to LinkPlace-C", css="title"),
        text(left, 66, "Matched five-seed means on 16 circuits. Negative values favor LinkPlace-M; lower RUDY is better.", css="subtitle"),
    ]
    for tick in range(-axis_max, axis_max + 1, 10):
        x = left + (tick + axis_max) / (2 * axis_max) * plot_w
        body += [line(x, top, x, top + plot_h, stroke="#64748B" if tick == 0 else COLORS["grid"], width=1.5 if tick == 0 else 1, dash=None if tick == 0 else "4 4"), text(x, top + plot_h + 26, f"{tick:+d}%", anchor="middle")]
    row_h = plot_h / len(benchmarks)
    for i, benchmark in enumerate(benchmarks):
        y = top + row_h * (i + 0.5)
        body.append(text(left - 16, y + 4, benchmark, css="label", anchor="end"))
        if i in {7}:
            body.append(line(left - 130, y + row_h / 2, width - right, y + row_h / 2, stroke="#94A3B8", width=2))
        for metric, dy, color in [("peak", -6, "#2563EB"), ("top5", 6, "#F97316")]:
            value = values[(benchmark, metric)]
            x0 = left + plot_w / 2
            x = left + (value + axis_max) / (2 * axis_max) * plot_w
            body.append(line(x0, y + dy, x, y + dy, stroke=color, width=2.2))
            if metric == "peak":
                body.append(circle(x, y + dy, 5.5, fill=color))
            else:
                body.append(rect(x - 5, y + dy - 5, 10, 10, fill=color, radius=1))
    body += [circle(left + 35, 91, 6, fill="#2563EB"), text(left + 50, 96, "Peak RUDY", css="legend"), rect(left + 180, 85, 12, 12, fill="#F97316", radius=1), text(left + 200, 96, "Top-5% mean", css="legend"), text(width - right, 96, "Ariane excluded: LinkPlace-M seed CSV is not in this public snapshot.", css="small", anchor="end")]
    return svg_document(width, height, "RUDY relative change", "Dot plot compares LinkPlace-M and LinkPlace-C five-seed RUDY means per circuit.", body)


def fmt_scaled(row: dict[str, str], scale: float, mean_key: str = "comp_res_hpwl_mean", std_key: str = "comp_res_hpwl_std", digits: int = 3) -> str:
    mean, std = number(row.get(mean_key)), number(row.get(std_key))
    if mean is None:
        return "—"
    if std is None:
        return f"{mean / scale:.{digits}f}"
    return f"{mean / scale:.{digits}f} ± {std / scale:.{digits}f}"


def fmt_runtime(row: dict[str, str]) -> str:
    value = number(row.get("wall_seconds_mean"))
    return "—" if value is None else f"{value / 3600:.2f} h"


def markdown_section(
    main_summary: list[dict[str, str]],
    main_seeds: list[dict[str, str]],
    grid_summary: list[dict[str, str]],
    grid_seeds: list[dict[str, str]],
    iccad_m_summary: list[dict[str, str]],
    iccad_m_seeds: list[dict[str, str]],
    baseline_summary: list[dict[str, str]],
    baseline_seeds: list[dict[str, str]],
) -> str:
    main = indexed(main_summary, "benchmark")
    grid = indexed(grid_summary, "grid", "benchmark", "variant")
    iccad_m = indexed(iccad_m_summary, "benchmark")
    baselines = indexed(baseline_summary, "method", "benchmark")
    baseline_records = indexed(baseline_seeds, "method", "benchmark")
    main_runtime_values: dict[str, list[float]] = {}
    for row in main_seeds:
        value = number(row.get("wall_seconds"))
        if value is not None:
            main_runtime_values.setdefault(row["benchmark"], []).append(value)
    main_runtime_hours = {
        benchmark: sum(values) / len(values) / 3600
        for benchmark, values in main_runtime_values.items()
    }
    legal_grid = sum(row["legal"].lower() == "true" for row in grid_seeds)
    legal_baselines = sum(row["legal"].lower() == "true" for row in baseline_seeds)

    output = [
        "<!-- README_RESULTS:BEGIN -->",
        "## Detailed experimental results",
        "",
        "This section is generated from the archived server-run CSV records under [`artifacts/tables/`](artifacts/tables/) by [`tools/render_readme_results.py`](tools/render_readme_results.py). It intentionally exposes more numerical detail than the compact paper tables. All reported placements use the same CompRes legality/HPWL evaluator; lower HPWL and RUDY are better. `±` denotes the sample standard deviation across seeds. Absolute values should be compared within a circuit, not across unrelated benchmarks with different coordinate scales.",
        "",
        "### Coverage and protocol",
        "",
        "| Result family | Public records | Protocol | Terminal outcome |",
        "|---|---:|---|---|",
        f"| LinkPlace-C main | {len(main_seeds)} | 17 circuits × seeds 999–1003 | 85/85 complete legal layouts |",
        f"| Dual-grid ablation | {len(grid_seeds)} | 8 ISPD2005 × 2 grids × 3 variants × 5 seeds | {legal_grid}/{len(grid_seeds)} legal; All-greedy failures retained |",
        f"| LinkPlace-M ICCAD2015 | {len(iccad_m_seeds)} | 8 circuits × seeds 999–1003 | 40/40 complete legal layouts |",
        f"| Official baselines | {len(baseline_seeds)} | 3 methods × 8 ISPD2005 × seed 1000 | {legal_baselines}/{len(baseline_seeds)} legal; 2 technical failures retained |",
        "",
        "Formal PPO runs use a 448 × 448 action grid, 1,000 episodes, and seeds 999–1003. The grid ablation additionally evaluates 224 × 224. Official MaskPlace, WireMask-EA, and EfficientPlace rows are single-seed (`1000`) results and are not presented as five-seed statistics.",
        "",
        "### ISPD2005 method comparison",
        "",
        "![ISPD2005 normalized HPWL comparison](assets/results/ispd2005_hpwl_relative.svg)",
        "",
        "Exact CompRes MacroHPWL values below are scaled by `1e5`. LinkPlace rows are five-seed mean ± sample standard deviation; official baselines are seed 1000 only.",
        "",
        "| Circuit | LinkPlace-C | LinkPlace-M | Δ M vs C | MaskPlace | WireMask-EA | EfficientPlace |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ISPD:
        c = grid[("448", benchmark, "linkplace-c")]
        m = grid[("448", benchmark, "linkplace-m")]
        c_mean, m_mean = number(c["comp_res_hpwl_mean"]), number(m["comp_res_hpwl_mean"])
        assert c_mean is not None and m_mean is not None
        baseline_values = []
        for method in ("maskplace", "wiremask", "efficientplace"):
            row = baselines[(method, benchmark)]
            value = number(row["comp_res_hpwl_mean"])
            if value is None:
                status = baseline_records[(method, benchmark)]["controller_status"].replace("_", " ")
                baseline_values.append(f"*{status}*")
            else:
                baseline_values.append(f"{value / 1e5:.3f}")
        output.append(f"| {benchmark} | {fmt_scaled(c, 1e5)} | {fmt_scaled(m, 1e5)} | {(m_mean / c_mean - 1) * 100:+.2f}% | " + " | ".join(baseline_values) + " |")

    output += [
        "",
        "### Grid-resolution ablation",
        "",
        "![Grid ablation](assets/results/grid_ablation.svg)",
        "",
        "<details>",
        "<summary>Exact dual-grid HPWL and All-greedy success counts</summary>",
        "",
        "| Circuit | LinkPlace-C 448 | LinkPlace-C 224 | LinkPlace-M 448 | LinkPlace-M 224 | All-greedy legal 448 | All-greedy legal 224 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ISPD:
        c448 = grid[("448", benchmark, "linkplace-c")]
        c224 = grid[("224", benchmark, "linkplace-c")]
        m448 = grid[("448", benchmark, "linkplace-m")]
        m224 = grid[("224", benchmark, "linkplace-m")]
        greedy448 = grid[("448", benchmark, "all-greedy")]["successful_seeds"]
        greedy224 = grid[("224", benchmark, "all-greedy")]["successful_seeds"]
        output.append(
            f"| {benchmark} | {fmt_scaled(c448, 1e5)} | {fmt_scaled(c224, 1e5)} | "
            f"{fmt_scaled(m448, 1e5)} | {fmt_scaled(m224, 1e5)} | {greedy448}/5 | {greedy224}/5 |"
        )
    output += ["", "</details>", "", "### Ariane and ICCAD2015-derived instances", ""]
    ariane = main[("ariane",)]
    output += [
        f"The public LinkPlace-C artifact contains all five Ariane seeds: MacroHPWL `{fmt_scaled(ariane, 1e5)}` × `1e5`, peak RUDY `{number(ariane['rudy_peak_mean']):.4f} ± {number(ariane['rudy_peak_std']):.4f}`, mean runtime `{main_runtime_hours['ariane']:.2f} h`, and best seed `{ariane['best_seed']}`. The paper also reports the LinkPlace-M Ariane summary, but its seed-level CSV is not present in this public snapshot; the README therefore does not synthesize unversioned per-seed values.",
        "",
        "For the eight ICCAD2015-derived macro-only instances, the table gives unrounded five-seed statistics scaled by `1e8`.",
        "",
        "| Circuit | LinkPlace-C HPWL | LinkPlace-M HPWL | Δ M vs C | C runtime | M runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ICCAD:
        c, m = main[(benchmark,)], iccad_m[(benchmark,)]
        c_mean, m_mean = number(c["comp_res_hpwl_mean"]), number(m["comp_res_hpwl_mean"])
        assert c_mean is not None and m_mean is not None
        output.append(f"| {benchmark} | {fmt_scaled(c, 1e8)} | {fmt_scaled(m, 1e8)} | {(m_mean / c_mean - 1) * 100:+.2f}% | {main_runtime_hours[benchmark]:.2f} h | {fmt_runtime(m)} |")

    output += [
        "",
        "### Seed stability and RUDY trade-offs",
        "",
        "![Five-seed stability heatmap](assets/results/seed_stability.svg)",
        "",
        "The heatmap exposes every public seed instead of only mean ± standard deviation. The normalization is performed independently within each method/circuit pair, so it measures stochastic spread rather than cross-circuit quality.",
        "",
        "![RUDY relative changes](assets/results/rudy_relative_delta.svg)",
        "",
        "<details>",
        "<summary>Exact matched RUDY means for LinkPlace-C and LinkPlace-M</summary>",
        "",
        "| Circuit | C peak | M peak | C top-5% mean | M top-5% mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark in ISPD + ICCAD:
        c = grid[("448", benchmark, "linkplace-c")] if benchmark in ISPD else main[(benchmark,)]
        m = grid[("448", benchmark, "linkplace-m")] if benchmark in ISPD else iccad_m[(benchmark,)]
        output.append(f"| {benchmark} | {number(c['rudy_peak_mean']):.6g} | {number(m['rudy_peak_mean']):.6g} | {number(c['rudy_top5_mean_mean']):.6g} | {number(m['rudy_top5_mean_mean']):.6g} |")
    output += [
        "",
        "</details>",
        "",
        "### Machine-readable records",
        "",
        "- [LinkPlace-C: 85 per-seed records](artifacts/tables/main_seed_results.csv) and [17-circuit summary](artifacts/tables/main_mean_std.csv)",
        "- [Dual-grid: 240 per-seed records](artifacts/tables/grid_ablation_five_seed_results.csv) and [48 summary rows](artifacts/tables/grid_ablation_five_seed_mean_std.csv)",
        "- [LinkPlace-M ICCAD2015: 40 per-seed records](artifacts/tables/linkplace_m_iccad2015_seed_results.csv) and [8-circuit summary](artifacts/tables/linkplace_m_iccad2015_mean_std.csv)",
        "- [Official baselines: 24 seed records](artifacts/tables/baseline_seed_results.csv), including WireMask-EA/adaptec4 and EfficientPlace/bigblue1 technical failures",
        "",
        "Regenerate and verify the README section and all SVGs with:",
        "",
        "```bash",
        "python tools/render_readme_results.py --write",
        "python tools/render_readme_results.py --check",
        "```",
        "<!-- README_RESULTS:END -->",
    ]
    return "\n".join(output)


def update_readme(section: str, check: bool) -> None:
    current = README.read_text(encoding="utf-8")
    begin, end = "<!-- README_RESULTS:BEGIN -->", "<!-- README_RESULTS:END -->"
    if begin in current and end in current:
        prefix = current.split(begin, 1)[0].rstrip()
        suffix = current.split(end, 1)[1].lstrip("\r\n")
        expected = prefix + "\n\n" + section + "\n\n" + suffix
    else:
        marker = "## Formal reproduction"
        if marker not in current:
            raise SystemExit(f"README insertion marker not found: {marker}")
        prefix, suffix = current.split(marker, 1)
        expected = prefix.rstrip() + "\n\n" + section + "\n\n" + marker + suffix
    if not expected.endswith("\n"):
        expected += "\n"
    if check:
        if current != expected:
            raise SystemExit("README generated result section is stale")
    else:
        README.write_text(expected, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write generated SVGs and README section")
    mode.add_argument("--check", action="store_true", help="fail if generated outputs are stale")
    args = parser.parse_args()
    check = args.check

    main_summary = rows("main_mean_std.csv")
    main_seeds = rows("main_seed_results.csv")
    grid_summary = rows("grid_ablation_five_seed_mean_std.csv")
    grid_seeds = rows("grid_ablation_five_seed_results.csv")
    iccad_m_summary = rows("linkplace_m_iccad2015_mean_std.csv")
    iccad_m_seeds = rows("linkplace_m_iccad2015_seed_results.csv")
    baseline_summary = rows("baseline_mean_std.csv")
    baseline_seeds = rows("baseline_seed_results.csv")

    validate_inputs(
        main_summary,
        main_seeds,
        grid_summary,
        grid_seeds,
        iccad_m_summary,
        iccad_m_seeds,
        baseline_summary,
        baseline_seeds,
    )

    figures = {
        FIGURES / "ispd2005_hpwl_relative.svg": render_ispd_comparison(grid_summary, baseline_summary),
        FIGURES / "grid_ablation.svg": render_grid_ablation(grid_summary),
        FIGURES / "seed_stability.svg": render_seed_stability(main_seeds, grid_seeds, iccad_m_seeds),
        FIGURES / "rudy_relative_delta.svg": render_rudy_delta(main_summary, grid_summary, iccad_m_summary),
    }
    for path, content in figures.items():
        write_or_check(path, content, check)

    section = markdown_section(main_summary, main_seeds, grid_summary, grid_seeds, iccad_m_summary, iccad_m_seeds, baseline_summary, baseline_seeds)
    update_readme(section, check)
    action = "verified" if check else "wrote"
    print(f"{action} {len(figures)} SVG figures and the generated README result section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

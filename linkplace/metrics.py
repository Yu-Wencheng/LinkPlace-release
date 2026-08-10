from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import Benchmark, Net, Rect


@dataclass(frozen=True)
class LegalityReport:
    legal: bool
    boundary_violations: Tuple[str, ...]
    obstacle_violations: Tuple[Tuple[str, str], ...]
    overlaps: Tuple[Tuple[str, str], ...]
    missing_macros: Tuple[str, ...]
    nonfinite_macros: Tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def _pin_coordinates(benchmark: Benchmark, net: Net, placement: Mapping[str, Sequence[float]]):
    xs: List[float] = []
    ys: List[float] = []
    for pin in net.pins:
        if pin.owner in placement:
            node = benchmark.nodes[pin.owner]
            x, y = placement[pin.owner]
            xs.append(float(x) + node.width / 2.0 + pin.offset_x)
            ys.append(float(y) + node.height / 2.0 + pin.offset_y)
        elif pin.owner in benchmark.fixed_terminals:
            x, y = benchmark.fixed_terminals[pin.owner]
            xs.append(float(x) + pin.offset_x)
            ys.append(float(y) + pin.offset_y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def hpwl_for_nets(benchmark: Benchmark, placement, nets: Iterable[Net]) -> float:
    total = 0.0
    for net in nets:
        xs, ys = _pin_coordinates(benchmark, net, placement)
        if xs.size >= 2:
            total += float(xs.max() - xs.min() + ys.max() - ys.min())
    return total


def macro_hpwl(benchmark: Benchmark, placement) -> float:
    return hpwl_for_nets(benchmark, placement, benchmark.evaluated_nets)


def validate_placement(benchmark: Benchmark, placement, tolerance: float = 1e-9) -> LegalityReport:
    missing = tuple(name for name in benchmark.selected_macros if name not in placement)
    nonfinite = []
    boundary = []
    rectangles = {}
    canvas = benchmark.canvas
    for name in benchmark.selected_macros:
        if name not in placement:
            continue
        node = benchmark.nodes[name]
        x, y = map(float, placement[name])
        rect = Rect(x, y, node.width, node.height)
        rectangles[name] = rect
        if not all(math.isfinite(value) for value in (x, y, node.width, node.height)):
            nonfinite.append(name)
        elif (
            rect.x < canvas.x - tolerance
            or rect.y < canvas.y - tolerance
            or rect.x2 > canvas.x2 + tolerance
            or rect.y2 > canvas.y2 + tolerance
        ):
            boundary.append(name)
    overlaps = []
    ordered = sorted(rectangles.items(), key=lambda item: (item[1].x, item[0]))
    for index, (left_name, left) in enumerate(ordered):
        for right_name, right in ordered[index + 1 :]:
            if right.x >= left.x2 - tolerance:
                break
            if min(left.y2, right.y2) - max(left.y, right.y) > tolerance:
                overlaps.append((left_name, right_name))
    obstacle_hits = []
    for name, rect in rectangles.items():
        for obstacle_name, obstacle in benchmark.fixed_obstacles.items():
            if (
                min(rect.x2, obstacle.x2) - max(rect.x, obstacle.x) > tolerance
                and min(rect.y2, obstacle.y2) - max(rect.y, obstacle.y) > tolerance
            ):
                obstacle_hits.append((name, obstacle_name))
    return LegalityReport(
        legal=not (missing or nonfinite or boundary or overlaps or obstacle_hits),
        boundary_violations=tuple(boundary),
        obstacle_violations=tuple(obstacle_hits),
        overlaps=tuple(overlaps),
        missing_macros=missing,
        nonfinite_macros=tuple(nonfinite),
    )


def _axis_overlap_groups(low: float, high: float, cell_size: float, grid: int):
    start = max(0, min(grid - 1, int(math.floor(low / cell_size))))
    end = max(0, min(grid - 1, int(math.ceil(high / cell_size) - 1)))
    if start == end:
        return ((start, end, max(0.0, high - low) / cell_size),)
    first = ((start + 1) * cell_size - low) / cell_size
    last = (high - end * cell_size) / cell_size
    groups = [(start, start, min(1.0, max(0.0, first)))]
    if start + 1 <= end - 1:
        groups.append((start + 1, end - 1, 1.0))
    groups.append((end, end, min(1.0, max(0.0, last))))
    return tuple(item for item in groups if item[2] > 0.0)


def rudy_map(benchmark: Benchmark, placement, grid: int = 224) -> np.ndarray:
    if grid <= 0:
        raise ValueError("RUDY grid must be positive")
    cell_w = benchmark.canvas.width / grid
    cell_h = benchmark.canvas.height / grid
    difference = np.zeros((grid + 1, grid + 1), dtype=np.float64)
    for net in benchmark.evaluated_nets:
        xs, ys = _pin_coordinates(benchmark, net, placement)
        if xs.size < 2:
            continue
        raw_x0, raw_x1 = float(xs.min()), float(xs.max())
        raw_y0, raw_y1 = float(ys.min()), float(ys.max())
        center_x = (raw_x0 + raw_x1) / 2.0 - benchmark.canvas.x
        center_y = (raw_y0 + raw_y1) / 2.0 - benchmark.canvas.y
        span_x = max(raw_x1 - raw_x0, cell_w)
        span_y = max(raw_y1 - raw_y0, cell_h)
        x0 = max(0.0, center_x - span_x / 2.0)
        x1 = min(benchmark.canvas.width, center_x + span_x / 2.0)
        y0 = max(0.0, center_y - span_y / 2.0)
        y1 = min(benchmark.canvas.height, center_y + span_y / 2.0)
        span_x = max(x1 - x0, np.finfo(np.float64).eps)
        span_y = max(y1 - y0, np.finfo(np.float64).eps)
        density = (span_x + span_y) / (span_x * span_y)
        for ix0, ix1, fraction_x in _axis_overlap_groups(x0, x1, cell_w, grid):
            for iy0, iy1, fraction_y in _axis_overlap_groups(y0, y1, cell_h, grid):
                value = density * fraction_x * fraction_y
                difference[ix0, iy0] += value
                difference[ix1 + 1, iy0] -= value
                difference[ix0, iy1 + 1] -= value
                difference[ix1 + 1, iy1 + 1] += value
    return difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1)


def rudy_peak_tail(demand: np.ndarray, tail_fraction: float = 0.05):
    flat = np.asarray(demand, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(tail_fraction * flat.size)))
    top = flat if count == flat.size else np.partition(flat, flat.size - count)[-count:]
    return float(flat.max(initial=0.0)), float(top.mean())

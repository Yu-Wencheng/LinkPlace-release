from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import Benchmark, Rect
from .graph import ComponentPlan


@dataclass(frozen=True)
class GridGeometry:
    benchmark: Benchmark
    grid: int = 448

    @property
    def extent(self) -> float:
        """Square canvas extent used by the LinkPlace placement database.

        The Bookshelf loader derives both ``max_height`` and ``max_width`` from
        the maximum ``.pl`` macro extent.  Keeping this single value is
        essential: the original environment uses one ratio for both axes.
        """

        extents = []
        for name in self.benchmark.selected_macros:
            node = self.benchmark.nodes[name]
            if node.original_x is not None:
                extents.append(float(node.original_x) + node.width)
            if node.original_y is not None:
                extents.append(float(node.original_y) + node.height)
        if extents:
            return max(extents)
        return max(
            float(self.benchmark.canvas.x2),
            float(self.benchmark.canvas.y2),
            float(self.benchmark.canvas.width),
            float(self.benchmark.canvas.height),
        )

    @property
    def scale_x(self) -> float:
        return self.extent / self.grid

    @property
    def scale_y(self) -> float:
        return self.extent / self.grid

    @property
    def ratio(self) -> float:
        return self.extent / self.grid

    def footprint(self, name: str) -> Tuple[int, int]:
        node = self.benchmark.nodes[name]
        return (
            max(1, int(math.ceil(node.width / self.scale_x - 1e-12))),
            max(1, int(math.ceil(node.height / self.scale_y - 1e-12))),
        )

    def physical(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.ratio, gy * self.ratio

    def grid_rect(self, rect: Rect) -> Tuple[int, int, int, int]:
        x0 = int(math.floor(rect.x / self.ratio + 1e-12))
        y0 = int(math.floor(rect.y / self.ratio + 1e-12))
        x1 = int(math.ceil(rect.x2 / self.ratio - 1e-12))
        y1 = int(math.ceil(rect.y2 / self.ratio - 1e-12))
        return max(0, x0), max(0, y0), min(self.grid, x1), min(self.grid, y1)


@dataclass
class GreedyResult:
    legal: bool
    grid_placement: Dict[str, Tuple[int, int]]
    placement: Dict[str, Tuple[float, float]]
    component_placements: Dict[int, Dict[str, Tuple[float, float]]]
    component_relative_grid_placements: Dict[int, Dict[str, Tuple[int, int]]]
    component_relative_placements: Dict[int, Dict[str, Tuple[float, float]]]
    component_translations: Dict[int, Tuple[int, int]]
    component_relative_hpwl: Dict[int, float]
    blank_rectangle_history: List[Dict[str, object]]
    fallback_history: List[Dict[str, object]]
    attempt_history: List[Dict[str, object]]
    reserve: Tuple[int, int, int, int]
    reserve_scale: float
    attempts: int
    failure: Optional[str] = None


def occupancy_from_grid(
    geometry: GridGeometry,
    placement: Mapping[str, Tuple[int, int]],
) -> np.ndarray:
    occupied = np.zeros((geometry.grid, geometry.grid), dtype=np.bool_)
    for name, (gx, gy) in placement.items():
        sx, sy = geometry.footprint(name)
        occupied[gx : gx + sx, gy : gy + sy] = True
    return occupied


def _legal_mask(occupied: np.ndarray, sx: int, sy: int) -> np.ndarray:
    grid = occupied.shape[0]
    if sx > grid or sy > grid:
        return np.zeros((0, 0), dtype=np.bool_)
    prefix = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window = prefix[sx:, sy:] - prefix[:-sx, sy:] - prefix[sx:, :-sy] + prefix[:-sx, :-sy]
    return window == 0


def _reserve_rect(grid: int, reserve_area: float, canvas_area: float, scale: float):
    fraction = min(0.95, max(0.0, reserve_area / max(canvas_area, 1e-12))) * scale
    side_fraction = math.sqrt(fraction)
    width = min(grid, int(round(grid * side_fraction)))
    height = min(grid, int(round(grid * side_fraction)))
    x0 = (grid - width) // 2
    y0 = (grid - height) // 2
    return x0, y0, x0 + width, y0 + height


def _exclude_reserve(mask: np.ndarray, sx: int, sy: int, reserve):
    if mask.size == 0:
        return mask
    rx0, ry0, rx1, ry1 = reserve
    if rx0 >= rx1 or ry0 >= ry1:
        return mask
    xs = np.arange(mask.shape[0])[:, None]
    ys = np.arange(mask.shape[1])[None, :]
    intersects = (xs < rx1) & (xs + sx > rx0) & (ys < ry1) & (ys + sy > ry0)
    return mask & ~intersects


def _incremental_hpwl(
    benchmark: Benchmark,
    name: str,
    candidates_x: np.ndarray,
    candidates_y: np.ndarray,
    geometry: GridGeometry,
    placement: Mapping[str, Tuple[float, float]],
    include_fixed_terminals: bool = True,
) -> np.ndarray:
    node = benchmark.nodes[name]
    base_x = candidates_x.astype(np.float64) * geometry.ratio + node.width / 2.0
    base_y = candidates_y.astype(np.float64) * geometry.ratio + node.height / 2.0
    total = np.zeros(candidates_x.shape[0], dtype=np.float64)
    for net in benchmark.evaluated_nets:
        own_pins = [next((pin for pin in net.pins if pin.owner == name), None)]
        own_pins = [pin for pin in own_pins if pin is not None]
        if not own_pins:
            continue
        fixed_x = []
        fixed_y = []
        seen_owners = set()
        for pin in net.pins:
            if pin.owner in seen_owners:
                continue
            seen_owners.add(pin.owner)
            if pin.owner == name:
                continue
            if pin.owner in placement:
                other = benchmark.nodes[pin.owner]
                px, py = placement[pin.owner]
                fixed_x.append(float(px) + other.width / 2.0 + pin.offset_x)
                fixed_y.append(float(py) + other.height / 2.0 + pin.offset_y)
            elif include_fixed_terminals and pin.owner in benchmark.fixed_terminals:
                px, py = benchmark.fixed_terminals[pin.owner]
                fixed_x.append(float(px) + pin.offset_x)
                fixed_y.append(float(py) + pin.offset_y)
        if not fixed_x:
            continue
        own_x = np.stack([base_x + pin.offset_x for pin in own_pins], axis=0)
        own_y = np.stack([base_y + pin.offset_y for pin in own_pins], axis=0)
        min_x = np.minimum(own_x.min(axis=0), min(fixed_x))
        max_x = np.maximum(own_x.max(axis=0), max(fixed_x))
        min_y = np.minimum(own_y.min(axis=0), min(fixed_y))
        max_y = np.maximum(own_y.max(axis=0), max(fixed_y))
        total += (max_x - min_x) + (max_y - min_y)
    return total


def _minimum_choices(values: np.ndarray) -> np.ndarray:
    best = float(values.min()) if values.size else 0.0
    tolerance = max(1e-9, abs(best) * 1e-12)
    choices = np.flatnonzero(np.abs(values - best) <= tolerance)
    if choices.size == 0:
        choices = np.asarray([int(np.argmin(values))])
    return choices


def _relative_component_layout(
    benchmark: Benchmark,
    geometry: GridGeometry,
    plan: ComponentPlan,
    rng: random.Random,
):
    """Greedily minimize internal HPWL before the component sees the canvas."""

    if not plan.order:
        return {}
    if len(plan.order) == 1:
        return {plan.order[0]: (0, 0)}
    largest_footprint = max(max(geometry.footprint(name)) for name in plan.order)
    local_grid = geometry.grid * 2 + largest_footprint * 2
    occupied = np.zeros((local_grid, local_grid), dtype=np.bool_)
    local_grid_placement: Dict[str, Tuple[int, int]] = {}
    local_physical_placement: Dict[str, Tuple[float, float]] = {}

    for index, name in enumerate(plan.order):
        sx, sy = geometry.footprint(name)
        if index == 0:
            gx = (local_grid - sx) // 2
            gy = (local_grid - sy) // 2
        else:
            candidate_x, candidate_y = np.nonzero(_legal_mask(occupied, sx, sy))
            if candidate_x.size == 0:
                return None
            hpwl = _incremental_hpwl(
                benchmark,
                name,
                candidate_x,
                candidate_y,
                geometry,
                local_physical_placement,
                include_fixed_terminals=False,
            )
            choices = _minimum_choices(hpwl)
            chosen = int(choices[rng.randrange(choices.size)])
            gx = int(candidate_x[chosen])
            gy = int(candidate_y[chosen])
        occupied[gx : gx + sx, gy : gy + sy] = True
        local_grid_placement[name] = (gx, gy)
        local_physical_placement[name] = geometry.physical(gx, gy)

    origin_x = min(value[0] for value in local_grid_placement.values())
    origin_y = min(value[1] for value in local_grid_placement.values())
    relative = {
        name: (value[0] - origin_x, value[1] - origin_y)
        for name, value in local_grid_placement.items()
    }
    width = max(relative[name][0] + geometry.footprint(name)[0] for name in relative)
    height = max(relative[name][1] + geometry.footprint(name)[1] for name in relative)
    if width > geometry.grid or height > geometry.grid:
        return None
    return relative


def _component_translation_hpwl(
    benchmark: Benchmark,
    geometry: GridGeometry,
    relative: Mapping[str, Tuple[int, int]],
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    fixed_placement: Mapping[str, Tuple[float, float]],
    include_fixed_terminals: bool = True,
) -> np.ndarray:
    """Evaluate the component's HPWL for every rigid canvas translation."""

    members = set(relative)
    total = np.zeros(candidate_x.shape[0], dtype=np.float64)
    for net in benchmark.evaluated_nets:
        if not any(pin.owner in members for pin in net.pins):
            continue
        minimum_x = maximum_x = minimum_y = maximum_y = None
        point_count = 0
        for pin in net.pins:
            if pin.owner in relative:
                node = benchmark.nodes[pin.owner]
                rx, ry = relative[pin.owner]
                px = (
                    (candidate_x.astype(np.float64) + rx) * geometry.ratio
                    + node.width / 2.0
                    + pin.offset_x
                )
                py = (
                    (candidate_y.astype(np.float64) + ry) * geometry.ratio
                    + node.height / 2.0
                    + pin.offset_y
                )
            elif pin.owner in fixed_placement:
                node = benchmark.nodes[pin.owner]
                fixed_x, fixed_y = fixed_placement[pin.owner]
                px = np.full(candidate_x.shape, float(fixed_x) + node.width / 2.0 + pin.offset_x)
                py = np.full(candidate_y.shape, float(fixed_y) + node.height / 2.0 + pin.offset_y)
            elif include_fixed_terminals and pin.owner in benchmark.fixed_terminals:
                fixed_x, fixed_y = benchmark.fixed_terminals[pin.owner]
                px = np.full(candidate_x.shape, float(fixed_x) + pin.offset_x)
                py = np.full(candidate_y.shape, float(fixed_y) + pin.offset_y)
            else:
                continue
            if minimum_x is None:
                minimum_x = maximum_x = px
                minimum_y = maximum_y = py
            else:
                minimum_x = np.minimum(minimum_x, px)
                maximum_x = np.maximum(maximum_x, px)
                minimum_y = np.minimum(minimum_y, py)
                maximum_y = np.maximum(maximum_y, py)
            point_count += 1
        if point_count >= 2:
            total += (maximum_x - minimum_x) + (maximum_y - minimum_y)
    return total


def _largest_empty_rectangle(occupied: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the largest empty axis-aligned grid rectangle as [x0,y0,x1,y1]."""

    rows, columns = occupied.shape
    heights = np.zeros(columns, dtype=np.int32)
    best = (0, 0, 0, 0)
    best_area = 0
    for x in range(rows):
        heights = np.where(occupied[x], 0, heights + 1)
        stack = []
        for y in range(columns + 1):
            height = int(heights[y]) if y < columns else 0
            start = y
            while stack and stack[-1][1] > height:
                start_y, popped_height = stack.pop()
                area = popped_height * (y - start_y)
                candidate = (x + 1 - popped_height, start_y, x + 1, y)
                if area > best_area or (area == best_area and candidate < best):
                    best_area = area
                    best = candidate
                start = start_y
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def _surviving_blank_area(
    blank: Tuple[int, int, int, int],
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Largest part of ``blank`` surviving each translated component bbox."""

    blank_x0, blank_y0, blank_x1, blank_y1 = blank
    blank_width = blank_x1 - blank_x0
    blank_height = blank_y1 - blank_y0
    blank_area = blank_width * blank_height
    candidate_x1 = candidate_x + width
    candidate_y1 = candidate_y + height
    intersects = (
        (candidate_x < blank_x1)
        & (candidate_x1 > blank_x0)
        & (candidate_y < blank_y1)
        & (candidate_y1 > blank_y0)
    )
    left = np.maximum(0, np.minimum(candidate_x, blank_x1) - blank_x0) * blank_height
    right = np.maximum(0, blank_x1 - np.maximum(candidate_x1, blank_x0)) * blank_height
    bottom = blank_width * np.maximum(0, np.minimum(candidate_y, blank_y1) - blank_y0)
    top = blank_width * np.maximum(0, blank_y1 - np.maximum(candidate_y1, blank_y0))
    surviving = np.maximum.reduce((left, right, bottom, top)).astype(np.int64)
    surviving[~intersects] = blank_area
    return surviving


def _place_component(
    benchmark: Benchmark,
    geometry: GridGeometry,
    occupied: np.ndarray,
    fixed_placement: Mapping[str, Tuple[float, float]],
    relative: Mapping[str, Tuple[int, int]],
    blank_rectangle: Tuple[int, int, int, int],
    rng: random.Random,
    legal_mask_cache: Optional[Mapping[Tuple[int, int], np.ndarray]] = None,
):
    width = max(relative[name][0] + geometry.footprint(name)[0] for name in relative)
    height = max(relative[name][1] + geometry.footprint(name)[1] for name in relative)
    count_x = geometry.grid - width + 1
    count_y = geometry.grid - height + 1
    if count_x <= 0 or count_y <= 0:
        return None

    legal_translation = np.ones((count_x, count_y), dtype=np.bool_)
    for name, (relative_x, relative_y) in relative.items():
        sx, sy = geometry.footprint(name)
        macro_legal = (
            legal_mask_cache[(sx, sy)]
            if legal_mask_cache is not None and (sx, sy) in legal_mask_cache
            else _legal_mask(occupied, sx, sy)
        )
        legal_translation &= macro_legal[
            relative_x : relative_x + count_x,
            relative_y : relative_y + count_y,
        ]
    candidate_x, candidate_y = np.nonzero(legal_translation)
    if candidate_x.size == 0:
        return None

    blank_area = _surviving_blank_area(
        blank_rectangle, candidate_x, candidate_y, width, height
    )
    blank_choices = np.flatnonzero(blank_area == blank_area.max())
    del fixed_placement
    chosen = int(blank_choices[rng.randrange(blank_choices.size)])
    return (
        int(candidate_x[chosen]),
        int(candidate_y[chosen]),
        int(blank_area[chosen]),
    )


def _place_component_macros_individually(
    benchmark: Benchmark,
    geometry: GridGeometry,
    plan: ComponentPlan,
    occupied: np.ndarray,
    grid_placement: Dict[str, Tuple[int, int]],
    placement: Dict[str, Tuple[float, float]],
    rng: random.Random,
):
    """Fallback requested for small components that cannot be rigidly merged."""

    history = []
    component_grid = {}
    component_physical = {}
    for order_index, name in enumerate(plan.order):
        sx, sy = geometry.footprint(name)
        candidate_x, candidate_y = np.nonzero(_legal_mask(occupied, sx, sy))
        if candidate_x.size == 0:
            return None, {
                "status": "failed",
                "component_id": plan.component_id,
                "macro": name,
                "order_index": order_index,
                "reason": "no legal grid position",
                "history": history,
            }
        incremental = _incremental_hpwl(
            benchmark,
            name,
            candidate_x,
            candidate_y,
            geometry,
            placement,
            include_fixed_terminals=(benchmark.name == "ariane"),
        )
        hpwl_choices = _minimum_choices(incremental)
        blank_before = _largest_empty_rectangle(occupied)
        surviving = _surviving_blank_area(
            blank_before,
            candidate_x[hpwl_choices],
            candidate_y[hpwl_choices],
            sx,
            sy,
        )
        blank_choices = hpwl_choices[np.flatnonzero(surviving == surviving.max())]
        chosen = int(blank_choices[rng.randrange(blank_choices.size)])
        gx = int(candidate_x[chosen])
        gy = int(candidate_y[chosen])
        occupied[gx : gx + sx, gy : gy + sy] = True
        physical = geometry.physical(gx, gy)
        grid_placement[name] = (gx, gy)
        placement[name] = physical
        component_grid[name] = (gx, gy)
        component_physical[name] = physical
        blank_after = _largest_empty_rectangle(occupied)
        history.append(
            {
                "component_id": plan.component_id,
                "macro": name,
                "order_index": order_index,
                "legal_candidates": int(candidate_x.size),
                "minimum_incremental_hpwl": float(incremental[chosen]),
                "hpwl_tied_candidates": int(hpwl_choices.size),
                "blank_tied_candidates": int(blank_choices.size),
                "selected_grid": [gx, gy],
                "blank_before": list(blank_before),
                "blank_after": list(blank_after),
                "blank_area_after": int(
                    (blank_after[2] - blank_after[0])
                    * (blank_after[3] - blank_after[1])
                ),
            }
        )
    return (component_grid, component_physical), {
        "status": "complete",
        "component_id": plan.component_id,
        "history": history,
    }


def greedy_place_components(
    benchmark: Benchmark,
    plans: Sequence[ComponentPlan],
    seed: int,
    grid: int = 448,
    reserve_area: float = 0.0,
    shrink_step: float = 0.05,
    initial_grid_placement: Optional[Mapping[str, Tuple[int, int]]] = None,
    initial_placement: Optional[Mapping[str, Tuple[float, float]]] = None,
    allow_individual_fallback: bool = False,
) -> GreedyResult:
    geometry = GridGeometry(benchmark, grid)
    ordered = tuple(plans)
    # ``reserve_area`` remains in the public signature for runner compatibility.
    # The dynamic largest-blank-rectangle objective supersedes a fixed reserve.
    _ = reserve_area
    del shrink_step
    max_attempts = 1
    last_failure = None
    attempt_history: List[Dict[str, object]] = []
    for attempt in range(1, max_attempts + 1):
        rng = random.Random(int(seed) + attempt * 1000003)
        grid_placement: Dict[str, Tuple[int, int]] = dict(initial_grid_placement or {})
        placement: Dict[str, Tuple[float, float]] = dict(initial_placement or {})
        component_placements: Dict[int, Dict[str, Tuple[float, float]]] = {}
        component_relative_grid_placements: Dict[int, Dict[str, Tuple[int, int]]] = {}
        component_relative_placements: Dict[int, Dict[str, Tuple[float, float]]] = {}
        component_translations: Dict[int, Tuple[int, int]] = {}
        component_relative_hpwl: Dict[int, float] = {}
        blank_rectangle_history: List[Dict[str, object]] = []
        fallback_history: List[Dict[str, object]] = []
        occupied = occupancy_from_grid(geometry, grid_placement)
        failed = False
        for plan in ordered:
            relative = _relative_component_layout(benchmark, geometry, plan, rng)
            if relative is None:
                last_failure = "no local relative layout for component {} on attempt {}".format(
                    plan.component_id, attempt
                )
                failed = True
                break
            blank_before = _largest_empty_rectangle(occupied)
            translation = _place_component(
                benchmark, geometry, occupied, placement, relative, blank_before, rng
            )
            if translation is None:
                if allow_individual_fallback:
                    fallback_result, fallback_record = _place_component_macros_individually(
                        benchmark,
                        geometry,
                        plan,
                        occupied,
                        grid_placement,
                        placement,
                        rng,
                    )
                    fallback_history.append(fallback_record)
                    if fallback_result is not None:
                        actual_grid, actual_physical = fallback_result
                        component_placements[plan.component_id] = actual_physical
                        component_relative_grid_placements[plan.component_id] = dict(relative)
                        component_relative_placements[plan.component_id] = {
                            name: (
                                relative[name][0] * geometry.ratio,
                                relative[name][1] * geometry.ratio,
                            )
                            for name in relative
                        }
                        component_relative_hpwl[plan.component_id] = float(
                            _component_translation_hpwl(
                                benchmark,
                                geometry,
                                relative,
                                np.asarray([0], dtype=np.int64),
                                np.asarray([0], dtype=np.int64),
                                {},
                                include_fixed_terminals=False,
                            )[0]
                        )
                        blank_after = _largest_empty_rectangle(occupied)
                        blank_rectangle_history.append(
                            {
                                "sequence_index": len(blank_rectangle_history),
                                "component_id": plan.component_id,
                                "component_area": plan.area,
                                "merge_mode": "individual_macro_fallback",
                                "blank_before": list(blank_before),
                                "blank_after": list(blank_after),
                                "blank_area_after": int(
                                    (blank_after[2] - blank_after[0])
                                    * (blank_after[3] - blank_after[1])
                                ),
                            }
                        )
                        continue
                last_failure = "no legal rigid translation for component {} on attempt {}".format(
                    plan.component_id, attempt
                )
                if fallback_history and fallback_history[-1].get("status") == "failed":
                    last_failure = fallback_history[-1].get("reason", last_failure)
                failed = True
                break
            translate_x, translate_y, selected_blank_score = translation
            component_placements[plan.component_id] = {}
            component_relative_grid_placements[plan.component_id] = dict(relative)
            component_relative_placements[plan.component_id] = {
                name: (relative[name][0] * geometry.scale_x, relative[name][1] * geometry.scale_y)
                for name in relative
            }
            component_translations[plan.component_id] = (translate_x, translate_y)
            component_relative_hpwl[plan.component_id] = float(
                _component_translation_hpwl(
                    benchmark,
                    geometry,
                    relative,
                    np.asarray([0], dtype=np.int64),
                    np.asarray([0], dtype=np.int64),
                    {},
                    include_fixed_terminals=False,
                )[0]
            )
            for name in plan.order:
                relative_x, relative_y = relative[name]
                gx = translate_x + relative_x
                gy = translate_y + relative_y
                sx, sy = geometry.footprint(name)
                occupied[gx : gx + sx, gy : gy + sy] = True
                grid_placement[name] = (gx, gy)
                placement[name] = geometry.physical(gx, gy)
                component_placements[plan.component_id][name] = placement[name]
            blank_after = _largest_empty_rectangle(occupied)
            blank_rectangle_history.append(
                {
                    "sequence_index": len(blank_rectangle_history),
                    "component_id": plan.component_id,
                    "component_area": plan.area,
                    "merge_mode": "rigid_translation",
                    "blank_before": list(blank_before),
                    "blank_area_before": int((blank_before[2] - blank_before[0]) * (blank_before[3] - blank_before[1])),
                    "selected_surviving_blank_area": selected_blank_score,
                    "blank_after": list(blank_after),
                    "blank_area_after": int((blank_after[2] - blank_after[0]) * (blank_after[3] - blank_after[1])),
                }
            )
            if failed:
                break
        if not failed:
            final_blank = _largest_empty_rectangle(occupied)
            attempt_history.append(
                {
                    "attempt": attempt,
                    "status": "complete",
                    "placed_macros": len(grid_placement),
                }
            )
            return GreedyResult(
                legal=True,
                grid_placement=grid_placement,
                placement=placement,
                component_placements=component_placements,
                component_relative_grid_placements=component_relative_grid_placements,
                component_relative_placements=component_relative_placements,
                component_translations=component_translations,
                component_relative_hpwl=component_relative_hpwl,
                blank_rectangle_history=blank_rectangle_history,
                fallback_history=fallback_history,
                attempt_history=attempt_history,
                reserve=final_blank,
                reserve_scale=1.0,
                attempts=attempt,
            )
        attempt_history.append(
            {
                "attempt": attempt,
                "status": "failed",
                "failure": last_failure,
                "placed_macros": len(grid_placement),
                "blank_rectangle_history": blank_rectangle_history,
                "fallback_history": fallback_history,
            }
        )
    final_blank = _largest_empty_rectangle(occupied)
    return GreedyResult(
        legal=False,
        grid_placement=grid_placement,
        placement=placement,
        component_placements=component_placements,
        component_relative_grid_placements=component_relative_grid_placements,
        component_relative_placements=component_relative_placements,
        component_translations=component_translations,
        component_relative_hpwl=component_relative_hpwl,
        blank_rectangle_history=blank_rectangle_history,
        fallback_history=fallback_history,
        attempt_history=attempt_history,
        reserve=final_blank,
        reserve_scale=0.0,
        attempts=max_attempts,
        failure=last_failure or "greedy placement failed",
    )

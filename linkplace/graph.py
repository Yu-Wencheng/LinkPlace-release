from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .data import Benchmark


@dataclass(frozen=True)
class ComponentPlan:
    component_id: int
    macros: Tuple[str, ...]
    order: Tuple[str, ...]
    area: float

    @property
    def size(self) -> int:
        return len(self.macros)


def macro_graph(benchmark: Benchmark) -> Tuple[Mapping[str, Set[str]], Mapping[str, Set[str]]]:
    selected = set(benchmark.selected_macros)
    adjacency: Dict[str, Set[str]] = {name: set() for name in selected}
    node_to_nets: Dict[str, Set[str]] = {name: set() for name in selected}
    for net in benchmark.evaluated_nets:
        owners = tuple(dict.fromkeys(pin.owner for pin in net.pins if pin.owner in selected))
        for name in owners:
            node_to_nets[name].add(net.name)
        for index, left in enumerate(owners):
            for right in owners[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency, node_to_nets


def connected_components(benchmark: Benchmark) -> Tuple[Tuple[str, ...], ...]:
    adjacency, _ = macro_graph(benchmark)
    remaining = set(benchmark.selected_macros)
    components: List[Tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        remaining.remove(root)
        members = []
        while stack:
            name = stack.pop()
            members.append(name)
            for neighbor in sorted(adjacency[name], reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members)))
    components.sort(key=lambda item: item[0])
    return tuple(components)


def fixed_macro_order(
    benchmark: Benchmark,
    component: Sequence[str],
) -> Tuple[str, ...]:
    """Deterministic connected order approved for the paper experiments.

    The first macro has maximum area.  Every later macro must share at least
    one distinct net with the placed set; candidates are ordered by the number
    of such nets, then area, then macro name.
    """

    members = set(component)
    if not members:
        return ()
    adjacency, node_to_nets = macro_graph(benchmark)
    first = min(members, key=lambda name: (-benchmark.nodes[name].area, name))
    order = [first]
    visited = {first}
    placed_nets = set(node_to_nets[first])
    while len(order) < len(members):
        frontier = {
            candidate
            for placed in visited
            for candidate in adjacency[placed]
            if candidate in members and candidate not in visited
        }
        if not frontier:
            raise ValueError(
                "component is disconnected while constructing fixed order: {}".format(
                    sorted(members - visited)[:5]
                )
            )
        shared_net_count = {
            candidate: len(node_to_nets[candidate].intersection(placed_nets))
            for candidate in frontier
        }
        selected = min(
            frontier,
            key=lambda name: (
                -shared_net_count[name],
                -benchmark.nodes[name].area,
                name,
            ),
        )
        if shared_net_count[selected] <= 0:
            raise AssertionError("selected macro shares no net with the placed set")
        order.append(selected)
        visited.add(selected)
        placed_nets.update(node_to_nets[selected])
    return tuple(order)


def component_plans(benchmark: Benchmark, seed: int) -> Tuple[ComponentPlan, ...]:
    del seed  # The approved macro order is fixed and seed-independent.
    rows = []
    for component_id, members in enumerate(connected_components(benchmark)):
        area = sum(benchmark.nodes[name].area for name in members)
        order = fixed_macro_order(benchmark, members)
        rows.append(ComponentPlan(component_id, members, order, float(area)))
    rows.sort(key=lambda item: item.component_id)
    return tuple(rows)


def ordered_large_components(
    plans: Sequence[ComponentPlan],
    threshold: int = 20,
) -> Tuple[ComponentPlan, ...]:
    return tuple(
        sorted(
            (item for item in plans if item.size >= threshold),
            key=lambda item: (-item.size, -item.area, min(item.macros)),
        )
    )


def ordered_small_components(
    plans: Sequence[ComponentPlan],
    threshold: int = 20,
) -> Tuple[ComponentPlan, ...]:
    return tuple(
        sorted(
            (item for item in plans if item.size < threshold),
            key=lambda item: (-item.area, -item.size, min(item.macros)),
        )
    )


def component_statistics(benchmark: Benchmark, seed: int, threshold: int = 20):
    plans = component_plans(benchmark, seed)
    large = ordered_large_components(plans, threshold)
    small = ordered_small_components(plans, threshold)
    ordered = large + small
    largest = max((item.size for item in plans), default=0)
    return {
        "component_count": len(plans),
        "large_components": sum(item.size >= threshold for item in plans),
        "small_components": sum(1 < item.size < threshold for item in plans),
        "singletons": sum(item.size == 1 for item in plans),
        "largest_component_size": largest,
        "dominant_component_ratio": largest / max(1, len(benchmark.selected_macros)),
        "ordered_components": [
            {
                "sequence_index": index,
                "component_id": item.component_id,
                "size": item.size,
                "area": item.area,
                "kind": "large" if item.size >= threshold else ("singleton" if item.size == 1 else "small"),
                "macros": list(item.macros),
                "macro_order": list(item.order),
            }
            for index, item in enumerate(ordered)
        ],
    }

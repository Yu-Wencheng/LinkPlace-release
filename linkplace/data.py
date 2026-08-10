from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Node:
    name: str
    width: float
    height: float
    original_x: Optional[float] = None
    original_y: Optional[float] = None
    original_fixed: bool = False

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Pin:
    owner: str
    offset_x: float
    offset_y: float


@dataclass(frozen=True)
class Net:
    name: str
    pins: Tuple[Pin, ...]

    @property
    def owners(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(pin.owner for pin in self.pins))


@dataclass
class Benchmark:
    name: str
    source: str
    canvas: Rect
    nodes: Dict[str, Node]
    raw_net_count: int
    raw_pin_count: int
    evaluated_nets: List[Net]
    selected_macros: Tuple[str, ...]
    fixed_terminals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed_obstacles: Dict[str, Rect] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def evaluated_pin_count(self) -> int:
        return sum(len(net.pins) for net in self.evaluated_nets)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cache(benchmark: Benchmark, path: Path) -> Mapping[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "benchmark": {
            "name": benchmark.name,
            "source": benchmark.source,
            "canvas": [
                benchmark.canvas.x,
                benchmark.canvas.y,
                benchmark.canvas.width,
                benchmark.canvas.height,
            ],
            "nodes": {
                name: {
                    "width": node.width,
                    "height": node.height,
                    "original_x": node.original_x,
                    "original_y": node.original_y,
                    "original_fixed": node.original_fixed,
                }
                for name, node in benchmark.nodes.items()
            },
            "raw_net_count": benchmark.raw_net_count,
            "raw_pin_count": benchmark.raw_pin_count,
            "evaluated_nets": [
                {
                    "name": net.name,
                    "pins": [
                        [pin.owner, pin.offset_x, pin.offset_y] for pin in net.pins
                    ],
                }
                for net in benchmark.evaluated_nets
            ],
            "selected_macros": list(benchmark.selected_macros),
            "fixed_terminals": benchmark.fixed_terminals,
            "fixed_obstacles": {
                name: [rect.x, rect.y, rect.width, rect.height]
                for name, rect in benchmark.fixed_obstacles.items()
            },
            "metadata": benchmark.metadata,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)
    manifest = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "benchmark": benchmark.name,
        "selected_macros": len(benchmark.selected_macros),
        "raw_nets": benchmark.raw_net_count,
        "raw_pins": benchmark.raw_pin_count,
        "evaluated_nets": len(benchmark.evaluated_nets),
        "evaluated_pins": benchmark.evaluated_pin_count,
        "fixed_terminals": len(benchmark.fixed_terminals),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_cache(path: Path) -> Benchmark:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != 1:
        raise ValueError("unsupported benchmark cache version: {}".format(payload.get("version")))
    item = payload["benchmark"]
    canvas = Rect(*map(float, item["canvas"]))
    nodes = {
        name: Node(
            name=name,
            width=float(row["width"]),
            height=float(row["height"]),
            original_x=(None if row.get("original_x") is None else float(row["original_x"])),
            original_y=(None if row.get("original_y") is None else float(row["original_y"])),
            original_fixed=bool(row.get("original_fixed", False)),
        )
        for name, row in item["nodes"].items()
    }
    nets = [
        Net(
            name=row["name"],
            pins=tuple(Pin(str(owner), float(dx), float(dy)) for owner, dx, dy in row["pins"]),
        )
        for row in item["evaluated_nets"]
    ]
    return Benchmark(
        name=str(item["name"]),
        source=str(item.get("source", path)),
        canvas=canvas,
        nodes=nodes,
        raw_net_count=int(item["raw_net_count"]),
        raw_pin_count=int(item["raw_pin_count"]),
        evaluated_nets=nets,
        selected_macros=tuple(item["selected_macros"]),
        fixed_terminals={
            name: (float(value[0]), float(value[1]))
            for name, value in item.get("fixed_terminals", {}).items()
        },
        fixed_obstacles={
            name: Rect(*map(float, value))
            for name, value in item.get("fixed_obstacles", {}).items()
        },
        metadata=dict(item.get("metadata", {})),
    )


def from_external_benchmark(value, metadata: Optional[Mapping[str, object]] = None) -> Benchmark:
    """Detach a parsed benchmark from the old data-preparation package."""

    selected = tuple(value.selected_macros)
    nodes = {
        name: Node(
            name=name,
            width=float(value.nodes[name].width),
            height=float(value.nodes[name].height),
            original_x=(None if value.nodes[name].original_x is None else float(value.nodes[name].original_x)),
            original_y=(None if value.nodes[name].original_y is None else float(value.nodes[name].original_y)),
            original_fixed=bool(value.nodes[name].original_fixed),
        )
        for name in selected
    }
    nets = [
        Net(
            str(net.name),
            tuple(Pin(str(pin.owner), float(pin.offset_x), float(pin.offset_y)) for pin in net.pins),
        )
        for net in value.evaluated_nets
    ]
    return Benchmark(
        name=str(value.name),
        source=str(value.source_aux),
        canvas=Rect(
            float(value.canvas.x),
            float(value.canvas.y),
            float(value.canvas.width),
            float(value.canvas.height),
        ),
        nodes=nodes,
        raw_net_count=int(value.raw_net_count),
        raw_pin_count=int(value.raw_pin_count),
        evaluated_nets=nets,
        selected_macros=selected,
        fixed_terminals={
            str(name): (float(point[0]), float(point[1]))
            for name, point in value.fixed_terminals.items()
        },
        fixed_obstacles={
            str(name): Rect(float(rect.x), float(rect.y), float(rect.width), float(rect.height))
            for name, rect in value.fixed_obstacles.items()
        },
        metadata=dict(metadata or {}),
    )


def benchmark_summary(benchmark: Benchmark) -> Mapping[str, object]:
    return {
        "benchmark": benchmark.name,
        "source": benchmark.source,
        "canvas": {
            "x": benchmark.canvas.x,
            "y": benchmark.canvas.y,
            "width": benchmark.canvas.width,
            "height": benchmark.canvas.height,
        },
        "selected_macros": len(benchmark.selected_macros),
        "macro_area": sum(benchmark.nodes[name].area for name in benchmark.selected_macros),
        "area_utilization": sum(benchmark.nodes[name].area for name in benchmark.selected_macros) / benchmark.canvas.area,
        "raw_nets": benchmark.raw_net_count,
        "raw_pins": benchmark.raw_pin_count,
        "evaluated_nets": len(benchmark.evaluated_nets),
        "evaluated_pins": benchmark.evaluated_pin_count,
        "fixed_terminals": len(benchmark.fixed_terminals),
        "fixed_obstacles": len(benchmark.fixed_obstacles),
    }

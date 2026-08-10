#!/usr/bin/env python3
from pathlib import Path

from linkplace.data import load_cache


def main():
    benchmark = load_cache(Path("datasets/cache/adaptec1.json.gz"))
    values = [
        (
            name,
            benchmark.nodes[name].original_x,
            benchmark.nodes[name].original_y,
            benchmark.nodes[name].width,
            benchmark.nodes[name].height,
        )
        for name in benchmark.selected_macros
    ]
    valid = [row for row in values if row[1] is not None and row[2] is not None]
    print("selected", len(values), "missing_original", len(values) - len(valid))
    print("extent", max(max(x + width, y + height) for _, x, y, width, height in valid))
    print("max_x", max(x + width for _, x, _, width, _ in valid))
    print("max_y", max(y + height for _, _, y, _, height in valid))
    print("canvas", benchmark.canvas)


if __name__ == "__main__":
    main()

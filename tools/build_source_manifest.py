#!/usr/bin/env python3
"""Write deterministic SHA-256 hashes for the lightweight release tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "outputs", "datasets", "third_party"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="SOURCE_MANIFEST.sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        records.append(f"{digest(path)}  {relative.as_posix()}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(f"wrote {len(records)} hashes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

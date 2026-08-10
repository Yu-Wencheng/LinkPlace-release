#!/usr/bin/env python3
"""Create self-contained LinkPlace benchmark caches from prepared source data.

The old project is used only as a one-time, read-only parser.  Formal training
reads only the caches written inside this LinkPlace checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from linkplace.data import from_external_benchmark, save_cache, sha256_file


ISPD_FULL = ("adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3")
ICCAD = ("superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18")


def _selection(path: Path):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value.split()[0])
    if len(values) != len(set(values)):
        raise ValueError("duplicate macro in selection: {}".format(path))
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Checkout containing the external benchmark parsers and source datasets.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/cache"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    sys.path.insert(0, str(source_root))
    from codeplace.ariane import load_ariane
    from codeplace.bookshelf import load_ispd_bookshelf
    from codeplace.iccad2015 import load_iccad2015

    requested = set(args.only)
    jobs = []
    for name in ISPD_FULL:
        jobs.append((name, "bookshelf", source_root / "datasets" / "ispd2005" / name / (name + ".aux"), None))
    for name in ("bigblue2", "bigblue4"):
        key = name + "-1024"
        jobs.append((key, "bookshelf", source_root / "datasets" / "projected" / "ispd2005" / key / (name + ".aux"), None))
    jobs.append(("ariane", "ariane", source_root / "datasets" / "ariane" / "netlist.pb.txt", source_root / "datasets" / "ariane" / "laiyao_pb2.py"))
    for name in ICCAD:
        jobs.append(
            (
                name + "-512",
                "iccad",
                source_root / "datasets" / "iccad2015-oci" / name / (name + ".iccad2015"),
                source_root / "datasets" / "selections" / "iccad2015-area" / name / "area-ranked-512.txt",
            )
        )

    manifests = {}
    for key, kind, source, auxiliary in jobs:
        if requested and key not in requested:
            continue
        output = output_root / (key + ".json.gz")
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        if output.exists() and manifest_path.exists() and not args.force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if sha256_file(output) == manifest.get("sha256"):
                manifests[key] = manifest
                print("[cache] reuse {}".format(output), flush=True)
                continue
        print("[cache] parsing {} from {}".format(key, source), flush=True)
        if kind == "bookshelf":
            external = load_ispd_bookshelf(source)
            diagnostics = {"format": kind, "selection": "terminal macros"}
        elif kind == "ariane":
            external = load_ariane(source, auxiliary)
            diagnostics = {"format": kind, "selection": "all macros"}
        elif kind == "iccad":
            selected = _selection(auxiliary)
            if len(selected) != 512:
                raise ValueError("{} contains {} macros, expected 512".format(auxiliary, len(selected)))
            external, diagnostics = load_iccad2015(source, selected)
            diagnostics = dict(diagnostics)
            diagnostics.update({"format": kind, "selection_file": str(auxiliary), "selection_file_sha256": sha256_file(auxiliary)})
        else:
            raise AssertionError(kind)
        benchmark = from_external_benchmark(external, diagnostics)
        manifests[key] = save_cache(benchmark, output)
        print("[cache] wrote {}".format(output), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    index = {
        "version": 1,
        "source_root": str(source_root),
        "benchmarks": manifests,
    }
    (output_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "benchmarks": len(manifests), "index": str(output_root / "index.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

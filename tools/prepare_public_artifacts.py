#!/usr/bin/env python3
"""Copy small paper tables into the release while removing machine-local paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLES = {
    "main_mean_std.csv": set(),
    "main_seed_results.csv": {"result_path", "final_placement"},
    "grid_ablation_five_seed_mean_std.csv": set(),
    "grid_ablation_five_seed_results.csv": {"result_path"},
    "monolithic_iccad2015_mean_std.csv": set(),
    "monolithic_iccad2015_seed_results.csv": {"result_path"},
    "baseline_mean_std.csv": set(),
    "baseline_seed_results.csv": {"attempt_dir", "source_root"},
}

OUTPUT_NAMES = {
    "monolithic_iccad2015_mean_std.csv": "linkplace_m_iccad2015_mean_std.csv",
    "monolithic_iccad2015_seed_results.csv": "linkplace_m_iccad2015_seed_results.csv",
}

VARIANT_NAMES = {
    "codeplace": "linkplace-c",
    "current": "linkplace-c",
    "current-best-of-5": "linkplace-c-best-of-5",
    "monolithic": "linkplace-m",
}


def sanitize_csv(source: Path, destination: Path, dropped: set[str]) -> dict[str, object]:
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in dropped]
        rows = []
        for row in reader:
            clean = {name: row.get(name, "") for name in fieldnames}
            if "variant" in clean:
                clean["variant"] = VARIANT_NAMES.get(clean["variant"], clean["variant"])
            rows.append(clean)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"source_name": source.name, "output_name": destination.name, "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Directory containing paper table CSV files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tables"))
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    records = []
    missing = []
    for name, dropped in TABLES.items():
        source = source_root / name
        if not source.exists():
            missing.append(name)
            continue
        destination = output_root / OUTPUT_NAMES.get(name, name)
        records.append(sanitize_csv(source, destination, dropped))

    manifest = {
        "schema_version": 1,
        "naming": {"codeplace": "LinkPlace-C", "monolithic": "LinkPlace-M"},
        "tables": records,
        "missing": missing,
        "note": "Machine-local result and source directory columns were intentionally removed.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

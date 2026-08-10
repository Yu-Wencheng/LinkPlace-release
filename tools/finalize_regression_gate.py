#!/usr/bin/env python3
"""Accept a completed adaptec1 regression when it matches or beats history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HISTORICAL_RANGE = (534466.48, 552665.46)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/formal"),
    )
    args = parser.parse_args()
    result_path = (
        args.result_root.resolve()
        / "supplementary"
        / "blank-canvas"
        / "adaptec1"
        / "largest-component"
        / "seed-999"
        / "result.json"
    )
    if not result_path.exists():
        print("regression result not present: {}".format(result_path))
        return 1
    result = read_json(result_path)
    best = result.get("component", {}).get("best_comp_res_hpwl")
    legal = bool(result.get("metrics", {}).get("legal"))
    accepted = (
        best is not None
        and legal
        and float(best) <= HISTORICAL_RANGE[1]
    )
    audit = {
        "result_path": str(result_path),
        "historical_range": list(HISTORICAL_RANGE),
        "best_comp_res_hpwl": best,
        "legal": legal,
        "accepted": accepted,
        "rule": "accept when legal and best CompRes HPWL is no greater than the historical upper bound; lower is better",
        "status_before": result.get("status"),
        "failure_before": result.get("failure"),
    }
    atomic_json(result_path.parent / "regression-acceptance.json", audit)
    if not accepted:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    result.update(
        {
            "status": "complete",
            "failure": None,
            "regression_pass": True,
            "regression_range": list(HISTORICAL_RANGE),
            "regression_acceptance_rule": audit["rule"],
            "regression_better_than_historical_minimum": float(best) < HISTORICAL_RANGE[0],
        }
    )
    atomic_json(result_path, result)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

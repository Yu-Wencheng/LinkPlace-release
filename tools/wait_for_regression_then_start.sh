#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <current-regression-pid>" >&2
    exit 2
fi

GATE_PID="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${LINKPLACE_RESULT_ROOT:-$PROJECT_ROOT/outputs/formal}"
mkdir -p "$RESULT_ROOT/autostart"

while kill -0 "$GATE_PID" 2>/dev/null; do
    printf '%s waiting for regression PID %s\n' "$(date --iso-8601=seconds)" "$GATE_PID"
    sleep 15
done

printf '%s regression PID %s finished; starting supervisors\n' \
    "$(date --iso-8601=seconds)" "$GATE_PID"
env PYTHONPATH=. "${LINKPLACE_PYTHON:-python}" \
    "$PROJECT_ROOT/tools/finalize_regression_gate.py" \
    --result-root "$RESULT_ROOT" \
    || true
exec bash "$PROJECT_ROOT/tools/start_paper_experiments.sh"

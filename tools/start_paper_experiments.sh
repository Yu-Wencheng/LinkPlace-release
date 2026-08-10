#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${LINKPLACE_RESULT_ROOT:-$PROJECT_ROOT/outputs/formal}"
TRAIN_PYTHON="${LINKPLACE_PYTHON:-python}"
POST_PYTHON="${LINKPLACE_POST_PYTHON:-$TRAIN_PYTHON}"
GPUS="${PAPER_GPUS:-0,1,2}"

mkdir -p "$RESULT_ROOT/queue" "$RESULT_ROOT/postprocess"
cd "$PROJECT_ROOT"

start_if_needed() {
    local name="$1"
    local pid_file="$2"
    local log_file="$3"
    shift 3
    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "$name already running as PID $old_pid"
            return
        fi
    fi
    nohup "$@" >"$log_file" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    echo "$name started as PID $pid"
}

start_if_needed \
    "training queue" \
    "$RESULT_ROOT/queue/supervisor.pid" \
    "$RESULT_ROOT/queue/supervisor.log" \
    env PYTHONPATH=. PYTHONUNBUFFERED=1 \
    "$TRAIN_PYTHON" tools/run_paper_queue.py \
    --python "$TRAIN_PYTHON" \
    --result-root "$RESULT_ROOT" \
    --cache-root "$PROJECT_ROOT/datasets/cache" \
    --episodes 1000 \
    --gpus "$GPUS"

start_if_needed \
    "postprocess watcher" \
    "$RESULT_ROOT/postprocess/watcher.pid" \
    "$RESULT_ROOT/postprocess/watcher.log" \
    env PYTHONPATH=. PYTHONUNBUFFERED=1 \
    "$POST_PYTHON" tools/run_postprocess_pipeline.py \
    --python "$POST_PYTHON" \
    --result-root "$RESULT_ROOT" \
    --gpus "$GPUS"

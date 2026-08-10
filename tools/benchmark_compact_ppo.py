#!/usr/bin/env python3
"""Measure one full-buffer LinkPlace PPO update without placement episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from linkplace.models.ppo import PPO


class _CompletedEpisodeEnvironment:
    def __init__(self, macro_count: int):
        self.t = int(macro_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=448)
    parser.add_argument("--macros", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    namespace = SimpleNamespace(
        grid=args.grid,
        A_lr=1e-3,
        C_lr=1e-4,
        batch_size=args.batch_size,
        gamma=0.95,
        pnm=args.macros,
        device=args.device,
        buffer_device="auto",
        compact_state_buffer=True,
    )
    environment = _CompletedEpisodeEnvironment(args.macros)
    agent = PPO(environment, namespace)
    agent.ppo_epoch = args.ppo_epochs
    device = torch.device(args.device)
    state = torch.zeros((3, args.grid, args.grid), dtype=torch.float32, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocation_started = time.perf_counter()
    agent._allocate(state)
    if not agent.compact_state_buffer:
        raise RuntimeError("benchmark requires the compact state buffer")
    agent._canvas_codes.zero_()
    agent._wire_masks.zero_()
    agent._position_masks.zero_()
    agent._actions.zero_()
    agent._rewards.uniform_(-1.0, 1.0)
    agent._old_log_probs.fill_(-12.0)
    agent.buffer_count = agent.buffer_capacity
    agent.counter = agent.buffer_capacity
    torch.cuda.synchronize(device)
    allocation_seconds = time.perf_counter() - allocation_started
    allocated_before_update = int(torch.cuda.memory_allocated(device))
    reserved_before_update = int(torch.cuda.memory_reserved(device))

    writer = None
    if args.tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(args.tensorboard_dir))

    update_started = time.perf_counter()
    agent.update(writer)
    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    if writer is not None:
        writer.close()

    payload = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "grid": args.grid,
        "macros": args.macros,
        "buffer_capacity": agent.buffer_capacity,
        "batch_size": args.batch_size,
        "ppo_epochs_measured": args.ppo_epochs,
        "training_steps": agent.training_step,
        "buffer_device": str(agent.buffer_device),
        "buffer_memory_bytes": agent.buffer_memory_bytes,
        "allocation_seconds": allocation_seconds,
        "allocated_before_update_bytes": allocated_before_update,
        "reserved_before_update_bytes": reserved_before_update,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "update_seconds": update_seconds,
        "seconds_per_training_step": update_seconds / max(1, agent.training_step),
        "estimated_ten_epoch_update_seconds": update_seconds * (10.0 / args.ppo_epochs),
        "tensorboard_enabled": writer is not None,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

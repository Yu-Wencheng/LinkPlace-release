from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .models.ppo import PPO, Transition
from .evaluation.comp_res import comp_res_hpwl

from .data import Benchmark, Rect, benchmark_summary, load_cache
from .graph import (
    ComponentPlan,
    component_plans,
    component_statistics,
    ordered_large_components,
    ordered_small_components,
)
from .greedy import (
    GridGeometry,
    _largest_empty_rectangle,
    _legal_mask,
    _place_component,
    greedy_place_components,
    occupancy_from_grid,
)
from .metrics import macro_hpwl, rudy_map, rudy_peak_tail, validate_placement
from .component_env import ComponentPlaceDB, InitialCanvasPlaceEnv


SEEDS = (999, 1000, 1001, 1002, 1003)
ISPD_FULL = (
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue2",
    "bigblue3",
    "bigblue4",
)
ICCAD2015_DERIVED = (
    "superblue1",
    "superblue3",
    "superblue4",
    "superblue5",
    "superblue7",
    "superblue10",
    "superblue16",
    "superblue18",
)
MAIN_BENCHMARKS = ISPD_FULL + ("ariane",) + ICCAD2015_DERIVED
ABLATION_BENCHMARKS = ISPD_FULL + ("ariane",) + ICCAD2015_DERIVED
LINKPLACE_C = "linkplace-c"
LINKPLACE_M = "linkplace-m"
LEGACY_LINKPLACE_M = "monolithic"


def normalize_variant(variant: str) -> str:
    """Return the public variant name while accepting legacy result labels."""
    if variant == LEGACY_LINKPLACE_M:
        return LINKPLACE_M
    return variant


def atomic_json(path: Path, payload: Mapping[str, object]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_key(name: str) -> str:
    if name in {"bigblue2", "bigblue4"}:
        return name + "-1024"
    if name.startswith("superblue"):
        return name + "-512"
    return name


def load_named_benchmark(name: str, cache_root: Path) -> Benchmark:
    path = Path(cache_root) / (cache_key(name) + ".json.gz")
    if not path.exists():
        raise FileNotFoundError("benchmark cache missing: {}".format(path))
    return load_cache(path)


def grid_to_physical(benchmark: Benchmark, grid_placement, grid: int = 448):
    geometry = GridGeometry(benchmark, grid)
    return {name: geometry.physical(value[0], value[1]) for name, value in grid_placement.items()}


def render_layout(benchmark: Benchmark, placement, path: Path, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import patches

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 7.5))
    geometry = GridGeometry(benchmark)
    axis.set_xlim(0.0, geometry.extent)
    axis.set_ylim(0.0, geometry.extent)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title or benchmark.name)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    for name in benchmark.selected_macros:
        if name not in placement:
            continue
        node = benchmark.nodes[name]
        x, y = placement[name]
        axis.add_patch(
            patches.Rectangle(
                (x, y), node.width, node.height, facecolor="#4C78A8", edgecolor="black", linewidth=0.25, alpha=0.72
            )
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_training_curves(csv_path: Path, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with Path(csv_path).open("r", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    epochs = [int(item["epoch"]) for item in rows]
    rewards = [float(item["reward"]) for item in rows]
    hpwl = [
        float(item["comp_res_hpwl"]) if item["comp_res_hpwl"] else np.nan
        for item in rows
    ]
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(epochs, rewards, linewidth=0.8)
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, hpwl, linewidth=0.8)
    axes[1].set_ylabel("CompRes HPWL")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_rudy(demand, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(np.asarray(demand).T, origin="lower", cmap="magma", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("x bin")
    axis.set_ylabel("y bin")
    figure.colorbar(image, ax=axis, label="RUDY")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_placement(path: Path, placement):
    atomic_json(
        path,
        {"placement": {name: [float(value[0]), float(value[1])] for name, value in sorted(placement.items())}},
    )


def save_grid_placement(path: Path, placement):
    atomic_json(
        path,
        {"grid_placement": {name: [int(value[0]), int(value[1])] for name, value in sorted(placement.items())}},
    )


def load_saved_placement(path: Path, key: str = "placement"):
    item = read_json(path)[key]
    cast = int if key == "grid_placement" else float
    return {name: (cast(value[0]), cast(value[1])) for name, value in item.items()}


def _model_snapshot(agent: PPO):
    return {
        "actor_net_dict": {name: value.detach().cpu().clone() for name, value in agent.actor_net.state_dict().items()},
        "critic_net_dict": {name: value.detach().cpu().clone() for name, value in agent.critic_net.state_dict().items()},
        "training_step": agent.training_step,
        "counter": agent.counter,
    }


def _component_execution_dir(parent: Path, reuse_completed: bool = True):
    parent.mkdir(parents=True, exist_ok=True)
    successful = []
    for child in sorted(parent.glob("execution-*")):
        summary = child / "summary.json"
        if summary.exists():
            item = read_json(summary)
            if item.get("status") == "complete":
                successful.append((child, item))
    if successful and reuse_completed:
        return successful[-1][0], successful[-1][1], True
    index = 0
    while (parent / "execution-{:02d}".format(index)).exists():
        index += 1
    directory = parent / "execution-{:02d}".format(index)
    directory.mkdir(parents=True)
    return directory, None, False


def train_component(
    benchmark: Benchmark,
    plan: ComponentPlan,
    fixed_grid: Mapping[str, Tuple[int, int]],
    fixed_physical: Mapping[str, Tuple[float, float]],
    output_parent: Path,
    seed: int,
    episodes: int = 1000,
    grid: int = 448,
    device: str = "cuda",
    checkpoint_interval: int = 100,
    tensorboard: bool = True,
    reuse_completed: bool = True,
):
    execution, previous, reused = _component_execution_dir(
        Path(output_parent), reuse_completed=reuse_completed
    )
    if reused:
        return previous
    started = time.time()
    configuration = {
        "benchmark": benchmark.name,
        "component_id": plan.component_id,
        "component_size": plan.size,
        "component_area": plan.area,
        "macro_order": list(plan.order),
        "seed": int(seed),
        "episodes": int(episodes),
        "grid": int(grid),
        "gamma": 0.95,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-4,
        "batch_size": 64,
        "ppo_clip": 0.2,
        "ppo_update_epochs": 10,
        "compact_state_buffer": True,
        "minibatch_loss_logging": "batched-device-transfer",
        "fixed_macros": len(fixed_grid),
        "device": device,
        "selection_metric": "original a1_448 utils/comp_res.py HPWL",
        "relative_layout_source": "minimum-comp_res legal LinkPlace episode on the fixed initial canvas",
        "rigid_translation_objective": [
            "maximum surviving blank rectangle",
            "seeded random tie-break",
        ],
    }
    atomic_json(execution / "configuration.json", configuration)
    atomic_json(execution / "status.json", {"status": "running", "started_at": started, **configuration})
    seed_everything(seed)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    placedb = ComponentPlaceDB(benchmark, plan.macros, plan.order)
    env = InitialCanvasPlaceEnv(placedb, fixed_grid, grid=grid, device=device)
    args = SimpleNamespace(
        grid=grid,
        A_lr=1e-3,
        C_lr=1e-4,
        batch_size=64,
        gamma=0.95,
        pnm=plan.size,
        device=device,
        buffer_device="auto",
        compact_state_buffer=True,
    )
    agent = PPO(env, args)
    writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(execution / "tensorboard"))
        except Exception as error:
            atomic_json(execution / "tensorboard-unavailable.json", {"error": repr(error)})

    log_path = execution / "train.csv"
    stream = log_path.open("w", newline="", encoding="utf-8")
    columns = [
        "epoch",
        "reward",
        "running_reward",
        "training_step",
        "comp_res_hpwl",
        "legal",
        "illegal_actions",
        "epoch_seconds",
        "legal_action_count_at_end",
    ]
    log = csv.DictWriter(stream, fieldnames=columns)
    log.writeheader()
    stream.flush()
    best_hpwl = math.inf
    best_epoch = None
    best_policy_grid = None
    best_model = None
    running_reward = None
    legal_episodes = 0
    geometry = GridGeometry(benchmark, grid)
    fixed_occupied = occupancy_from_grid(geometry, fixed_grid)
    blank_before = _largest_empty_rectangle(fixed_occupied)

    try:
        for epoch in range(episodes):
            epoch_started = time.time()
            state = env.reset()
            done = False
            total_reward = 0.0
            illegal_actions = 0
            while not done:
                stored_state = state.clone()
                action, log_probability = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                illegal_actions += int(bool(info.get("illegal_action", False)))
                transition = Transition(stored_state, action, reward, log_probability, next_state)
                if agent.store_transition(transition):
                    agent.update(writer)
                total_reward += reward
                state = next_state
            running_reward = total_reward if running_reward is None else running_reward * 0.9 + total_reward * 0.1
            legal = not env.had_illegal_action and len(env.node_pos) == plan.size
            episode_hpwl = math.inf
            if legal:
                policy_grid = {
                    name: (int(value[0]), int(value[1]))
                    for name, value in env.node_pos.items()
                }
                episode_hpwl = float(comp_res_hpwl(placedb, env.node_pos, env.ratio))
                legal_episodes += 1
                if episode_hpwl < best_hpwl:
                    best_hpwl = episode_hpwl
                    best_epoch = epoch
                    best_policy_grid = policy_grid
                    best_model = _model_snapshot(agent)
                    save_grid_placement(execution / "best-policy-grid-placement.json", best_policy_grid)
                    atomic_json(
                        execution / "best-raw-metadata.json",
                        {
                            "epoch": epoch,
                            "comp_res_hpwl": best_hpwl,
                            "reward": total_reward,
                        },
                    )
            elapsed = time.time() - epoch_started
            log.writerow(
                {
                    "epoch": epoch,
                    "reward": total_reward,
                    "running_reward": running_reward,
                    "training_step": agent.training_step,
                    "comp_res_hpwl": "" if not legal else episode_hpwl,
                    "legal": int(legal),
                    "illegal_actions": illegal_actions,
                    "epoch_seconds": elapsed,
                    "legal_action_count_at_end": env.legal_action_count,
                }
            )
            stream.flush()
            if writer is not None:
                writer.add_scalar("reward/episode", total_reward, epoch)
                writer.add_scalar("reward/running", running_reward, epoch)
                if legal:
                    writer.add_scalar("eval/comp_res_hpwl", episode_hpwl, epoch)
                writer.add_scalar("eval/legal", int(legal), epoch)
                writer.add_scalar("runtime/epoch_seconds", elapsed, epoch)
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = execution / "checkpoints" / "epoch-{:04d}.pth".format(epoch + 1)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                agent.save_param(checkpoint)
                atomic_json(
                    execution / "status.json",
                    {
                        "status": "running",
                        "completed_episodes": epoch + 1,
                        "legal_episodes": legal_episodes,
                        "best_comp_res_hpwl": None if best_epoch is None else best_hpwl,
                        **configuration,
                    },
                )
        best_grid = None
        best_physical = None
        best_relative_grid = None
        best_translation = None
        best_blank_rectangle = None
        best_blank_area = None
        translation_failure = None
        if best_policy_grid is not None:
            origin_x = min(value[0] for value in best_policy_grid.values())
            origin_y = min(value[1] for value in best_policy_grid.values())
            best_relative_grid = {
                name: (value[0] - origin_x, value[1] - origin_y)
                for name, value in best_policy_grid.items()
            }
            translated = _place_component(
                benchmark,
                geometry,
                fixed_occupied,
                fixed_physical,
                best_relative_grid,
                blank_before,
                random.Random(int(seed) + plan.component_id * 10007),
                legal_mask_cache={
                    footprint: _legal_mask(fixed_occupied, *footprint)
                    for footprint in sorted({geometry.footprint(name) for name in plan.macros})
                },
            )
            if translated is None:
                translation_failure = "minimum-comp_res layout has no legal rigid translation"
            else:
                translate_x, translate_y, _ = translated
                best_translation = (translate_x, translate_y)
                best_grid = {
                    name: (
                        best_relative_grid[name][0] + translate_x,
                        best_relative_grid[name][1] + translate_y,
                    )
                    for name in plan.order
                }
                best_physical = {
                    name: geometry.physical(*best_grid[name]) for name in plan.order
                }
                occupied_after = fixed_occupied.copy()
                for name, (gx, gy) in best_grid.items():
                    sx, sy = geometry.footprint(name)
                    occupied_after[gx : gx + sx, gy : gy + sy] = True
                best_blank_rectangle = _largest_empty_rectangle(occupied_after)
                best_blank_area = int(
                    (best_blank_rectangle[2] - best_blank_rectangle[0])
                    * (best_blank_rectangle[3] - best_blank_rectangle[1])
                )
                save_grid_placement(execution / "best-grid-placement.json", best_grid)
                save_placement(execution / "best-placement.json", best_physical)
                save_grid_placement(execution / "best-relative-grid-placement.json", best_relative_grid)
                atomic_json(
                    execution / "best-metadata.json",
                    {
                        "epoch": best_epoch,
                        "comp_res_hpwl": best_hpwl,
                        "blank_rectangle": list(best_blank_rectangle),
                        "blank_rectangle_area": best_blank_area,
                        "translation_grid": list(best_translation),
                    },
                )
        agent.save_param(execution / "final-model.pth")
        if best_model is not None:
            torch.save(best_model, execution / "best-model.pth")
        peak_gpu = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        summary = {
            "status": "complete" if best_grid is not None else "failed",
            "failure": (
                None
                if best_grid is not None
                else (translation_failure or "no legal episode")
            ),
            "benchmark": benchmark.name,
            "component_id": plan.component_id,
            "component_size": plan.size,
            "seed": seed,
            "episodes": episodes,
            "legal_episodes": legal_episodes,
            "legal_rate": legal_episodes / max(1, episodes),
            "best_epoch": best_epoch,
            "best_comp_res_hpwl": None if best_epoch is None else best_hpwl,
            "best_partial_global_hpwl": None if best_epoch is None else best_hpwl,
            "best_blank_rectangle": None if best_grid is None else list(best_blank_rectangle),
            "best_blank_rectangle_area": best_blank_area,
            "best_translation_grid": None if best_grid is None else list(best_translation),
            "training_steps": agent.training_step,
            "buffer_device": str(agent.buffer_device),
            "compact_state_buffer": agent.compact_state_buffer,
            "buffer_memory_bytes": agent.buffer_memory_bytes,
            "peak_gpu_memory_bytes": peak_gpu,
            "wall_seconds": time.time() - started,
            "execution_dir": str(execution.resolve()),
        }
        atomic_json(execution / "summary.json", summary)
        atomic_json(execution / "status.json", summary)
        render_training_curves(
            log_path,
            execution / "training-curves.png",
            "{} component {} seed {}".format(benchmark.name, plan.component_id, seed),
        )
        if best_policy_grid is not None:
            raw_physical = {
                name: geometry.physical(*best_policy_grid[name]) for name in plan.order
            }
            save_placement(execution / "best-policy-placement.json", raw_physical)
            render_layout(
                benchmark,
                raw_physical,
                execution / "best-policy-layout.png",
                "{} component {} raw best epoch {}".format(
                    benchmark.name, plan.component_id, best_epoch
                ),
            )
        if best_physical is not None:
            combined = dict(fixed_physical)
            combined.update(best_physical)
            render_layout(
                benchmark,
                combined,
                execution / "best-layout.png",
                "{} component {} best epoch {}".format(benchmark.name, plan.component_id, best_epoch),
            )
        return summary
    except BaseException as error:
        failure = {
            "status": "crashed",
            "error_type": type(error).__name__,
            "error": repr(error),
            "wall_seconds": time.time() - started,
            "execution_dir": str(execution.resolve()),
            **configuration,
        }
        atomic_json(execution / "status.json", failure)
        raise
    finally:
        stream.close()
        if writer is not None:
            writer.close()


def _save_greedy_artifacts(
    benchmark: Benchmark,
    plans: Sequence[ComponentPlan],
    result,
    directory: Path,
    label: str,
):
    directory.mkdir(parents=True, exist_ok=True)
    save_grid_placement(directory / "grid-placement.json", result.grid_placement)
    save_placement(directory / "placement.json", result.placement)
    atomic_json(
        directory / "blank-rectangle-history.json",
        {"history": result.blank_rectangle_history},
    )
    atomic_json(
        directory / "individual-fallback-history.json",
        {"history": result.fallback_history},
    )
    atomic_json(
        directory / "attempt-history.json",
        {"history": result.attempt_history},
    )
    atomic_json(
        directory / "summary.json",
        {
            "status": "complete" if result.legal else "failed",
            "legal": result.legal,
            "final_blank_rectangle": list(result.reserve),
            "final_blank_rectangle_area": int(
                (result.reserve[2] - result.reserve[0]) * (result.reserve[3] - result.reserve[1])
            ),
            "attempts": result.attempts,
            "failure": result.failure,
            "placed_macros": len(result.placement),
            "components": len(plans),
            "greedy_objective": [
                "minimum incremental internal HPWL in a local coordinate system",
                "caller-specified component order",
                "rigid translation maximizing the surviving blank rectangle",
                "seeded random tie-break",
                "optional per-macro fallback: minimum incremental HPWL, then blank rectangle, then seeded random",
            ],
        },
    )
    for plan in plans:
        component_directory = directory / "components" / "component-{:04d}".format(plan.component_id)
        component_directory.mkdir(parents=True, exist_ok=True)
        placement = result.component_placements.get(plan.component_id, {})
        relative_grid = result.component_relative_grid_placements.get(plan.component_id, {})
        relative_placement = result.component_relative_placements.get(plan.component_id, {})
        blank_step = next(
            (item for item in result.blank_rectangle_history if item["component_id"] == plan.component_id),
            None,
        )
        save_placement(component_directory / "placement.json", placement)
        save_grid_placement(component_directory / "relative-grid-placement.json", relative_grid)
        save_placement(component_directory / "relative-placement.json", relative_placement)
        atomic_json(
            component_directory / "metadata.json",
            {
                "component_id": plan.component_id,
                "size": plan.size,
                "area": plan.area,
                "macro_order": list(plan.order),
                "placed_macros": len(placement),
                "relative_internal_hpwl": result.component_relative_hpwl.get(plan.component_id),
                "canvas_translation_grid": list(result.component_translations[plan.component_id])
                if plan.component_id in result.component_translations
                else None,
                "relative_layout_fixed_before_canvas_translation": True,
                "blank_rectangle_step": blank_step,
            },
        )
        if placement:
            render_layout(
                benchmark,
                placement,
                component_directory / "placement.png",
                "{} component {}".format(benchmark.name, plan.component_id),
            )
        if relative_placement:
            render_layout(
                benchmark,
                relative_placement,
                component_directory / "relative-placement.png",
                "{} component {} independent greedy layout".format(
                    benchmark.name, plan.component_id
                ),
            )
    if result.placement:
        render_layout(benchmark, result.placement, directory / "placement.png", label)


def _grid_legality(benchmark: Benchmark, grid_placement, grid: int = 448):
    geometry = GridGeometry(benchmark, grid)
    missing = tuple(name for name in benchmark.selected_macros if name not in grid_placement)
    boundary = []
    occupied = np.zeros((grid, grid), dtype=np.int32)
    overlaps = []
    owner = {}
    for name in benchmark.selected_macros:
        if name not in grid_placement:
            continue
        gx, gy = map(int, grid_placement[name])
        sx, sy = geometry.footprint(name)
        if gx < 0 or gy < 0 or gx + sx > grid or gy + sy > grid:
            boundary.append(name)
            continue
        conflict_cells = np.argwhere(occupied[gx : gx + sx, gy : gy + sy] > 0)
        for dx, dy in conflict_cells[:1]:
            other = owner.get((gx + int(dx), gy + int(dy)))
            if other is not None:
                overlaps.append((other, name))
        occupied[gx : gx + sx, gy : gy + sy] += 1
        for x in range(gx, gx + sx):
            for y in range(gy, gy + sy):
                owner.setdefault((x, y), name)
    legal = not (missing or boundary or overlaps)
    return {
        "legal": legal,
        "boundary_violations": list(boundary),
        "obstacle_violations": [],
        "overlaps": [list(item) for item in overlaps],
        "missing_macros": list(missing),
        "nonfinite_macros": [],
    }


def _final_metrics(benchmark: Benchmark, placement, grid_placement, grid: int = 448):
    legality = _grid_legality(benchmark, grid_placement, grid)
    geometry = GridGeometry(benchmark, grid)
    plans = component_plans(benchmark, 0)
    order = tuple(
        name
        for plan in ordered_large_components(plans) + ordered_small_components(plans)
        for name in plan.order
    )
    placedb = ComponentPlaceDB(benchmark, benchmark.selected_macros, order)
    node_pos = {
        name: (int(value[0]), int(value[1]), *geometry.footprint(name))
        for name, value in grid_placement.items()
        if name in benchmark.selected_macros
    }
    official_hpwl = float(comp_res_hpwl(placedb, node_pos, geometry.ratio))
    metric_benchmark = copy.copy(benchmark)
    metric_benchmark.canvas = Rect(0.0, 0.0, geometry.extent, geometry.extent)
    metric_benchmark.fixed_obstacles = {}
    demand = (
        rudy_map(metric_benchmark, placement, grid=224)
        if legality["legal"]
        else np.zeros((224, 224))
    )
    peak, tail = rudy_peak_tail(demand, tail_fraction=0.05)
    return {
        "legal": legality["legal"],
        "legality": legality,
        "comp_res_hpwl": official_hpwl,
        "macro_hpwl": official_hpwl,
        "all_pin_macro_hpwl": macro_hpwl(metric_benchmark, placement),
        "rudy_grid": 224,
        "rudy_peak": peak,
        "rudy_top5_mean": tail,
    }, demand


def run_linkplace_c(
    benchmark_name: str,
    seed: int,
    cache_root: Path,
    result_root: Path,
    episodes: int = 1000,
    grid: int = 448,
    threshold: int = 20,
    device: str = "cuda",
    component_retries: int = 1,
    max_flow_restarts: int = 0,
):
    benchmark = load_named_benchmark(benchmark_name, cache_root)
    seed_root = Path(result_root) / "main" / benchmark_name / "seed-{}".format(seed)
    final_pointer = seed_root / "result.json"
    if final_pointer.exists() and read_json(final_pointer).get("status") == "complete":
        return read_json(final_pointer)
    seed_root.mkdir(parents=True, exist_ok=True)
    atomic_json(seed_root / "benchmark.json", benchmark_summary(benchmark))
    flow_restart = 0
    while max_flow_restarts <= 0 or flow_restart < max_flow_restarts:
        flow_seed = int(seed) + flow_restart * 10000019
        flow_root = seed_root / "flow-restart-{:03d}".format(flow_restart)
        flow_summary_path = flow_root / "summary.json"
        if flow_summary_path.exists() and read_json(flow_summary_path).get("status") == "failed":
            flow_restart += 1
            continue
        flow_root.mkdir(parents=True, exist_ok=True)
        started = time.time()
        plans = component_plans(benchmark, flow_seed)
        large = ordered_large_components(plans, threshold)
        small = ordered_small_components(plans, threshold)
        statistics = component_statistics(benchmark, flow_seed, threshold)
        atomic_json(flow_root / "components.json", statistics)
        fixed_grid = {}
        fixed_physical = {}
        failed_component = None
        component_records = []
        for sequence_index, plan in enumerate(large):
            success = None
            for retry in range(component_retries):
                training_seed = flow_seed + sequence_index * 10007 + retry * 1000003
                parent = (
                    flow_root
                    / "large-components"
                    / "sequence-{:03d}-component-{:04d}".format(sequence_index, plan.component_id)
                    / "retry-{:02d}".format(retry)
                )
                summary = train_component(
                    benchmark,
                    plan,
                    fixed_grid,
                    fixed_physical,
                    parent,
                    seed=training_seed,
                    episodes=episodes,
                    grid=grid,
                    device=device,
                )
                record = {
                    "sequence_index": sequence_index,
                    "component_id": plan.component_id,
                    "retry": retry,
                    "training_seed": training_seed,
                    "summary": summary,
                }
                component_records.append(record)
                if summary.get("status") == "complete":
                    execution = Path(summary["execution_dir"])
                    component_grid = load_saved_placement(execution / "best-grid-placement.json", "grid_placement")
                    component_physical = load_saved_placement(execution / "best-placement.json", "placement")
                    fixed_grid.update(component_grid)
                    fixed_physical.update(component_physical)
                    success = summary
                    cumulative = flow_root / "cumulative" / "sequence-{:03d}".format(sequence_index)
                    save_grid_placement(cumulative / "grid-placement.json", fixed_grid)
                    save_placement(cumulative / "placement.json", fixed_physical)
                    render_layout(
                        benchmark,
                        fixed_physical,
                        cumulative / "placement.png",
                        "{} after large component {}".format(benchmark_name, sequence_index),
                    )
                    break
            if success is None:
                failed_component = {
                    "sequence_index": sequence_index,
                    "component_id": plan.component_id,
                    "retries": component_retries,
                }
                break
        atomic_json(flow_root / "component-attempts.json", {"attempts": component_records})
        if failed_component is not None:
            failure = {
                "status": "failed",
                "stage": "large_component",
                "failed_component": failed_component,
                "flow_restart": flow_restart,
                "flow_seed": flow_seed,
                "wall_seconds": time.time() - started,
            }
            atomic_json(flow_summary_path, failure)
            flow_restart += 1
            continue
        greedy = greedy_place_components(
            benchmark,
            small,
            seed=flow_seed,
            grid=grid,
            initial_grid_placement=fixed_grid,
            initial_placement=fixed_physical,
            allow_individual_fallback=True,
        )
        _save_greedy_artifacts(
            benchmark,
            small,
            greedy,
            flow_root / "small-greedy",
            "{} small components seed {}".format(benchmark_name, seed),
        )
        if not greedy.legal:
            failure = {
                "status": "failed",
                "stage": "small_greedy",
                "failure": greedy.failure,
                "flow_restart": flow_restart,
                "flow_seed": flow_seed,
                "wall_seconds": time.time() - started,
            }
            atomic_json(flow_summary_path, failure)
            flow_restart += 1
            continue
        fixed_grid = dict(greedy.grid_placement)
        fixed_physical = dict(greedy.placement)
        metrics, demand = _final_metrics(benchmark, fixed_physical, fixed_grid, grid)
        if not metrics["legal"]:
            failure = {
                "status": "failed",
                "stage": "final_legality",
                "metrics": metrics,
                "flow_restart": flow_restart,
                "flow_seed": flow_seed,
                "wall_seconds": time.time() - started,
            }
            atomic_json(flow_summary_path, failure)
            flow_restart += 1
            continue
        save_grid_placement(flow_root / "final" / "grid-placement.json", fixed_grid)
        save_placement(flow_root / "final" / "placement.json", fixed_physical)
        np.save(flow_root / "final" / "rudy-224.npy", demand)
        render_rudy(
            demand,
            flow_root / "final" / "rudy-224.png",
            "{} seed {} RUDY".format(benchmark_name, seed),
        )
        render_layout(
            benchmark,
            fixed_physical,
            flow_root / "final" / "placement.png",
            "{} seed {} final".format(benchmark_name, seed),
        )
        summary = {
            "status": "complete",
            "variant": LINKPLACE_C,
            "benchmark": benchmark_name,
            "seed": seed,
            "flow_seed": flow_seed,
            "flow_restart": flow_restart,
            "grid": grid,
            "threshold": threshold,
            "small_components": len(small),
            "large_components": len(large),
            "metrics": metrics,
            "wall_seconds": time.time() - started,
            "flow_root": str(flow_root.resolve()),
            "final_placement": str((flow_root / "final" / "placement.json").resolve()),
        }
        atomic_json(flow_summary_path, summary)
        atomic_json(final_pointer, summary)
        return summary
    failure = {
        "status": "failed",
        "benchmark": benchmark_name,
        "seed": seed,
        "reason": "flow restart limit exhausted",
        "max_flow_restarts": max_flow_restarts,
    }
    atomic_json(final_pointer, failure)
    return failure


def run_ablation(
    benchmark_name: str,
    variant: str,
    cache_root: Path,
    result_root: Path,
    episodes: int = 1000,
    grid: int = 448,
    seed: int = 999,
    device: str = "cuda",
):
    variant = normalize_variant(variant)
    if benchmark_name not in ABLATION_BENCHMARKS:
        raise ValueError("ablation benchmark not approved: {}".format(benchmark_name))
    if variant == "all-greedy" and benchmark_name not in ISPD_FULL:
        raise ValueError(
            "all-greedy ablation is approved only for ISPD2005: {}".format(
                benchmark_name
            )
        )
    benchmark = load_named_benchmark(benchmark_name, cache_root)
    output = Path(result_root) / "ablation" / variant / benchmark_name / "seed-{}".format(seed)
    result_path = output / "result.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)
    plans = component_plans(benchmark, seed)
    if variant == "all-greedy":
        large = ordered_large_components(plans)
        small = ordered_small_components(plans)
        large_result = greedy_place_components(
            benchmark,
            large,
            seed=seed,
            grid=grid,
        )
        _save_greedy_artifacts(
            benchmark,
            large,
            large_result,
            output / "large-greedy",
            "{} large components greedy".format(benchmark_name),
        )
        if not large_result.legal:
            summary = {
                "status": "failed",
                "variant": variant,
                "benchmark": benchmark_name,
                "seed": seed,
                "grid": grid,
                "stage": "large-greedy",
                "failure": large_result.failure,
            }
        else:
            result = greedy_place_components(
                benchmark,
                small,
                seed=seed,
                grid=grid,
                initial_grid_placement=large_result.grid_placement,
                initial_placement=large_result.placement,
                allow_individual_fallback=True,
            )
            _save_greedy_artifacts(
                benchmark,
                small,
                result,
                output / "small-greedy",
                "{} small components greedy".format(benchmark_name),
            )
            if not result.legal:
                summary = {
                    "status": "failed",
                    "variant": variant,
                    "benchmark": benchmark_name,
                    "seed": seed,
                    "grid": grid,
                    "stage": "small-greedy",
                    "failure": result.failure,
                }
                atomic_json(result_path, summary)
                return summary
            metrics, demand = _final_metrics(
                benchmark, result.placement, result.grid_placement, grid
            )
            save_placement(output / "final" / "placement.json", result.placement)
            save_grid_placement(output / "final" / "grid-placement.json", result.grid_placement)
            (output / "final").mkdir(parents=True, exist_ok=True)
            np.save(output / "final" / "rudy-224.npy", demand)
            render_rudy(
                demand,
                output / "final" / "rudy-224.png",
                "{} all-greedy RUDY".format(benchmark_name),
            )
            render_layout(benchmark, result.placement, output / "final" / "placement.png", "{} all greedy".format(benchmark_name))
            summary = {
                "status": "complete" if metrics["legal"] else "failed",
                "variant": variant,
                "benchmark": benchmark_name,
                "seed": seed,
                "grid": grid,
                "metrics": metrics,
            }
    elif variant == LINKPLACE_M:
        single_policy_plans = tuple(
            sorted(
                plans,
                key=lambda item: (-item.size, -item.area, min(item.macros)),
            )
        )
        order = tuple(name for plan in single_policy_plans for name in plan.order)
        single_policy = ComponentPlan(-1, tuple(benchmark.selected_macros), order, sum(item.area for item in plans))
        component = train_component(
            benchmark,
            single_policy,
            {},
            {},
            output / "training",
            seed=seed,
            episodes=episodes,
            grid=grid,
            device=device,
        )
        if component.get("status") != "complete":
            summary = {
                "status": "failed",
                "variant": variant,
                "benchmark": benchmark_name,
                "seed": seed,
                "grid": grid,
                "component": component,
            }
        else:
            execution = Path(component["execution_dir"])
            placement = load_saved_placement(execution / "best-placement.json", "placement")
            grid_placement = load_saved_placement(execution / "best-grid-placement.json", "grid_placement")
            metrics, demand = _final_metrics(
                benchmark, placement, grid_placement, grid
            )
            save_placement(output / "final" / "placement.json", placement)
            save_grid_placement(output / "final" / "grid-placement.json", grid_placement)
            (output / "final").mkdir(parents=True, exist_ok=True)
            np.save(output / "final" / "rudy-224.npy", demand)
            render_rudy(
                demand,
                output / "final" / "rudy-224.png",
                "{} LinkPlace-M RUDY".format(benchmark_name),
            )
            render_layout(benchmark, placement, output / "final" / "placement.png", "{} LinkPlace-M".format(benchmark_name))
            summary = {
                "status": "complete" if metrics["legal"] else "failed",
                "variant": variant,
                "benchmark": benchmark_name,
                "seed": seed,
                "grid": grid,
                "metrics": metrics,
                "component": component,
            }
    else:
        raise ValueError("unknown ablation variant: {}".format(variant))
    atomic_json(result_path, summary)
    return summary


def run_blank_large_component(
    benchmark_name: str,
    cache_root: Path,
    result_root: Path,
    episodes: int = 1000,
    grid: int = 448,
    seed: int = 999,
    threshold: int = 20,
    device: str = "cuda",
):
    """Train the largest component once with no pre-placed canvas occupancy."""

    benchmark = load_named_benchmark(benchmark_name, cache_root)
    plans = component_plans(benchmark, seed)
    large = ordered_large_components(plans, threshold)
    if not large:
        raise ValueError("benchmark has no component at or above threshold {}".format(threshold))
    plan = large[0]
    output = (
        Path(result_root)
        / "supplementary"
        / "blank-canvas"
        / benchmark_name
        / "largest-component"
        / "seed-{}".format(seed)
    )
    result_path = output / "result.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)

    # Empty means no small components, no preceding large components, and no
    # occupancy obstacles. Fixed terminal coordinates remain available to the
    # LinkPlace wirelength reward but do not occupy canvas area.
    blank_benchmark = copy.copy(benchmark)
    removed_fixed_obstacles = len(blank_benchmark.fixed_obstacles)
    blank_benchmark.fixed_obstacles = {}
    atomic_json(
        output / "experiment.json",
        {
            "benchmark": benchmark_name,
            "seed": seed,
            "episodes": episodes,
            "grid": grid,
            "threshold": threshold,
            "component_id": plan.component_id,
            "component_size": plan.size,
            "component_area": plan.area,
            "macro_order": list(plan.order),
            "initial_fixed_macros": 0,
            "initial_occupancy_cells": 0,
            "removed_fixed_obstacles": removed_fixed_obstacles,
        },
    )
    component = train_component(
        blank_benchmark,
        plan,
        {},
        {},
        output / "training",
        seed=seed,
        episodes=episodes,
        grid=grid,
        device=device,
        reuse_completed=not result_path.exists(),
    )
    if component.get("status") != "complete":
        summary = {
            "status": "failed",
            "variant": "largest-component-blank-canvas",
            "benchmark": benchmark_name,
            "seed": seed,
            "component": component,
        }
        atomic_json(result_path, summary)
        return summary

    execution = Path(component["execution_dir"])
    placement = load_saved_placement(execution / "best-placement.json", "placement")
    grid_placement = load_saved_placement(execution / "best-grid-placement.json", "grid_placement")
    partial_benchmark = copy.copy(blank_benchmark)
    partial_benchmark.selected_macros = tuple(plan.macros)
    metrics, demand = _final_metrics(
        partial_benchmark, placement, grid_placement, grid
    )
    regression_range = [534466.48, 552665.46]
    best_training_hpwl = component.get("best_comp_res_hpwl")
    regression_pass = (
        benchmark_name != "adaptec1"
        or seed != 999
        or episodes != 1000
        or (
            best_training_hpwl is not None
            and float(best_training_hpwl) <= regression_range[1]
        )
    )
    save_placement(output / "final" / "placement.json", placement)
    save_grid_placement(output / "final" / "grid-placement.json", grid_placement)
    (output / "final").mkdir(parents=True, exist_ok=True)
    np.save(output / "final" / "rudy-224.npy", demand)
    render_rudy(
        demand,
        output / "final" / "rudy-224.png",
        "{} blank-canvas RUDY".format(benchmark_name),
    )
    render_layout(
        partial_benchmark,
        placement,
        output / "final" / "placement.png",
        "{} largest component on blank canvas seed {}".format(benchmark_name, seed),
    )
    summary = {
        "status": "complete" if metrics["legal"] and regression_pass else "failed",
        "failure": (
            None
            if metrics["legal"] and regression_pass
            else (
                "a1_448 regression HPWL exceeds historical upper bound"
                if metrics["legal"] and not regression_pass
                else "illegal placement"
            )
        ),
        "variant": "largest-component-blank-canvas",
        "benchmark": benchmark_name,
        "seed": seed,
        "episodes": episodes,
        "grid": grid,
        "component_id": plan.component_id,
        "component_size": plan.size,
        "component_area": plan.area,
        "initial_fixed_macros": 0,
        "initial_occupancy_cells": 0,
        "metrics": metrics,
        "regression_range": regression_range,
        "regression_pass": regression_pass,
        "regression_acceptance_rule": (
            "legal best CompRes HPWL must not exceed the historical upper bound; lower is better"
        ),
        "component": component,
        "final_placement": str((output / "final" / "placement.json").resolve()),
    }
    atomic_json(result_path, summary)
    return summary


def inspect_benchmark(name: str, cache_root: Path, seed: int = 999, threshold: int = 20):
    benchmark = load_named_benchmark(name, cache_root)
    return {
        "benchmark": benchmark_summary(benchmark),
        "components": component_statistics(benchmark, seed, threshold),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="LinkPlace paper experiment runner")
    parser.add_argument("--cache-root", type=Path, default=Path("datasets/cache"))
    parser.add_argument("--result-root", type=Path, default=Path("outputs/formal"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("benchmark", choices=MAIN_BENCHMARKS)
    inspect_parser.add_argument("--seed", type=int, default=999)
    main_parser = subparsers.add_parser("run-main")
    main_parser.add_argument("benchmark", choices=MAIN_BENCHMARKS)
    main_parser.add_argument("--seed", type=int, required=True)
    main_parser.add_argument("--episodes", type=int, default=1000)
    main_parser.add_argument("--device", default="cuda")
    main_parser.add_argument("--grid", type=int, choices=(224, 448), default=448)
    main_parser.add_argument("--max-flow-restarts", type=int, default=0)
    ablation_parser = subparsers.add_parser("run-ablation")
    ablation_parser.add_argument(
        "variant",
        choices=(LINKPLACE_M, LEGACY_LINKPLACE_M, "all-greedy"),
        help="Use linkplace-m for the monolithic policy; monolithic is a legacy alias.",
    )
    ablation_parser.add_argument("benchmark", choices=ABLATION_BENCHMARKS)
    ablation_parser.add_argument("--episodes", type=int, default=1000)
    ablation_parser.add_argument("--seed", type=int, default=999)
    ablation_parser.add_argument("--device", default="cuda")
    ablation_parser.add_argument("--grid", type=int, choices=(224, 448), default=448)
    blank_parser = subparsers.add_parser("run-blank-large")
    blank_parser.add_argument("benchmark", choices=MAIN_BENCHMARKS)
    blank_parser.add_argument("--episodes", type=int, default=1000)
    blank_parser.add_argument("--seed", type=int, default=999)
    blank_parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "inspect":
        result = inspect_benchmark(args.benchmark, args.cache_root, args.seed)
    elif args.command == "run-main":
        result = run_linkplace_c(
            args.benchmark,
            args.seed,
            args.cache_root,
            args.result_root,
            episodes=args.episodes,
            grid=args.grid,
            device=args.device,
            max_flow_restarts=args.max_flow_restarts,
        )
    elif args.command == "run-ablation":
        result = run_ablation(
            args.benchmark,
            args.variant,
            args.cache_root,
            args.result_root,
            episodes=args.episodes,
            grid=args.grid,
            seed=args.seed,
            device=args.device,
        )
    elif args.command == "run-blank-large":
        result = run_blank_large_component(
            args.benchmark,
            args.cache_root,
            args.result_root,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status", "complete") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

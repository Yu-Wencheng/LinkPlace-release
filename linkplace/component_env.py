from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from .data import Benchmark
from .greedy import GridGeometry


class ComponentPlaceDB:
    """Compatibility view consumed by the LinkPlace Actor/PPO code."""

    def __init__(self, benchmark: Benchmark, component: Sequence[str], order: Sequence[str]):
        if set(component) != set(order) or len(component) != len(order):
            raise ValueError("macro order must be a permutation of the component")
        self.benchmark_name = benchmark.name
        self.benchmark_data = benchmark
        self.node_id_to_name = list(order)
        self.node_cnt = len(order)
        self.selected_macros = set(component)
        geometry = GridGeometry(benchmark)
        self.max_width = geometry.extent
        self.max_height = geometry.extent
        self.canvas_origin_x = 0.0
        self.canvas_origin_y = 0.0
        self.node_info = {
            name: {
                "id": index,
                "x": benchmark.nodes[name].width,
                "y": benchmark.nodes[name].height,
            }
            for index, name in enumerate(order)
        }
        # The original Bookshelf PlaceDB ignores all non-macro pins and ports.
        # Ariane is the one approved benchmark whose original protobuf loader
        # retains explicit ports.
        use_ports = benchmark.name == "ariane"
        self.port_info = (
            {
                name: {"x": point[0], "y": point[1]}
                for name, point in benchmark.fixed_terminals.items()
            }
            if use_ports
            else {}
        )
        self.net_info = {}
        component_set = set(component)
        for net in benchmark.evaluated_nets:
            nodes = {}
            for pin in net.pins:
                if pin.owner in component_set and pin.owner not in nodes:
                    nodes[pin.owner] = {"x_offset": pin.offset_x, "y_offset": pin.offset_y}
            if not nodes:
                continue
            ports = {}
            if use_ports:
                for pin in net.pins:
                    if pin.owner in benchmark.fixed_terminals and pin.owner not in ports:
                        ports[pin.owner] = {
                            "x": benchmark.fixed_terminals[pin.owner][0],
                            "y": benchmark.fixed_terminals[pin.owner][1],
                        }
            if (not use_ports and len(nodes) <= 1) or (use_ports and len(nodes) + len(ports) < 2):
                continue
            self.net_info[net.name] = {"nodes": nodes, "ports": ports, "weight": 1.0}
        self.net_cnt = len(self.net_info)
        self.node_to_net_dict: Dict[str, Set[str]] = {name: set() for name in order}
        self.port_to_net_dict: Dict[str, Set[str]] = {name: set() for name in self.port_info}
        for net_name, net in self.net_info.items():
            for name in net["nodes"]:
                self.node_to_net_dict[name].add(net_name)
            for name in net["ports"]:
                self.port_to_net_dict[name].add(net_name)


class InitialCanvasPlaceEnv:
    """LinkPlace environment initialized with already fixed macro occupancy."""

    def __init__(
        self,
        placedb: ComponentPlaceDB,
        initial_grid_placement: Mapping[str, Tuple[int, int]],
        grid: int = 448,
        reward_scale: float = 200.0,
        device: str = "cuda",
    ):
        self.placedb = placedb
        self.benchmark = placedb.benchmark_data
        self.geometry = GridGeometry(self.benchmark, grid)
        self.grid = int(grid)
        self.num_macros = placedb.node_cnt
        self.num_macros_to_place = placedb.node_cnt
        self.node_name_list = placedb.node_id_to_name
        self.scale_x = self.geometry.ratio
        self.scale_y = self.geometry.ratio
        self.ratio = self.geometry.ratio
        self.size_x = [self.geometry.footprint(name)[0] for name in self.node_name_list]
        self.size_y = [self.geometry.footprint(name)[1] for name in self.node_name_list]
        self.reward_scale = float(reward_scale)
        self.wire_mask_scale = self.grid * self.grid
        self.device = torch.device(device)
        self._rows = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)
        self._cols = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)
        self._initial_canvas = torch.zeros((self.grid, self.grid), device=self.device, dtype=torch.float32)
        for name, (gx, gy) in initial_grid_placement.items():
            sx, sy = self.geometry.footprint(name)
            self._paint(self._initial_canvas, gx, gy, sx, sy)
        self.initial_grid_placement = dict(initial_grid_placement)
        self.state = torch.zeros((3, self.grid, self.grid), device=self.device, dtype=torch.float32)
        self.t = 0
        self.node_pos: Dict[str, Tuple[int, int, int, int]] = {}
        self.net_bound_info = {}
        self.failure_reason = None
        self.had_illegal_action = False
        self._legal_action_count = 0

    @staticmethod
    def _paint(canvas, x: int, y: int, sx: int, sy: int):
        x1 = min(canvas.shape[0], x + sx)
        y1 = min(canvas.shape[1], y + sy)
        if x >= x1 or y >= y1:
            return
        canvas[x:x1, y:y1] = 1.0
        canvas[x:x1, y] = 0.5
        canvas[x:x1, y1 - 1] = 0.5
        canvas[x, y:y1] = 0.5
        canvas[x1 - 1, y:y1] = 0.5

    def seed(self, seed):
        return [int(seed)]

    def reset(self):
        self.t = 0
        self.node_pos.clear()
        self.net_bound_info.clear()
        self.failure_reason = None
        self.had_illegal_action = False
        canvas = self._initial_canvas.clone()
        wire_mask = torch.zeros_like(canvas)
        position_mask = self._calc_position_mask(canvas, self.size_x[0], self.size_y[0])
        self.state = torch.stack((canvas, wire_mask, position_mask), dim=0)
        return self.state

    @property
    def legal_action_count(self):
        return self._legal_action_count

    @torch.no_grad()
    def _calc_position_mask(self, canvas, next_x: int, next_y: int):
        grid = self.grid
        if next_x > grid or next_y > grid:
            self._legal_action_count = 0
            return torch.ones_like(canvas)
        if bool(canvas.sum() == 0):
            # Preserve the running a1_448 implementation exactly, including
            # its empty-canvas upper-bound convention (no +1).
            mask = torch.ones_like(canvas)
            mask[: grid - next_x, : grid - next_y] = 0.0
            self._legal_action_count = int((mask < 1.0).sum().item())
            return mask
        occupied = (canvas > 0).float().unsqueeze(0).unsqueeze(0)
        blocked = F.max_pool2d(occupied, kernel_size=(next_x, next_y), stride=1, padding=0)
        mask = torch.zeros_like(canvas)
        mask[: grid - next_x + 1, : grid - next_y + 1] = blocked[0, 0]
        if grid - next_x + 1 < grid:
            mask[grid - next_x + 1 :, :] = 1.0
        if grid - next_y + 1 < grid:
            mask[:, grid - next_y + 1 :] = 1.0
        self._legal_action_count = int((mask < 1.0).sum().item())
        return mask

    @torch.no_grad()
    def _calc_wire_mask(self, node_name: str):
        nets = [name for name in self.placedb.node_to_net_dict[node_name] if name in self.net_bound_info]
        if not nets:
            return torch.zeros((self.grid, self.grid), device=self.device, dtype=torch.float32)
        node = self.placedb.node_info[node_name]
        off_x = torch.tensor(
            [self.placedb.net_info[name]["nodes"][node_name]["x_offset"] for name in nets],
            device=self.device,
            dtype=torch.float32,
        )
        off_y = torch.tensor(
            [self.placedb.net_info[name]["nodes"][node_name]["y_offset"] for name in nets],
            device=self.device,
            dtype=torch.float32,
        )
        dx = torch.round((node["x"] / 2.0 + off_x) / self.scale_x)
        dy = torch.round((node["y"] / 2.0 + off_y) / self.scale_y)
        min_x = torch.tensor([self.net_bound_info[name]["min_x"] for name in nets], device=self.device, dtype=torch.float32)
        max_x = torch.tensor([self.net_bound_info[name]["max_x"] for name in nets], device=self.device, dtype=torch.float32)
        min_y = torch.tensor([self.net_bound_info[name]["min_y"] for name in nets], device=self.device, dtype=torch.float32)
        max_y = torch.tensor([self.net_bound_info[name]["max_y"] for name in nets], device=self.device, dtype=torch.float32)
        weights = torch.tensor(
            [self.placedb.net_info[name].get("weight", 1.0) for name in nets],
            device=self.device,
            dtype=torch.float32,
        )
        start_x = (min_x - dx).clamp(0.0, self.grid - 1.0)
        end_x = (max_x - dx).clamp(0.0, self.grid - 1.0)
        start_y = (min_y - dy).clamp(0.0, self.grid - 1.0)
        end_y = (max_y - dy).clamp(0.0, self.grid - 1.0)
        distance_x = torch.relu(start_x.unsqueeze(0) - self._rows) + torch.relu(self._rows - end_x.unsqueeze(0))
        distance_y = torch.relu(start_y.unsqueeze(0) - self._cols) + torch.relu(self._cols - end_y.unsqueeze(0))
        row_sum = (distance_x * weights.unsqueeze(0)).sum(dim=1)
        column_sum = (distance_y * weights.unsqueeze(0)).sum(dim=1)
        return (row_sum.unsqueeze(1) + column_sum.unsqueeze(0)) / self.wire_mask_scale

    @torch.no_grad()
    def _update_net_bounds(self, node_name: str, x: int, y: int):
        node = self.placedb.node_info[node_name]
        base_x = x * self.scale_x
        base_y = y * self.scale_y
        for net_name in self.placedb.node_to_net_dict[node_name]:
            pin = self.placedb.net_info[net_name]["nodes"][node_name]
            pin_x = int(round((base_x + node["x"] / 2.0 + pin["x_offset"]) / self.scale_x))
            pin_y = int(round((base_y + node["y"] / 2.0 + pin["y_offset"]) / self.scale_y))
            if net_name not in self.net_bound_info:
                self.net_bound_info[net_name] = {"min_x": pin_x, "max_x": pin_x, "min_y": pin_y, "max_y": pin_y}
            else:
                item = self.net_bound_info[net_name]
                item["min_x"] = min(item["min_x"], pin_x)
                item["max_x"] = max(item["max_x"], pin_x)
                item["min_y"] = min(item["min_y"], pin_y)
                item["max_y"] = max(item["max_y"], pin_y)

    @torch.no_grad()
    def step(self, action):
        if self.t >= self.num_macros_to_place:
            raise RuntimeError("step called after completion")
        value = int(action.item() if hasattr(action, "item") else action)
        if not 0 <= value < self.grid * self.grid:
            raise ValueError("action out of range")
        x = value // self.grid
        y = value % self.grid
        canvas, wire_mask, position_mask = self.state
        illegal_action = bool(position_mask[x, y] >= 1.0)
        sx = self.size_x[self.t]
        sy = self.size_y[self.t]
        reward = -wire_mask[x, y] / self.reward_scale * self.wire_mask_scale
        if illegal_action:
            reward = reward - torch.as_tensor(20000.0 / self.reward_scale, device=self.device)
            self.failure_reason = "illegal_action"
            self.had_illegal_action = True
        self._paint(canvas, x, y, sx, sy)
        node_name = self.node_name_list[self.t]
        self.node_pos[node_name] = (x, y, sx, sy)
        self._update_net_bounds(node_name, x, y)
        self.t += 1
        done = self.t >= self.num_macros_to_place
        if done:
            next_wire = torch.zeros_like(canvas)
            next_mask = torch.ones_like(canvas)
            self._legal_action_count = 0
        else:
            next_wire = self._calc_wire_mask(self.node_name_list[self.t])
            next_mask = self._calc_position_mask(canvas, self.size_x[self.t], self.size_y[self.t])
        self.state = torch.stack((canvas, next_wire, next_mask), dim=0)
        return self.state, float(reward.item()), bool(done), {
            "placed_node_idx": self.t - 1,
            "action_idx": value,
            "xy": (x, y),
            "node_name": node_name,
            "failure": self.failure_reason,
            "illegal_action": illegal_action,
        }

    def physical_placement(self):
        return {
            name: self.geometry.physical(value[0], value[1])
            for name, value in self.node_pos.items()
        }

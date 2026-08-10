import gym
import math
import logging
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, patches
from ..evaluation.comp_res import comp_res


# NOTE
# ----
# 这份 Environment 保留了你原来的三通道 state: [canvas, wire_mask, position_mask]
# 但在实现上做了彻底 GPU 向量化：
# - mask: 用 max_pool2d 实现“矩形膨胀”，O(1) 次算子完成挡位计算
# - wire_mask: 对所有相关网一次性广播/求和（行/列越界距离)
# - reward: 直接使用 wire_mask[x,y] 的增量定义，保持原语义
# - canvas: 原地切片写入（单矩形），若后续需要批量渲染可改差分阵
# 同时做了风格化清理：明确的 helper，缓存与设备管理，避免 .item() 参与控制流


class PlaceEnvGpu(gym.Env):
    def __init__(
            self,
            placedb,
            num_macros_to_place: int,
            grid: int,
            reward_scale: float = 200.0,
            device: str = "cuda",
    ) -> None:
        """
        Args
        ----
        placedb : 兼容你现有 PlaceDB 的对象（含 node/net 信息、映射字典等）
        num_macros_to_place : 训练/评估阶段要放置的宏模块数量上限
        grid : 画布离散分辨率 G（G×G）
        reward_scale : 奖励缩放
        device : 'cuda' / 'cpu'
        """
        super().__init__()
        self.placedb = placedb
        self.num_macros: int = placedb.node_cnt
        self.num_nets: int = placedb.net_cnt
        self.node_name_list: List[str] = placedb.node_id_to_name

        # 几何尺度
        self.grid: int = grid
        self.max_height: float = placedb.max_height
        self.max_width: float = placedb.max_width
        self.ratio: float = self.max_height / self.grid
        print(f"[Env] : self.max_height       {self.max_height}")
        print(f"[Env] : self.ratio       {self.ratio}")

        # 预计算每个宏在网格下的离散尺寸
        self.size_x: List[int] = [max(1, math.ceil(placedb.node_info[name]['x'] / self.ratio))
                                  for name in self.node_name_list]
        self.size_y: List[int] = [max(1, math.ceil(placedb.node_info[name]['y'] / self.ratio))
                                  for name in self.node_name_list]

        # 标量参数
        self.num_macros_to_place = num_macros_to_place
        self.wire_mask_scale = self.grid * self.grid  # 理论wire mask的最大值，几乎不可能达到
        self.reward_scale = float(reward_scale)
        self.chain_broken = False

        # 设备/缓存
        self.device = torch.device(device)
        self._rows = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)  # [G,1]
        self._cols = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)  # [G,1]

        # 运行态
        self.cum_reward: float = 0.0
        self.t: int = 0
        self.node_pos: Dict[str, Tuple[int, int, int, int]] = {}
        self.net_bound_info: Dict[str, Dict[str, int]] = {}
        self.state: torch.Tensor = torch.zeros((3, self.grid, self.grid), device=self.device,
                                               dtype=torch.float32)  # [3, G, G]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self.t = 0
        self.node_pos.clear()
        self.net_bound_info.clear()
        self.cum_reward = 0.0

        canvas = torch.zeros((self.grid, self.grid), device=self.device, dtype=torch.float32)
        wire_mask = torch.zeros_like(canvas)

        next_x = self.size_x[0]
        next_y = self.size_y[0]
        position_mask = self._calc_position_mask(canvas, next_x, next_y)

        self.state = torch.stack([canvas, wire_mask, position_mask], dim=0)
        return self.state

    @torch.no_grad()
    def step(self, action: int):
        """Execute one placement step.
        Returns: next_state, reward, done, info
        """
        assert 0 <= action < self.grid * self.grid, "action out of range"

        canvas, wire_mask, position_mask = self.state[0], self.state[1], self.state[2]

        # 解析动作 -> 坐标
        x = int(action // self.grid)
        y = int(action % self.grid)
        sx = self.size_x[self.t]
        sy = self.size_y[self.t]

        # --------- 执行动作：正常奖励 ----------
        # 奖励 = -wire_mask[x,y] / scale
        reward = - wire_mask[x, y] / self.reward_scale * self.wire_mask_scale

        # --------- 非法动作：给惩罚 ----------
        if position_mask[x, y] >= 1.0:
            penalty = torch.as_tensor(20000.0 / self.reward_scale,
                                      device=self.device, dtype=torch.float32)
            reward = reward - penalty

        # --------- 断链动作：给惩罚 ----------
        # if self.chain_broken:
        #     penalty = torch.as_tensor(1000.0 / self.reward_scale,
        #                               device=self.device, dtype=torch.float32)
        #     reward = reward - penalty
        #     self.chain_broken = False

        # if self.chain_broken and self.t < self.placedb.node_cnt - len(self.placedb.single_node):
        #     # 可调超参
        #     r = getattr(self, "chain_radius", 5)  # 半径
        #     scale = getattr(self, "chain_penalty_scale", 1.0)  # 强度
        #
        #     ring_mean = self._ring_mean(canvas, x, y, sx, sy, r)  # 越界按1处理
        #     # 解释：ring_mean ∈ [0,1+]，越靠近已有占用/边界则均值越高 -> 惩罚越大
        #     penalty = (ring_mean * scale) / self.reward_scale
        #     # 你也可以换成奖励（例如鼓励靠近已有模块：把惩罚改为 -(1 - ring_mean)*scale/self.reward_scale）
        #     reward = reward - penalty
        #
        #     self.chain_broken = False

        # 执行动作,写入 canvas（内部=1.0，边=0.5）
        canvas[x:x + sx, y:y + sy] = 1.0
        canvas[x:x + sx, y] = 0.5
        if y + sy - 1 < self.grid:
            canvas[x:x + sx, y + sy - 1] = 0.5
        canvas[x, y:y + sy] = 0.5
        if x + sx - 1 < self.grid:
            canvas[x + sx - 1, y:y + sy] = 0.5

        # 记录位置
        node_name = self.node_name_list[self.t]
        self.node_pos[node_name] = (x, y, sx, sy)

        # 更新与该宏相关的网络 bbox（离散网格坐标）
        self._update_net_bounds(node_name, x, y)

        # 推进步数
        self.t += 1
        done = self._is_done()

        # 计算下一状态
        if not done:
            next_x = self.size_x[self.t]
            next_y = self.size_y[self.t]
            position_mask = self._calc_position_mask(canvas, next_x, next_y)
            wire_mask = self._calc_wire_mask_for_node(self.node_name_list[self.t])
        else:
            position_mask = torch.ones_like(canvas)
            wire_mask = torch.zeros_like(canvas)

            # 最终指标由外部计算
            # hpwl, cost = comp_res(self.placedb, self.node_pos, self.ratio)

        self.state = torch.stack([canvas, wire_mask, position_mask], dim=0)

        info = {
            'placed_node_idx': self.t - 1,
            'action_idx': int(action),
            'xy': (int(x), int(y)),
            'node_name': self.node_name_list[self.t - 1],
        }
        return self.state, float(reward.item()), bool(done), info

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _calc_position_mask(self, canvas: torch.Tensor, next_x: int, next_y: int) -> torch.Tensor:
        """
        由已占用 canvas 计算“以左上角坐标表示”的不可放置区域：1=不可放置, 0=可放置。
        使用 max_pool2d 实现矩形膨胀，然后补边界越界。
        """
        G = self.grid

        # 检查canvas是否为空
        # 检查canvas是否为空
        if canvas.sum() == 0:  # 如果canvas全为0，则直接返回一个全0的mask，表示全部位置可放置
            # 初始化一个可以放置的区域，但需要确保不会超出边界
            mask = torch.ones_like(canvas)  # 假设全是不可放置区域
            mask[:G - next_x, :G - next_y] = 0  # 设置放置区域大小，确保不会越界
            return mask

        occ = (canvas > 0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, G, G]
        blocked = F.max_pool2d(occ, kernel_size=(next_x, next_y), stride=1, padding=0)  # [1, 1, G-next_x+1, G-next_y+1]
        mask = torch.zeros_like(canvas)
        mask[:G - next_x + 1, :G - next_y + 1] = blocked[0, 0]
        if G - next_x + 1 < G:
            mask[G - next_x + 1:, :] = 1.0
        if G - next_y + 1 < G:
            mask[:, G - next_y + 1:] = 1.0
        return mask

    @torch.no_grad()
    def _calc_wire_mask_for_node(self, node_name: str) -> torch.Tensor:
        """向量化 wire-mask：对 node_name 关联的所有已有 bbox 进行行/列越界距离求和"""
        G = self.grid
        rows = self._rows  # [G, 1]
        cols = self._cols  # [G, 1]
        dev = self.device

        # 关联的 nets；仅保留已有 bbox 的
        nets = [n for n in self.placedb.node_to_net_dict[node_name] if n in self.net_bound_info]

        # 如果没有关联的网络，则返回一个全零的 mask
        if not nets:
            self.chain_broken = True
            return torch.zeros((G, G), device=dev, dtype=torch.float32)

        # per-net 参数
        node_x_half = self.placedb.node_info[node_name]['x'] / 2.0
        node_y_half = self.placedb.node_info[node_name]['y'] / 2.0
        ratio = self.ratio

        off_x = torch.tensor([self.placedb.net_info[n]["nodes"][node_name]["x_offset"] for n in nets],
                             device=dev, dtype=torch.float32)
        off_y = torch.tensor([self.placedb.net_info[n]["nodes"][node_name]["y_offset"] for n in nets],
                             device=dev, dtype=torch.float32)
        dx = torch.round((torch.as_tensor(node_x_half, device=dev) + off_x) / ratio)  # [K]
        dy = torch.round((torch.as_tensor(node_y_half, device=dev) + off_y) / ratio)  # [K]

        min_x = torch.tensor([self.net_bound_info[n]['min_x'] for n in nets], device=dev, dtype=torch.float32)
        max_x = torch.tensor([self.net_bound_info[n]['max_x'] for n in nets], device=dev, dtype=torch.float32)
        min_y = torch.tensor([self.net_bound_info[n]['min_y'] for n in nets], device=dev, dtype=torch.float32)
        max_y = torch.tensor([self.net_bound_info[n]['max_y'] for n in nets], device=dev, dtype=torch.float32)

        # TODO: 权重可以考虑重新设计
        w = torch.tensor([self.placedb.net_info[n].get('weight', 1.0) for n in nets], device=dev, dtype=torch.float32)

        # 平移并 clamp 到 [0, G-1]
        sx = (min_x - dx).clamp_(0.0, G - 1.0)
        ex = (max_x - dx).clamp_(0.0, G - 1.0)
        sy = (min_y - dy).clamp_(0.0, G - 1.0)
        ey = (max_y - dy).clamp_(0.0, G - 1.0)

        # 行/列越界距离；[G, K]
        dist_x = torch.relu(sx.unsqueeze(0) - rows) + torch.relu(rows - ex.unsqueeze(0))
        dist_y = torch.relu(sy.unsqueeze(0) - cols) + torch.relu(cols - ey.unsqueeze(0))

        # 加权求和到 [G]，再外和得到 [G, G]
        row_sum = (dist_x * w.unsqueeze(0)).sum(dim=1)  # [G]
        col_sum = (dist_y * w.unsqueeze(0)).sum(dim=1)  # [G]
        net_img = row_sum.unsqueeze(1) + col_sum.unsqueeze(0)  # [G, G]

        return net_img / self.wire_mask_scale

    @torch.no_grad()
    def _update_net_bounds(self, node_name: str, x: int, y: int) -> None:
        """在网格坐标下，批量更新与 node_name 相连各网的 bbox。"""
        # 节点 pin 离散化坐标（以宏左上为参考）
        node_x = self.placedb.node_info[node_name]['x']
        node_y = self.placedb.node_info[node_name]['y']
        ratio = self.ratio

        # 该宏左上角在物理尺寸下的坐标
        base_x = x * ratio
        base_y = y * ratio

        for net_name in self.placedb.node_to_net_dict[node_name]:
            # pin 的物理坐标 → 网格坐标（离散）
            px = base_x + node_x / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]
            py = base_y + node_y / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
            pin_x = int(round(px / ratio))
            pin_y = int(round(py / ratio))

            if net_name in self.net_bound_info:
                b = self.net_bound_info[net_name]
                # 更新 bbox
                if pin_x < b['min_x']: b['min_x'] = pin_x
                if pin_x > b['max_x']: b['max_x'] = pin_x
                if pin_y < b['min_y']: b['min_y'] = pin_y
                if pin_y > b['max_y']: b['max_y'] = pin_y
            else:
                self.net_bound_info[net_name] = {
                    'min_x': pin_x, 'max_x': pin_x,
                    'min_y': pin_y, 'max_y': pin_y,
                }

    def _ring_mean(self, canvas: torch.Tensor, x: int, y: int, sx: int, sy: int, r: int = 10) -> torch.Tensor:
        """
        计算以 (x,y,sx,sy) 矩形为内核、宽度=r 的外环（不含内核）的 canvas 均值。
        约定：越界处视为1（通过常数填充实现）。
        返回：标量 tensor，dtype/设备与 canvas 一致。
        """
        assert canvas.dim() == 2, f"canvas must be [G, G], got {canvas.shape}"
        dev = canvas.device
        dtype = canvas.dtype

        # 用1进行常数填充：pad顺序为 (left, right, top, bottom)
        # 这样索引就不会越界，且越界区域的取值等于1（题设要求）
        pad_canvas = F.pad(canvas, (r, r, r, r), mode='constant', value=1.0)

        # 在pad坐标系下的矩形起点
        xp, yp = x + r, y + r

        # 定义四条“环带”条带（上/下/左/右），宽度均为 r
        top = pad_canvas[xp - r: xp, yp - r: yp + sy + r]
        bottom = pad_canvas[xp + sx: xp + sx + r, yp - r: yp + sy + r]
        left = pad_canvas[xp: xp + sx, yp - r: yp]
        right = pad_canvas[xp: xp + sx, yp + sy: yp + sy + r]

        ring_sum = top.sum() + bottom.sum() + left.sum() + right.sum()

        # 环带像素总数（避免双计数，四块是互不重叠的）
        ring_area = (sx + 2 * r) * (sy + 2 * r) - (sx * sy)
        ring_mean = ring_sum / torch.as_tensor(float(ring_area), device=dev, dtype=dtype)
        return ring_mean

    def _is_done(self) -> bool:
        return (self.t >= self.num_macros) or (self.t >= self.num_macros_to_place)

    # ---------------------------------------------------------------------
    # Greedy rollout (no-RUDY version, shadow_action only)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def greedy_finish(
            self,
            first_action: Optional[int] = None,
            max_steps: Optional[int] = None,
            return_trajectory: bool = True,
    ):
        """Greedy-complete the current episode from the *current* partial state.


        贪婪选择合法位置中 wire_mask 的全局最小；

        Args:
        first_action: 可选。影子落子的离散动作值，默认画布中心。
        max_steps: 最多前进步数（None 表示直到 done）。
        return_trajectory: 是否返回逐步轨迹（宏名, 动作idx, step奖励）。
        base_reward: 历史累计奖励（包含已布局的模块），用于输出 final_reward 叠加。

        Returns:
        result: dict，包含：
        - final_canvas: [G,G] tensor（贪婪终局）
        - final_node_pos: dict(name -> (x,y,sx,sy))
        - final_t: 结束时步数
        - final_reward: base_reward + 贪婪补完过程中 step 奖励之和
        - trajectory: List[(name, action, reward)]（当 return_trajectory=True）
        """

    def save_fig(self, file_path):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            ax1.add_patch(
                patches.Rectangle(
                    (x / self.grid, y / self.grid),  # (x,y)
                    size_x / self.grid,  # width
                    size_y / self.grid, linewidth=1, edgecolor='k',
                )
            )
        fig1.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close()

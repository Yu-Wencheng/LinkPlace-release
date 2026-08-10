"""Memory-efficient execution of the LinkPlace PPO update equations."""

from __future__ import annotations

from collections import namedtuple
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from .networks import Actor, RDC3


Transition = namedtuple("Transition", ["state", "action", "reward", "a_log_prob", "next_state"])


class PPO:
    """LinkPlace PPO with a preallocated transition buffer.

    The source implementation retained both state and unused next_state and then
    stacked them again at update time.  At 448 this can exceed a 24 GB GPU.  This
    class stores each float32 state once and streams only a minibatch when the
    buffer must live on CPU.  The Actor, critic, return target, clipping, losses,
    optimizers, update count, and batch sampling are otherwise unchanged.
    """

    def __init__(self, env, args):
        self.env = env
        self.device = torch.device(getattr(args, "device", "cuda"))
        self.actor_net = Actor(grid=args.grid).float().to(self.device)
        self.critic_net = RDC3(grid=args.grid).float().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.A_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), args.C_lr)
        self.training_step = 0
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epoch = 10
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.args = args
        self.placed_num_macro = args.pnm
        self.buffer_capacity = 5 * args.pnm
        self.buffer_count = 0
        self.counter = 0
        self.compact_state_buffer = bool(getattr(args, "compact_state_buffer", False))
        self._states = None
        self._canvas_codes = None
        self._wire_masks = None
        self._position_masks = None
        self._actions = None
        self._rewards = None
        self._old_log_probs = None
        self.buffer_device = self._select_buffer_device(getattr(args, "buffer_device", "auto"))

    def _estimated_state_bytes(self):
        elements = self.buffer_capacity * self.args.grid * self.args.grid
        if self.compact_state_buffer:
            # canvas: uint8 code for {0.0, 0.5, 1.0}; wire: float32;
            # position mask: bool.  Decoding recreates the original float32
            # network input exactly while halving persistent buffer storage.
            return elements * (1 + 4 + 1)
        return elements * 3 * 4

    def _select_buffer_device(self, requested):
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError("buffer_device must be auto, cuda, or cpu")
        if requested != "auto":
            return self.device if requested == "cuda" else torch.device("cpu")
        if self.device.type != "cuda":
            return torch.device("cpu")
        state_bytes = self._estimated_state_bytes()
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        return self.device if state_bytes < int(free_bytes * 0.50) else torch.device("cpu")

    def _allocate(self, state):
        if self.compact_state_buffer:
            spatial_shape = (self.buffer_capacity,) + tuple(state.shape[-2:])
            self._canvas_codes = torch.empty(
                spatial_shape, dtype=torch.uint8, device=self.buffer_device
            )
            self._wire_masks = torch.empty(
                spatial_shape, dtype=torch.float32, device=self.buffer_device
            )
            self._position_masks = torch.empty(
                spatial_shape, dtype=torch.bool, device=self.buffer_device
            )
        else:
            shape = (self.buffer_capacity,) + tuple(state.shape)
            self._states = torch.empty(shape, dtype=torch.float32, device=self.buffer_device)
        self._actions = torch.empty(self.buffer_capacity, dtype=torch.long, device=self.buffer_device)
        self._rewards = torch.empty(self.buffer_capacity, dtype=torch.float32, device=self.buffer_device)
        self._old_log_probs = torch.empty(self.buffer_capacity, dtype=torch.float32, device=self.buffer_device)

    def _store_state(self, index, state):
        state = state.detach()
        if self.compact_state_buffer:
            self._canvas_codes[index].copy_(
                state[0].mul(2.0).to(self.buffer_device, dtype=torch.uint8)
            )
            self._wire_masks[index].copy_(
                state[1].to(self.buffer_device, dtype=torch.float32)
            )
            self._position_masks[index].copy_(
                state[2].ge(1.0).to(self.buffer_device, dtype=torch.bool)
            )
        else:
            self._states[index].copy_(
                state.to(self.buffer_device, dtype=torch.float32)
            )

    def _state_batch(self, index):
        if not self.compact_state_buffer:
            return self._states.index_select(0, index).to(
                self.device, non_blocking=True
            )
        canvas = self._canvas_codes.index_select(0, index).to(
            self.device, dtype=torch.float32, non_blocking=True
        ).mul_(0.5)
        wire = self._wire_masks.index_select(0, index).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        position = self._position_masks.index_select(0, index).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        return torch.stack((canvas, wire, position), dim=1)

    @property
    def buffer_memory_bytes(self):
        if self._actions is None:
            return self._estimated_state_bytes() + self.buffer_capacity * (8 + 4 + 4)
        state_tensors = (
            (self._canvas_codes, self._wire_masks, self._position_masks)
            if self.compact_state_buffer
            else (self._states,)
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in state_tensors + (
                self._actions,
                self._rewards,
                self._old_log_probs,
            )
        )

    def select_action(self, state, Eval=False):
        state = state.detach().to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probabilities = self.actor_net(state)
        distribution = Categorical(probabilities)
        action = torch.argmax(probabilities, dim=-1) if Eval else distribution.sample()
        return action, distribution.log_prob(action)

    def store_transition(self, transition):
        if self._actions is None:
            self._allocate(transition.state)
        if self.buffer_count >= self.buffer_capacity:
            raise RuntimeError("LinkPlace transition buffer overflow")
        index = self.buffer_count
        self._store_state(index, transition.state)
        self._actions[index] = int(transition.action.item() if hasattr(transition.action, "item") else transition.action)
        self._rewards[index] = float(transition.reward.item() if hasattr(transition.reward, "item") else transition.reward)
        self._old_log_probs[index] = float(
            transition.a_log_prob.item() if hasattr(transition.a_log_prob, "item") else transition.a_log_prob
        )
        self.buffer_count += 1
        self.counter += 1
        return self.buffer_count == self.buffer_capacity

    def update(self, writer=None):
        if self.buffer_count != self.buffer_capacity:
            raise RuntimeError("LinkPlace PPO update requires a full five-episode buffer")
        rewards = self._rewards[: self.buffer_count]
        if self.env.t >= self.placed_num_macro - 1:
            # This is the branch taken by the reference PPO implementation:
            # env.t is at the completed-episode boundary, so target is reset for
            # every item and each target equals its immediate reward.  Keep that
            # behavior while avoiding thousands of scalar .item() calls.
            target_values = rewards.to(
                self.device, dtype=torch.float32, non_blocking=True
            ).view(-1, 1)
        else:
            target_list = []
            target = 0.0
            for reward in reversed(rewards):
                target = float(reward.item()) + self.gamma * target
                target_list.append(target)
            target_list.reverse()
            target_values = torch.tensor(
                target_list, dtype=torch.float32, device=self.device
            ).view(-1, 1)

        loss_values = None
        loss_steps = None
        loss_wall_times = None
        loss_offset = 0
        if writer is not None:
            batches_per_epoch = self.buffer_count // self.batch_size
            loss_values = torch.empty(
                (self.ppo_epoch * batches_per_epoch, 3),
                dtype=torch.float32,
                device=self.device,
            )
            loss_steps = []
            loss_wall_times = []

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(
                SubsetRandomSampler(range(self.buffer_count)), self.batch_size, True
            )
            for indices in sampler:
                self.training_step += 1
                index = torch.as_tensor(indices, dtype=torch.long, device=self.buffer_device)
                states = self._state_batch(index)
                actions = self._actions.index_select(0, index).to(self.device, non_blocking=True)
                old_log_probs = self._old_log_probs.index_select(0, index).to(self.device, non_blocking=True)
                target_index = index.to(self.device, non_blocking=True)
                target_batch = target_values.index_select(0, target_index)

                probabilities = self.actor_net(states)
                distribution = Categorical(probabilities)
                action_log_prob = distribution.log_prob(actions)
                ratio = torch.exp(action_log_prob - old_log_probs)
                value_total, value_immediate, value_future = self.critic_net(states)
                advantage = (target_batch - value_total).detach()
                surrogate_1 = ratio * advantage.squeeze()
                surrogate_2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * advantage.squeeze()
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss_total = F.smooth_l1_loss(value_total, target_batch)
                value_loss_balance = F.mse_loss(value_total, value_immediate + value_future)
                value_loss = value_loss_total + 0.1 * value_loss_balance
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                if loss_values is not None:
                    loss_values[loss_offset, 0] = actor_loss.detach()
                    loss_values[loss_offset, 1] = value_loss_total.detach()
                    loss_values[loss_offset, 2] = value_loss_balance.detach()
                    loss_steps.append(self.training_step)
                    loss_wall_times.append(time.time())
                    loss_offset += 1

        if loss_values is not None:
            # One device synchronization per full PPO update instead of three
            # per minibatch.  TensorBoard retains every original scalar/step.
            history = loss_values[:loss_offset].cpu().tolist()
            for step, wall_time, values in zip(loss_steps, loss_wall_times, history):
                writer.add_scalar("ppo/action_loss", values[0], step, walltime=wall_time)
                writer.add_scalar("ppo/value_loss_total", values[1], step, walltime=wall_time)
                writer.add_scalar("ppo/value_loss_balance", values[2], step, walltime=wall_time)
        self.buffer_count = 0

    def save_param(self, path):
        torch.save(
            {
                "actor_net_dict": self.actor_net.state_dict(),
                "critic_net_dict": self.critic_net.state_dict(),
                "actor_optimizer_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_dict": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "counter": self.counter,
            },
            path,
        )

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint["actor_net_dict"])
        self.critic_net.load_state_dict(checkpoint["critic_net_dict"])
        if "actor_optimizer_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_dict"])
        if "critic_optimizer_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_dict"])
        self.training_step = int(checkpoint.get("training_step", 0))
        self.counter = int(checkpoint.get("counter", 0))

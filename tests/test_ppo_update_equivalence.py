from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from linkplace.models.networks import PPO as LegacyPPO
from linkplace.models.networks import Transition as LegacyTransition
from linkplace.models.ppo import PPO as PaperPPO


class _FinishedEpisodeEnv:
    # This is the value seen by the original runner when PPO.update is called.
    t = 1


class _FixedBatchSampler:
    """Use one fixed valid minibatch order to compare update equations, not RNG plumbing."""

    def __init__(self, sampler, batch_size, drop_last):
        del drop_last
        self.indices = list(sampler.indices)
        self.batch_size = batch_size

    def __iter__(self):
        for offset in range(0, len(self.indices), self.batch_size):
            batch = self.indices[offset : offset + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch


class _CaptureWriter:
    def __init__(self):
        self.rows = []

    def add_scalar(self, tag, value, step, walltime=None):
        self.rows.append((tag, float(value), int(step), walltime))


def _maximum_parameter_difference(first, second):
    maximum = 0.0
    for first_parameter, second_parameter in zip(
        list(first.actor_net.parameters()) + list(first.critic_net.parameters()),
        list(second.actor_net.parameters()) + list(second.critic_net.parameters()),
    ):
        maximum = max(
            maximum,
            float((first_parameter - second_parameter).abs().max().item()),
        )
    return maximum


def _maximum_optimizer_difference(first_optimizer, second_optimizer):
    maximum = 0.0
    first_state = first_optimizer.state_dict()
    second_state = second_optimizer.state_dict()
    assert first_state["param_groups"] == second_state["param_groups"]
    assert first_state["state"].keys() == second_state["state"].keys()
    for key in first_state["state"]:
        for name, first_value in first_state["state"][key].items():
            second_value = second_state["state"][key][name]
            if torch.is_tensor(first_value):
                maximum = max(
                    maximum,
                    float((first_value - second_value).abs().max().item()),
                )
            else:
                assert first_value == second_value
    return maximum


class CompactBufferEquivalenceTests(unittest.TestCase):
    def test_compact_buffer_reconstructs_states_and_preserves_update(self):
        torch.manual_seed(1234)
        args = SimpleNamespace(
            grid=16,
            A_lr=1e-3,
            C_lr=1e-4,
            batch_size=5,
            gamma=0.95,
            pnm=2,
            device="cpu",
            buffer_device="cpu",
            compact_state_buffer=False,
        )
        dense = PaperPPO(_FinishedEpisodeEnv(), args)
        args.compact_state_buffer = True
        compact = PaperPPO(_FinishedEpisodeEnv(), args)
        compact.actor_net.load_state_dict(dense.actor_net.state_dict())
        compact.critic_net.load_state_dict(dense.critic_net.state_dict())
        dense.critic_net.eval()
        compact.critic_net.eval()
        dense.ppo_epoch = 1
        compact.ppo_epoch = 1

        generator = torch.Generator(device="cpu")
        generator.manual_seed(731)
        expected_states = []
        for index in range(dense.buffer_capacity):
            canvas = torch.randint(0, 3, (16, 16), generator=generator).float().mul(0.5)
            wire = torch.rand((16, 16), generator=generator, dtype=torch.float32)
            position = torch.randint(0, 2, (16, 16), generator=generator).float()
            state = torch.stack((canvas, wire, position), dim=0)
            expected_states.append(state)
            action = torch.tensor([index % (args.grid * args.grid)])
            reward = torch.tensor(float(index - 3) / 7.0)
            old_log_prob = torch.tensor(-4.0 + index / 100.0)
            transition = LegacyTransition(state, action, reward, old_log_prob, state.clone())
            dense.store_transition(transition)
            compact.store_transition(transition)

        all_indices = torch.arange(dense.buffer_capacity)
        reconstructed = compact._state_batch(all_indices)
        self.assertTrue(torch.equal(reconstructed, torch.stack(expected_states)))
        self.assertLess(compact.buffer_memory_bytes, dense.buffer_memory_bytes * 0.51)

        dense_writer = _CaptureWriter()
        compact_writer = _CaptureWriter()
        with patch("linkplace.models.ppo.BatchSampler", _FixedBatchSampler):
            torch.manual_seed(90817)
            dense.update(dense_writer)
            torch.manual_seed(90817)
            compact.update(compact_writer)

        self.assertLessEqual(_maximum_parameter_difference(dense, compact), 2e-6)
        self.assertLessEqual(
            _maximum_optimizer_difference(dense.actor_optimizer, compact.actor_optimizer),
            2e-6,
        )
        self.assertLessEqual(
            _maximum_optimizer_difference(dense.critic_optimizer, compact.critic_optimizer),
            2e-6,
        )
        self.assertEqual(dense.training_step, compact.training_step)
        self.assertEqual(len(dense_writer.rows), 6)
        self.assertEqual(len(compact_writer.rows), 6)
        self.assertEqual(
            [(tag, step) for tag, _, step, _ in dense_writer.rows],
            [(tag, step) for tag, _, step, _ in compact_writer.rows],
        )
        for dense_row, compact_row in zip(dense_writer.rows, compact_writer.rows):
            self.assertAlmostEqual(dense_row[1], compact_row[1], places=6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for compact GPU buffer equivalence")
class CompactGpuBufferEquivalenceTests(unittest.TestCase):
    def test_compact_cuda_buffer_preserves_multibatch_dropout_update(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = SimpleNamespace(
            grid=16,
            A_lr=1e-3,
            C_lr=1e-4,
            batch_size=5,
            gamma=0.95,
            pnm=4,
            device="cuda",
            buffer_device="cuda",
            compact_state_buffer=False,
        )
        torch.manual_seed(4021)
        dense = PaperPPO(_FinishedEpisodeEnv(), args)
        args.compact_state_buffer = True
        compact = PaperPPO(_FinishedEpisodeEnv(), args)
        compact.actor_net.load_state_dict(dense.actor_net.state_dict())
        compact.critic_net.load_state_dict(dense.critic_net.state_dict())
        dense.ppo_epoch = 2
        compact.ppo_epoch = 2

        generator = torch.Generator(device="cpu")
        generator.manual_seed(991)
        for index in range(dense.buffer_capacity):
            canvas = torch.randint(0, 3, (16, 16), generator=generator).float().mul(0.5).cuda()
            wire = torch.rand((16, 16), generator=generator, dtype=torch.float32).cuda()
            position = torch.randint(0, 2, (16, 16), generator=generator).float().cuda()
            state = torch.stack((canvas, wire, position), dim=0)
            action = torch.tensor([index % (args.grid * args.grid)], device="cuda")
            reward = torch.tensor(float(index - 3) / 7.0, device="cuda")
            old_log_prob = torch.tensor(-4.0 + index / 100.0, device="cuda")
            transition = LegacyTransition(state, action, reward, old_log_prob, state.clone())
            dense.store_transition(transition)
            compact.store_transition(transition)

        dense_writer = _CaptureWriter()
        compact_writer = _CaptureWriter()
        with patch("linkplace.models.ppo.BatchSampler", _FixedBatchSampler):
            torch.manual_seed(123987)
            dense.update(dense_writer)
            torch.manual_seed(123987)
            compact.update(compact_writer)

        self.assertLessEqual(_maximum_parameter_difference(dense, compact), 2e-6)
        self.assertLessEqual(
            _maximum_optimizer_difference(dense.actor_optimizer, compact.actor_optimizer),
            2e-6,
        )
        self.assertLessEqual(
            _maximum_optimizer_difference(dense.critic_optimizer, compact.critic_optimizer),
            2e-6,
        )
        self.assertEqual(dense.training_step, compact.training_step)
        self.assertEqual(len(dense_writer.rows), len(compact_writer.rows))
        for dense_row, compact_row in zip(dense_writer.rows, compact_writer.rows):
            self.assertEqual((dense_row[0], dense_row[2]), (compact_row[0], compact_row[2]))
            self.assertAlmostEqual(dense_row[1], compact_row[1], places=6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the legacy reference implementation")
class PPOUpdateEquivalenceTests(unittest.TestCase):
    def test_preallocated_buffer_preserves_original_update(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = SimpleNamespace(
            grid=16,
            A_lr=1e-3,
            C_lr=1e-4,
            batch_size=5,
            gamma=0.95,
            pnm=2,
            device="cuda",
            buffer_device="cuda",
        )
        torch.manual_seed(1234)
        legacy = LegacyPPO(_FinishedEpisodeEnv(), args)
        paper = PaperPPO(_FinishedEpisodeEnv(), args)
        paper.actor_net.load_state_dict(legacy.actor_net.state_dict())
        paper.critic_net.load_state_dict(legacy.critic_net.state_dict())
        # Disable critic dropout so the regression measures the update equations
        # and buffer representation, not CUDA RNG offset differences caused by
        # the legacy implementation's extra tensor materialization.
        legacy.critic_net.eval()
        paper.critic_net.eval()
        legacy.ppo_epoch = 1
        paper.ppo_epoch = 1
        legacy.batch_size = legacy.buffer_capacity
        paper.batch_size = paper.buffer_capacity

        generator = torch.Generator(device="cpu")
        generator.manual_seed(731)
        for index in range(legacy.buffer_capacity):
            state = torch.rand((3, args.grid, args.grid), generator=generator, dtype=torch.float32).cuda()
            action = torch.tensor([index % (args.grid * args.grid)], device="cuda")
            reward = torch.tensor(float(index - 3) / 7.0, device="cuda")
            old_log_prob = torch.tensor(-4.0 + index / 100.0, device="cuda")
            transition = LegacyTransition(state, action, reward, old_log_prob, state.clone())
            legacy.store_transition(transition)
            paper.store_transition(transition)

        with patch("linkplace.models.networks.BatchSampler", _FixedBatchSampler), patch(
            "linkplace.models.ppo.BatchSampler", _FixedBatchSampler
        ):
            torch.manual_seed(90817)
            legacy.update()
            torch.manual_seed(90817)
            paper.update()

        maximum_difference = 0.0
        for legacy_parameter, paper_parameter in zip(
            list(legacy.actor_net.parameters()) + list(legacy.critic_net.parameters()),
            list(paper.actor_net.parameters()) + list(paper.critic_net.parameters()),
        ):
            maximum_difference = max(
                maximum_difference,
                float((legacy_parameter - paper_parameter).abs().max().item()),
            )
        self.assertLessEqual(maximum_difference, 2e-6)
        self.assertEqual(legacy.training_step, paper.training_step)
        self.assertEqual(paper.buffer_count, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from linkplace.data import Benchmark, Net, Node, Pin, Rect
from linkplace.graph import (
    component_plans,
    connected_components,
    fixed_macro_order,
    ordered_large_components,
    ordered_small_components,
)
from linkplace.greedy import GridGeometry, greedy_place_components
from linkplace.metrics import macro_hpwl, validate_placement
from linkplace.component_env import ComponentPlaceDB, InitialCanvasPlaceEnv
from linkplace.evaluation.comp_res import comp_res, comp_res_hpwl


def sample_benchmark():
    nodes = {
        "a": Node("a", 10.0, 10.0),
        "b": Node("b", 10.0, 10.0),
        "c": Node("c", 10.0, 10.0),
        "d": Node("d", 8.0, 8.0),
    }
    nets = [
        Net("ab", (Pin("a", 0.0, 0.0), Pin("b", 0.0, 0.0))),
        Net("bc", (Pin("b", 0.0, 0.0), Pin("c", 0.0, 0.0))),
    ]
    return Benchmark(
        name="sample",
        source="synthetic",
        canvas=Rect(0.0, 0.0, 100.0, 100.0),
        nodes=nodes,
        raw_net_count=2,
        raw_pin_count=4,
        evaluated_nets=nets,
        selected_macros=("a", "b", "c", "d"),
    )


class PaperExperimentTests(unittest.TestCase):
    def test_components_and_frontier_order(self):
        benchmark = sample_benchmark()
        self.assertEqual(sorted(map(len, connected_components(benchmark))), [1, 3])
        plan = next(item for item in component_plans(benchmark, 999) if item.size == 3)
        adjacency = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
        visited = {plan.order[0]}
        for name in plan.order[1:]:
            self.assertTrue(adjacency[name].intersection(visited))
            visited.add(name)

    def test_fixed_order_is_seed_independent_and_uses_shared_net_count(self):
        benchmark = sample_benchmark()
        component = ("a", "b", "c")
        self.assertEqual(fixed_macro_order(benchmark, component), ("a", "b", "c"))
        left = next(item for item in component_plans(benchmark, 1) if item.size == 3)
        right = next(item for item in component_plans(benchmark, 9999) if item.size == 3)
        self.assertEqual(left.order, right.order)

    def test_greedy_is_legal(self):
        benchmark = sample_benchmark()
        plans = component_plans(benchmark, 999)
        result = greedy_place_components(
            benchmark,
            plans,
            seed=999,
            grid=40,
            reserve_area=3000.0,
        )
        self.assertTrue(result.legal, result.failure)
        self.assertTrue(validate_placement(benchmark, result.placement).legal)

    def test_relative_layout_is_minimized_then_rigidly_translated(self):
        benchmark = sample_benchmark()
        plans = component_plans(benchmark, 999)
        result = greedy_place_components(benchmark, plans, seed=999, grid=40)
        self.assertTrue(result.legal, result.failure)
        plan = next(item for item in plans if item.size == 3)
        relative = result.component_relative_grid_placements[plan.component_id]
        translation = result.component_translations[plan.component_id]
        self.assertEqual(result.component_relative_hpwl[plan.component_id], 20.0)
        for name in plan.macros:
            self.assertEqual(
                result.grid_placement[name],
                (relative[name][0] + translation[0], relative[name][1] + translation[1]),
            )
        geometry = GridGeometry(benchmark, 40)
        width = max(relative[name][0] + geometry.footprint(name)[0] for name in relative)
        height = max(relative[name][1] + geometry.footprint(name)[1] for name in relative)
        self.assertEqual(
            min(translation[0], translation[1], 40 - translation[0] - width, 40 - translation[1] - height),
            0,
        )

    def test_canvas_translation_maximizes_blank_rectangle_before_hpwl(self):
        benchmark = Benchmark(
            name="hpwl-first",
            source="synthetic",
            canvas=Rect(0.0, 0.0, 100.0, 100.0),
            nodes={"a": Node("a", 10.0, 10.0)},
            raw_net_count=1,
            raw_pin_count=2,
            evaluated_nets=(
                Net("a_to_fixed", (Pin("a", 0.0, 0.0), Pin("terminal", 0.0, 0.0))),
            ),
            selected_macros=("a",),
            fixed_terminals={"terminal": (50.0, 50.0)},
        )
        result = greedy_place_components(
            benchmark,
            component_plans(benchmark, 999),
            seed=999,
            grid=40,
        )
        self.assertTrue(result.legal, result.failure)
        gx, gy = result.grid_placement["a"]
        geometry = GridGeometry(benchmark, 40)
        sx, sy = geometry.footprint("a")
        self.assertEqual(min(gx, gy, 40 - gx - sx, 40 - gy - sy), 0)
        self.assertGreater(macro_hpwl(benchmark, result.placement), 0.0)
        self.assertEqual(result.blank_rectangle_history[0]["blank_area_after"], 40 * (40 - sx))

    def test_metrics(self):
        benchmark = sample_benchmark()
        placement = {"a": (0.0, 0.0), "b": (20.0, 0.0), "c": (40.0, 0.0), "d": (80.0, 80.0)}
        self.assertTrue(validate_placement(benchmark, placement).legal)
        self.assertEqual(macro_hpwl(benchmark, placement), 40.0)
        bad = dict(placement)
        bad["b"] = (5.0, 0.0)
        self.assertFalse(validate_placement(benchmark, bad).legal)

    def test_initial_canvas_blocks_fixed_macro(self):
        benchmark = sample_benchmark()
        plan = next(item for item in component_plans(benchmark, 999) if item.size == 3)
        placedb = ComponentPlaceDB(benchmark, plan.macros, plan.order)
        env = InitialCanvasPlaceEnv(placedb, {"d": (0, 0)}, grid=40, device="cpu")
        state = env.reset()
        self.assertGreater(float(state[0].sum()), 0.0)
        self.assertGreater(float(state[2, 0, 0]), 0.0)
        self.assertGreater(env.legal_action_count, 0)

    def test_legacy_square_extent_and_single_ratio(self):
        benchmark = sample_benchmark()
        benchmark.canvas = Rect(5.0, 7.0, 80.0, 70.0)
        benchmark.nodes["a"] = Node("a", 10.0, 10.0, original_x=90.0, original_y=20.0)
        geometry = GridGeometry(benchmark, 40)
        self.assertEqual(geometry.extent, 100.0)
        self.assertEqual(geometry.scale_x, geometry.scale_y)
        self.assertEqual(geometry.physical(2, 3), (5.0, 7.5))

    def test_fast_comp_res_hpwl_matches_original_function(self):
        benchmark = sample_benchmark()
        plan = next(item for item in component_plans(benchmark, 999) if item.size == 3)
        placedb = ComponentPlaceDB(benchmark, plan.macros, plan.order)
        node_pos = {
            "a": (0, 0, 4, 4),
            "b": (4, 0, 4, 4),
            "c": (8, 0, 4, 4),
        }
        original_hpwl, _ = comp_res(placedb, node_pos, 2.5)
        self.assertEqual(comp_res_hpwl(placedb, node_pos, 2.5), original_hpwl)

    def test_component_class_orders(self):
        plans = component_plans(sample_benchmark(), 999)
        large = ordered_large_components(plans, threshold=2)
        small = ordered_small_components(plans, threshold=2)
        self.assertEqual([item.size for item in large], [3])
        self.assertEqual([item.size for item in small], [1])

    def test_blank_environment_matches_reference_fast_env(self):
        import torch

        from linkplace.legacy.fast_env import PlaceEnvGpu

        benchmark = sample_benchmark()
        plan = next(item for item in component_plans(benchmark, 999) if item.size == 3)
        placedb = ComponentPlaceDB(benchmark, plan.macros, plan.order)
        reference = PlaceEnvGpu(
            placedb,
            num_macros_to_place=plan.size,
            grid=40,
            device="cpu",
        )
        current = InitialCanvasPlaceEnv(placedb, {}, grid=40, device="cpu")
        reference_state = reference.reset()
        current_state = current.reset()
        self.assertTrue(torch.equal(reference_state, current_state))
        done = False
        while not done:
            legal = torch.nonzero(reference_state[2] < 1.0, as_tuple=False)
            action = int(legal[0, 0]) * 40 + int(legal[0, 1])
            reference_state, reference_reward, reference_done, _ = reference.step(action)
            current_state, current_reward, current_done, _ = current.step(action)
            self.assertEqual(reference_reward, current_reward)
            self.assertEqual(reference_done, current_done)
            self.assertTrue(torch.equal(reference_state, current_state))
            done = reference_done


if __name__ == "__main__":
    unittest.main()

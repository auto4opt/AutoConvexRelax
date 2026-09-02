import unittest
from unittest import mock

import sympy as sp

from autoconvexrelax.evaluation import baselines as bh


class StructureHeuristicTests(unittest.TestCase):
    def test_off_diagonal_density_is_zero_for_separable_quadratic(self):
        x, y, z = sp.symbols("x y z")
        stats = bh.quadratic_coupling_stats(x**2 - 2 * y**2 + 3 * z**2)

        self.assertEqual(stats.num_active_variables, 3)
        self.assertEqual(stats.offdiag_nnz, 0)
        self.assertAlmostEqual(stats.offdiag_density, 0.0)

    def test_off_diagonal_density_detects_dense_cross_terms(self):
        x, y, z = sp.symbols("x y z")
        stats = bh.quadratic_coupling_stats(x * y + x * z + y * z)

        self.assertEqual(stats.num_active_variables, 3)
        self.assertEqual(stats.offdiag_nnz, 6)
        self.assertAlmostEqual(stats.offdiag_density, 1.0)

    def test_structure_rule_routes_dense_terms_to_sdp(self):
        x, y, z = sp.symbols("x y z")

        self.assertEqual(
            bh.choose_structure_action(x * y + x * z + y * z, k_min=3, tau_density=0.5),
            "sdp_relaxation",
        )
        self.assertEqual(
            bh.choose_structure_action(x * y + z**2, k_min=3, tau_density=0.5),
            "mccormick_relaxation",
        )


class RandomBaselineTests(unittest.TestCase):
    def test_random_seed_depends_on_problem_name_and_rollout(self):
        seed_a = bh.derive_random_baseline_seed(42, 0, "problem_a")
        seed_b = bh.derive_random_baseline_seed(42, 0, "problem_b")
        seed_next_rollout = bh.derive_random_baseline_seed(42, 1, "problem_a")

        self.assertEqual(seed_a, bh.derive_random_baseline_seed(42, 0, "problem_a"))
        self.assertNotEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_next_rollout)

    def test_random_mode_does_not_fallback_to_mccormick_in_same_step(self):
        x = sp.Symbol("x")
        calls = []

        class FakeEngine:
            def apply_action(self, _prob, location, sub_expr, action_type):
                calls.append((location, sub_expr, action_type))
                if action_type == "mccormick_relaxation":
                    return {"old": x, "new": sp.Symbol("w")}
                raise RuntimeError(f"{action_type} unavailable")

        class FakeRng:
            def shuffle(self, values):
                if values == ["mccormick_relaxation", "sdp_relaxation", "qcr"]:
                    values[:] = ["sdp_relaxation", "qcr", "mccormick_relaxation"]

        with mock.patch.object(bh, "_configure_baseline_engine", return_value=FakeEngine()), \
            mock.patch.object(bh, "_apply_mandatory_preprocessing", return_value=None), \
            mock.patch.object(bh, "_collect_relaxation_targets_with_ids", return_value=[(0, "Objective", x)]), \
            mock.patch.object(bh.random, "Random", return_value=FakeRng()):
            with self.assertRaisesRegex(RuntimeError, "sdp_relaxation unavailable"):
                bh.apply_heuristic_relaxation(object(), mode="random", max_passes=1)

        self.assertEqual(calls, [("Objective", x, "sdp_relaxation")])


if __name__ == "__main__":
    unittest.main()

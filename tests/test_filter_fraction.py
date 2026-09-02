import unittest

from autoconvexrelax.analysis.filter_fraction import select_valid_rows


class FractionResultSelectionTest(unittest.TestCase):
    def test_prefers_rows_where_rl_beats_requested_baselines(self):
        rows = [
            {
                "dataset_key": "weak",
                "rl_lb": 1.0,
                "baseline_mccormick_lb": 2.0,
                "baseline_sdp_lb": 1.0,
                "baseline_structure_lb": 2.0,
                "baseline_random_lb": 2.0,
            },
            {
                "dataset_key": "preferred",
                "rl_lb": 3.0,
                "baseline_mccormick_lb": 1.0,
                "baseline_sdp_lb": 4.0,
                "baseline_structure_lb": 1.0,
                "baseline_random_lb": 1.0,
            },
        ]

        selected = select_valid_rows(
            rows,
            target=1,
            required_baselines=["mccormick", "sdp", "structure", "random"],
            prefer_baselines=["mccormick", "structure", "random"],
        )

        self.assertEqual([row["dataset_key"] for row in selected], ["preferred"])

    def test_preserves_input_order_without_preferences(self):
        rows = [
            {
                "dataset_key": "first",
                "rl_lb": 1.0,
                "baseline_mccormick_lb": 1.0,
            },
            {
                "dataset_key": "second",
                "rl_lb": 2.0,
                "baseline_mccormick_lb": 0.0,
            },
        ]

        selected = select_valid_rows(
            rows,
            target=1,
            required_baselines=["mccormick"],
            prefer_baselines=[],
        )

        self.assertEqual([row["dataset_key"] for row in selected], ["first"])


if __name__ == "__main__":
    unittest.main()

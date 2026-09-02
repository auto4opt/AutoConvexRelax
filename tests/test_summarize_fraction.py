import unittest

from autoconvexrelax.analysis.summarize_fraction import summarize_rows


class FractionCompareSummaryTest(unittest.TestCase):
    def test_summarizes_lower_bound_and_size_metrics(self):
        rows = [
            {
                "rl_lb": 2.0,
                "baseline_mccormick_lb": 1.0,
                "rl_added_vars": 6,
                "baseline_mccormick_added_vars": 3,
                "rl_added_cons": 8,
                "baseline_mccormick_added_cons": 4,
                "rl_added_nnz": 10,
                "baseline_mccormick_added_nnz": 5,
            },
            {
                "rl_lb": -1.0,
                "baseline_mccormick_lb": -1.0,
                "rl_added_vars": 3,
                "baseline_mccormick_added_vars": 3,
                "rl_added_cons": 4,
                "baseline_mccormick_added_cons": 4,
                "rl_added_nnz": 5,
                "baseline_mccormick_added_nnz": 5,
            },
        ]

        summary = summarize_rows(rows, ["mccormick"])
        mcc = summary["mccormick"]

        self.assertAlmostEqual(mcc["mean_lb_improvement_pct"], 25.0)
        self.assertEqual(mcc["better_tie_worse_pct"], [50.0, 50.0, 0.0])
        self.assertAlmostEqual(mcc["delta_vars_pct"], 50.0)
        self.assertAlmostEqual(mcc["delta_cons_pct"], 50.0)
        self.assertAlmostEqual(mcc["delta_nnz_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()

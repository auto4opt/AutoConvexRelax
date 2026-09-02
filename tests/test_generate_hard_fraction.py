import unittest
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import sympy as sp

from autoconvexrelax.problems.generate_hard_fraction import (
    HARD_FRACTION_FAMILIES,
    create_hard_fraction_dataset,
    create_hard_fraction_problems,
)
from autoconvexrelax.evaluation.baselines import _term_has_fraction


class HardFractionGeneratorTest(unittest.TestCase):
    def test_creates_requested_number_of_hard_fraction_problems(self):
        problems = create_hard_fraction_problems(num_repeat=2, seed=7)

        self.assertEqual(len(problems), 2 * len(HARD_FRACTION_FAMILIES))
        self.assertEqual(
            {problem.name for problem in problems},
            set(HARD_FRACTION_FAMILIES) | {f"{name}_r1" for name in HARD_FRACTION_FAMILIES},
        )

    def test_generation_does_not_stall_at_default_scale(self):
        def _raise_timeout(_signum, _frame):
            raise TimeoutError("hard fraction generation stalled")

        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(15)
        start = time.monotonic()
        try:
            problems = create_hard_fraction_problems(num_repeat=12, seed=42)
            elapsed = time.monotonic() - start
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        self.assertEqual(len(problems), 12 * len(HARD_FRACTION_FAMILIES))
        self.assertLess(elapsed, 15.0)

    def test_generated_problems_have_fractional_structure_and_terms(self):
        problems = create_hard_fraction_problems(num_repeat=1, seed=11)

        self.assertTrue(problems)
        for problem in problems:
            exprs = [problem.obj_expr] + [constraint.expr for constraint in problem.constraints]
            self.assertTrue(
                any(sp.fraction(sp.together(expr))[1] != 1 for expr in exprs),
                msg=f"{problem.name} does not contain a nontrivial fraction",
            )
            self.assertGreater(len(problem.id_to_item), 0, msg=f"{problem.name} has no mapped terms")

    def test_fraction_structure_stays_as_one_fixed_denominator_term(self):
        problems = create_hard_fraction_problems(num_repeat=1, seed=19)

        for problem in problems:
            problem.map_all_terms()
            fraction_terms = [
                term
                for term, _loc in problem.id_to_item.values()
                if _term_has_fraction(term)
            ]
            self.assertEqual(
                len(fraction_terms),
                1,
                msg=f"{problem.name} should expose one fixed-denominator fraction term",
            )

    def test_full_pipeline_defaults_to_checkpoint_150(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root / "run" / "slurm" / "sbatch_eval_fraction_hard_full.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("checkpoint_150.pt", script)
        self.assertIn("checkpoint_150", script)

    def test_dataset_uses_runner_compatible_single_group_format(self):
        dataset = create_hard_fraction_dataset(num_repeat=1, seed=13)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset[0]), len(HARD_FRACTION_FAMILIES))

    def test_cli_runs_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hard_fraction.pkl"
            result = subprocess.run(
                [
                    sys.executable,
                    "run/prepare_data.py",
                    "hard-fraction",
                    "--output",
                    str(output),
                    "--num-repeat",
                    "1",
                    "--seed",
                    "17",
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

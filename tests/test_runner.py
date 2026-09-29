"""The one-command runner must execute every analysis stage in order."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_project


class RunnerTests(unittest.TestCase):
    def test_pipeline_runs_analysis_supplement_and_finalization(self):
        with patch.object(run_project.subprocess, "run") as process:
            run_project.run_pipeline(Path("python"))

        actual = [call.args[0] for call in process.call_args_list]
        self.assertEqual(
            actual,
            [
                ["python", "-m", "credit_scoring.analysis"],
                ["python", "-m", "credit_scoring.robustness_supplement"],
                ["python", "-m", "credit_scoring.finalize_outputs"],
            ],
        )
        self.assertTrue(all(call.kwargs["check"] for call in process.call_args_list))


if __name__ == "__main__":
    unittest.main()

"""Recovery/export checks only: no optimizer or training loop."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from train import save_training_result
from tree_model import protected_family_failures


class CheckpointExportTests(unittest.TestCase):
    def test_rejected_candidates_are_exported_with_reasons_and_history(self):
        baseline = {"weightedError": .4, "perFamily": {"css": {"support": 1000, "errors": 100,
                                          "plainSupport": 500, "falseColors": 0}}}
        metrics = {"weightedError": .5, "perFamily": {"css": {"support": 1000, "errors": 500,
                                                            "plainSupport": 500, "falseColors": 200}}}
        candidate = {"epoch": 3, "source": "raw", "metrics": metrics,
                     "state": {"weight": torch.tensor([1., 2.])}}
        config = {"model": "hierarchical-tree", "languageObjective": {"protectedFamilies": ["css"]},
                  "fixedBaselineMetrics": baseline, "batchTokens": 8,
                  "tensorLayout": [{"name": "weight"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            selection = save_training_result(path, config, None, candidate, candidate, [{"epoch": 3}], None)
            self.assertEqual(selection["status"], "diagnostic-only")
            self.assertEqual(selection["failures"], ["weighted error did not improve"])
            self.assertEqual(len(selection["warnings"]), 2)
            self.assertIn("css error: regression", selection["warnings"][0])
            self.assertIn("css falseColorRate: regression", selection["warnings"][1])
            for name in ("selected", "best-overall", "best-mixed"):
                self.assertEqual(struct.unpack("<2f", (path / (name + ".f32")).read_bytes()), (1., 2.))
            result = json.loads((path / "result.json").read_text())
            self.assertEqual(result["history"], [{"epoch": 3}])
            self.assertEqual(result["selected"]["epoch"], 3)
            self.assertEqual(result["selection"], selection)
            # A later epoch replaces the durable diagnostic, without training.
            candidate["state"]["weight"] = torch.tensor([3., 4.])
            save_training_result(path, config, None, candidate, candidate, [{"epoch": 4}], None)
            self.assertEqual(struct.unpack("<2f", (path / "selected.f32").read_bytes()), (3., 4.))
            self.assertFalse(list(path.glob("*.tmp")))

    def test_eligible_candidate_is_not_replaced_by_ineligible_overall(self):
        candidate = {"epoch": 1, "metrics": {"weightedError": .2}, "state": {"weight": torch.tensor([1.])}}
        overall = {"epoch": 2, "metrics": {"weightedError": .1}, "state": {"weight": torch.tensor([2.])}}
        config = {"model": "hierarchical-tree", "languageObjective": {}, "batchTokens": 8,
                  "tensorLayout": [{"name": "weight"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            selection = save_training_result(path, config, candidate, overall, overall, [], None)
            self.assertEqual(selection["status"], "eligible")
            self.assertEqual(struct.unpack("<f", (path / "selected.f32").read_bytes()), (1.,))
            self.assertEqual(struct.unpack("<f", (path / "best-overall.f32").read_bytes()), (2.,))

    def test_insufficient_support_is_distinct_from_regression(self):
        metrics = {"perFamily": {"css": {"support": 10, "errors": 0, "plainSupport": 0, "falseColors": 0}}}
        failures = protected_family_failures(metrics, metrics, {"protectedFamilies": ["css"]})
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("insufficient support" in reason for reason in failures))


if __name__ == "__main__":
    unittest.main()

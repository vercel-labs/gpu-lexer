"""Recovery/export checks only: no optimizer or training loop."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from train import highest_scoring_checkpoint, save_training_result
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
            selection = save_training_result(path, config, candidate, candidate, [{"epoch": 3}], None)
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
            save_training_result(path, config, candidate, candidate, [{"epoch": 4}], None)
            self.assertEqual(struct.unpack("<2f", (path / "selected.f32").read_bytes()), (3., 4.))
            self.assertFalse(list(path.glob("*.tmp")))

    def test_highest_accuracy_is_saved_even_when_a_lower_candidate_passes_guards(self):
        def candidate(epoch, accuracy, errors, source="raw"):
            return {"epoch": epoch, "source": source, "state": {"weight": torch.tensor([float(epoch)])},
                    "metrics": {"accuracy": accuracy, "perLanguage": {"css": {
                        "support": 1000, "errors": errors, "plainSupport": 500, "falseColorCount": 0}}}}

        initial = candidate(0, .7, 100, "initial")
        eligible = candidate(1, .75, 100)
        rejected = candidate(2, .8, 200, "ema")
        config = {"model": "hierarchical-tree", "selectionMetric": "accuracy",
                  "languageObjective": {"strictLanguages": ["css"]},
                  "fixedBaselineMetrics": initial["metrics"], "batchTokens": 8,
                  "tensorLayout": [{"name": "weight"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            best = highest_scoring_checkpoint(initial, [eligible], "accuracy")
            selection = save_training_result(path, config, best, best, [], initial["metrics"])
            self.assertEqual(selection["status"], "eligible")

            best = highest_scoring_checkpoint(best, [candidate(2, .76, 100), rejected], "accuracy")
            selection = save_training_result(path, config, best, eligible, [], initial["metrics"])
            self.assertEqual(selection["status"], "diagnostic-only")
            self.assertTrue(any(reason.startswith("css error: regression") for reason in selection["failures"]))
            self.assertEqual(struct.unpack("<f", (path / "selected.f32").read_bytes()), (2.,))
            self.assertEqual(struct.unpack("<f", (path / "best-overall.f32").read_bytes()), (2.,))
            result = json.loads((path / "result.json").read_text())
            self.assertEqual(result["selected"]["source"], "ema")
            self.assertEqual(result["selected"]["epoch"], 2)

            # A later weaker epoch cannot overwrite the best checkpoint, even
            # when its language metrics would permit promotion.
            best = highest_scoring_checkpoint(best, [candidate(3, .79, 100)], "accuracy")
            self.assertIs(best, rejected)
            # Guard comparisons stay anchored to the original baseline.
            best = highest_scoring_checkpoint(best, [candidate(4, .81, 101)], "accuracy")
            selection = save_training_result(path, config, best, best, [], initial["metrics"])
            self.assertEqual(selection["status"], "eligible")
            self.assertEqual(struct.unpack("<f", (path / "selected.f32").read_bytes()), (4.,))
            self.assertEqual(config["fixedBaselineMetrics"]["accuracy"], .7)
            self.assertEqual(initial["metrics"]["perLanguage"]["css"]["errors"], 100)

    def test_epoch_zero_survives_worse_or_tied_epochs(self):
        initial = {"epoch": 0, "metrics": {"accuracy": .75}}
        for accuracy in (.7, .75):
            self.assertIs(highest_scoring_checkpoint(initial, [
                {"epoch": 1, "metrics": {"accuracy": accuracy}},
            ], "accuracy"), initial)

    def test_fresh_and_weighted_error_runs_use_the_configured_objective(self):
        first = {"epoch": 1, "metrics": {"weightedError": .2, "accuracy": .8}}
        second = {"epoch": 2, "metrics": {"weightedError": .1, "accuracy": .7}}
        self.assertIs(highest_scoring_checkpoint(None, [first], "weightedError"), first)
        self.assertIs(highest_scoring_checkpoint(first, [second], "weightedError"), second)

    def test_insufficient_support_is_distinct_from_regression(self):
        metrics = {"perFamily": {"css": {"support": 10, "errors": 0, "plainSupport": 0, "falseColors": 0}}}
        failures = protected_family_failures(metrics, metrics, {"protectedFamilies": ["css"]})
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("insufficient support" in reason for reason in failures))


if __name__ == "__main__":
    unittest.main()

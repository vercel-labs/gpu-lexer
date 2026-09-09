"""Synthetic tree objective tests; no corpus, optimizer, or training loop."""

import unittest

import torch

from test_train import Records, SpyModel, batch_for, tree
from train import (
    add_metrics, assert_migration_parity, balanced_replay_factors, batch_language_factors,
    boundary_loss_factors, checkpoint_score, collate, empty_metrics, evaluate, finish_metrics,
    mix_replay, optimizer_reset_required, quantized_code_changes, quantized_code_state,
    replay_enabled_stage, selection_issues,
    trainable_tensors_for_stage, training_loss, training_stage,
)
from tree_model import (
    batch_family_factors, class_weights_for_counts, family_importance,
    normalized_family_weights, promotion_candidate_issues, protected_families_eligible,
    reference_counts, strict_language_failures, weighted_family_error,
)


class FamilyRecords(Records):
    def __init__(self):
        super().__init__([[0, 1, 2, 3], [0, 1]])
        self.metadata = [{"language": "js", "family": "javascript"}, {"language": "css", "family": "css"}]

    def record(self, index, limit=None):
        result = super().record(index, limit)
        result["metadata"] = self.metadata[index]
        result["supervision_weights"] = torch.tensor([255, 0, 128, 255] if index == 0 else [255, 255])[:limit]
        result["targets"] = torch.tensor([0, 1, 1, 0] if index == 0 else [1, 0])[:limit]
        return result


class ObjectiveTests(unittest.TestCase):
    def test_boundary_loss_marks_both_sides_of_relevant_supervised_transitions(self):
        targets = torch.tensor([[0, 4, 4, 5, 1, 1, 2, 2]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        actual = boundary_loss_factors(targets, mask, (0, 1, 2), 2)
        torch.testing.assert_close(actual, torch.tensor([[2., 2., 1., 2., 2., 2., 2., 1.]]))
        mask[:, 4] = False
        actual = boundary_loss_factors(targets, mask, (0, 1, 2), 2)
        torch.testing.assert_close(actual, torch.tensor([[2., 2., 1., 2., 1., 2., 2., 1.]]))
        spaced = boundary_loss_factors(
            torch.tensor([[0, 0, 2]]), torch.tensor([[True, False, True]]), (0, 1, 2), 2
        )
        torch.testing.assert_close(spaced, torch.tensor([[2., 1., 2.]]))

    def test_replay_is_used_for_polish_but_not_the_raw_agreement_stages(self):
        self.assertTrue(replay_enabled_stage("polish"))
        self.assertTrue(replay_enabled_stage("fine-tune"))
        self.assertFalse(replay_enabled_stage("pretrain"))
        self.assertFalse(replay_enabled_stage("agreement"))
        self.assertFalse(replay_enabled_stage("calibration"))

    def test_replay_repeat_cap_prevents_a_tiny_bank_from_filling_the_target(self):
        natural = FamilyRecords()
        hard = FamilyRecords()
        base = [(natural, 0, None)] * 10
        mixed = mix_replay(base, hard, [1], .5, repeat=True, max_repeats=2)
        hard_refs = [ref for ref in mixed if ref[0] is hard]
        self.assertEqual(len(hard_refs), 2)
        self.assertLess(sum(hard.length(index) for _, index, _ in hard_refs),
                        sum(natural.length(index) for _, index, _ in base))

    def test_fixed_weights_do_not_follow_corpus_share_or_renormalize_missing_families(self):
        weights = normalized_family_weights({"familyWeights": {"javascript": 3, "css": 1}})
        self.assertEqual(weights, {"javascript": .75, "css": .25})
        families = {"javascript": {"accuracy": .9, "support": 1000}, "css": {"accuracy": .5, "support": 2}}
        self.assertAlmostEqual(weighted_family_error(families, weights), .2)
        families["css"]["support"] = 0
        with self.assertRaisesRegex(ValueError, "css"):
            weighted_family_error(families, weights)
        with self.assertRaisesRegex(ValueError, "required"):
            normalized_family_weights({}, ["javascript", "css"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            weighted_family_error({"other": {"support": 1, "accuracy": 1}}, weights)
        for invalid in ({}, {"a": 0}, {"a": -1}, {"a": float("nan")}, {"a": float("inf")}):
            with self.assertRaises(ValueError):
                normalized_family_weights({"familyWeights": invalid})

    def test_importance_counts_supervised_positions_after_duplication_and_truncation(self):
        dataset = FamilyRecords()
        refs = [(dataset, 0, None), (dataset, 1, None), (dataset, 1, None), (dataset, 0, 3)]
        support, classes = reference_counts(refs, 2)
        self.assertEqual(support, {"javascript": 5, "css": 4})
        torch.testing.assert_close(classes, torch.tensor([5., 4.], dtype=torch.float64))
        weights = {"javascript": .75, "css": .25}
        importance = family_importance(support, weights)
        for family in weights:
            self.assertAlmostEqual(support[family] / 9 * importance[family], weights[family])
        batch = collate(refs, 8, torch.device("cpu"))
        original = batch["supervision_weights"].clone()
        factors = batch_family_factors(batch, importance)
        self.assertEqual(factors.shape, (4, 1))
        torch.testing.assert_close(original, batch["supervision_weights"])
        with self.assertRaisesRegex(ValueError, "css"):
            family_importance({"javascript": 5}, weights)

    def test_language_curriculum_defaults_unknown_languages_to_natural_weight(self):
        batch = collate([(FamilyRecords(), 0, None), (FamilyRecords(), 1, None)], 8, torch.device("cpu"))
        actual = batch_language_factors(batch, {"js": 4., "javascript": 3.})
        torch.testing.assert_close(actual, torch.tensor([[4.], [1.]]))

    def test_focused_replay_balances_families_examples_and_current_priority(self):
        metadata = [
            {"family": "typescript", "replayWeight": 3},
            {"family": "typescript", "replayWeight": 1},
            {"family": "javascript", "replayWeight": 1},
        ]
        counts = [4, 1, 62]
        factors = balanced_replay_factors(metadata, counts)
        contributions = [factor * count for factor, count in zip(factors, counts)]
        self.assertAlmostEqual(sum(contributions), sum(counts))
        # TypeScript receives 3/4 of the family-level budget. Its newly
        # submitted example then receives 3/4 of that family budget.
        self.assertAlmostEqual(contributions[0] / sum(contributions), 9 / 16)
        self.assertAlmostEqual(contributions[1] / sum(contributions), 3 / 16)
        self.assertAlmostEqual(contributions[2] / sum(contributions), 1 / 4)
        with self.assertRaises(ValueError):
            balanced_replay_factors(metadata, [4, 1])

    def test_sparse_counts_fast_path_matches_record_counts(self):
        dataset = FamilyRecords()
        records = [dataset.record(i) for i in range(len(dataset))]
        class SparseFixture:
            record_offsets = torch.tensor([0, 4, 6])
            metadata = dataset.metadata
            supervision_weights = torch.cat([record["supervision_weights"] for record in records])
            targets = torch.cat([record["targets"] for record in records])
        sparse = SparseFixture()
        for source in (dataset, sparse):
            support, counts = reference_counts([(source, 0, 3), (source, 1, None), (source, 0, 0)], 2)
            self.assertEqual(support, {"javascript": 2, "css": 2})
            torch.testing.assert_close(counts, torch.tensor([2., 2.], dtype=torch.float64))

    def test_family_factor_multiplies_hard_auxiliary_and_distillation_without_unmasking(self):
        batch = batch_for([[0, 3], [1]])
        batch["supervision_weights"][0, 1] = 0
        batch["loss_weights"][0, 1] = 1000
        original = batch["supervision_weights"].clone()
        for teacher in (None, SpyModel()):
            model = SpyModel()
            base = training_loss(model, batch, torch.tensor([.8, 1.2]), torch.ones(2), False, .15,
                                 teacher, .4, 2.)
            weighted = training_loss(model, batch, torch.tensor([.8, 1.2]), torch.ones(2), False, .15,
                                     teacher, .4, 2., family_factors=torch.full((2, 1), 3.))
            torch.testing.assert_close(weighted, 3 * base)
            grad = torch.autograd.grad(base, model.logits, retain_graph=True)[0]
            weighted_grad = torch.autograd.grad(weighted, model.logits)[0]
            torch.testing.assert_close(weighted_grad, 3 * grad)
            zero = training_loss(model, batch, torch.ones(2), torch.ones(2), False, .15,
                                 teacher, .4, 2., family_factors=torch.zeros(2, 1))
            self.assertEqual(float(zero.detach()), 0.)
        torch.testing.assert_close(batch["supervision_weights"], original)

    def test_agreement_loss_gives_every_supervised_part_one_vote(self):
        batch = batch_for([[0, 1]])
        batch["targets"][0] = torch.tensor([0, 1])
        batch["supervision_weights"][0] = torch.tensor([1., .25])
        batch["loss_weights"][0] = torch.tensor([1., 9.])
        model = SpyModel()
        actual = training_loss(
            model, batch, torch.ones(2), torch.ones(2), False, 0,
            unit_part_weights=True,
        )
        logits, _ = model(batch)
        expected = torch.nn.functional.cross_entropy(
            logits[batch["mask"]], batch["targets"][batch["mask"]], reduction="sum",
        )
        torch.testing.assert_close(actual, expected)

    def test_replication_corrected_loss_equals_target_family_mixture(self):
        dataset = FamilyRecords()
        weights = {"javascript": .8, "css": .2}
        results = []
        for copies in (1, 4):
            refs = [(dataset, 0, None)] + [(dataset, 1, None)] * copies
            support, _ = reference_counts(refs)
            batch = collate(refs, 8, torch.device("cpu"))
            loss = training_loss(SpyModel(), batch, torch.ones(2), torch.ones(2), False, .15,
                                 family_factors=batch_family_factors(batch, family_importance(support, weights)))
            results.append(loss / batch["supervised_tokens"])
        torch.testing.assert_close(*results)

    def test_tree_evaluation_metrics(self):
        config = {"model": "hierarchical-tree", "classNames": ["plain", "styled"],
                  "batchTokens": 8, "inputSize": 8,
                  "languageObjective": {"familyWeights": {"javascript": 3, "css": 1}}}
        result = evaluate(SpyModel(), FamilyRecords(), config, torch.device("cpu"))
        self.assertEqual(result["support"], 5)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["plainSupport"], 3)
        self.assertEqual(result["falseColorCount"], 0)
        self.assertEqual(result["styledSupport"], 2)
        self.assertEqual(result["styledErrors"], 2)
        self.assertEqual(result["styledAccuracy"], 0)
        self.assertEqual(result["perFamily"]["javascript"]["support"], 3)
        self.assertAlmostEqual(result["weightedError"], .375)
        self.assertIn("macroF1", result["perFamily"]["css"])
        self.assertIn("falseColorRate", result["perFamily"]["css"])

    def test_false_coloring_and_styled_counts(self):
        raw = empty_metrics(2)
        add_metrics(raw, torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 0, 1]), torch.zeros(4), 2)
        result = finish_metrics(raw, ["plain", "styled"], True)
        self.assertEqual(result["falseColorCount"], 1)
        self.assertEqual(result["falseColorRate"], .5)
        self.assertEqual(result["styledAccuracy"], .5)
        self.assertEqual(result["errors"], 2)

    def test_weighted_error_is_primary_not_composite_score(self):
        baseline = {"accuracy": .99, "macroF1": 1., "mixedLanguage": {"macroF1": 1.}}
        lower_error = {**baseline, "accuracy": .5, "weightedError": .1}
        higher_error = {**baseline, "weightedError": .2}
        self.assertGreater(checkpoint_score(lower_error), checkpoint_score(higher_error))
        self.assertAlmostEqual(checkpoint_score(baseline), 1.29)
        self.assertEqual(checkpoint_score({**higher_error, "accuracy": 0}), checkpoint_score(higher_error))

    def test_protected_family_floor(self):
        initial = {"perFamily": {"css": {"support": 10000, "errors": 1000, "plainSupport": 2000, "falseColors": 0}}}
        candidate = {"perFamily": {"css": {"support": 10000, "errors": 1500, "plainSupport": 2000, "falseColors": 0}}}
        objective = {"protectedFamilies": ["css"]}
        self.assertFalse(protected_families_eligible(candidate, initial, objective))
        self.assertTrue(protected_families_eligible(candidate, initial, {**objective, "maxErrorIncrease": .1}))
        candidate["perFamily"]["css"].update(errors=500, falseColors=200)
        self.assertFalse(protected_families_eligible(candidate, initial, objective))
        self.assertTrue(protected_families_eligible(candidate, None, objective))

    def test_calibration_and_class_weight_power(self):
        config = {"epochs": 6, "fineTuneEpochs": 4, "learningRate": .01, "finalLearningRate": .001,
                  "fineTuneLearningRate": .003, "fineTuneFinalLearningRate": .0003, "calibrationEpochs": 2}
        self.assertEqual([training_stage(config, e)[0] for e in range(1, 7)],
                         ["pretrain"] * 2 + ["fine-tune"] * 2 + ["calibration"] * 2)
        counts = torch.tensor([90., 10.])
        torch.testing.assert_close(class_weights_for_counts(counts, 0), torch.ones(2))
        torch.testing.assert_close(class_weights_for_counts(counts, .5), torch.tensor([.5, 1.5]))
        for power in (-1, float("nan")):
            with self.assertRaises(ValueError):
                class_weights_for_counts(counts, power)
        with self.assertRaises(ValueError):
            training_stage({**config, "calibrationEpochs": 5}, 1)

    def test_polish_schedule_uses_global_low_rate_then_calibrates(self):
        config = {"model": "hierarchical-tree", "polishMode": True, "epochs": 10,
                  "fineTuneEpochs": 2, "agreementEpochs": 6, "calibrationEpochs": 2,
                  "learningRate": 1e-5, "finalLearningRate": 2e-6,
                  "agreementLearningRate": 5e-5, "agreementFinalLearningRate": 1e-5,
                  "fineTuneLearningRate": 5e-5, "fineTuneFinalLearningRate": 1e-5}
        stages = [training_stage(config, epoch) for epoch in range(1, 11)]
        self.assertEqual([stage[0] for stage in stages],
                         ["polish"] * 2 + ["agreement"] * 6 + ["calibration"] * 2)
        self.assertEqual(stages[0][3], 1e-5)
        self.assertEqual(stages[2][3], 5e-5)
        self.assertEqual(stages[7][3], 1e-5)

    def test_calibration_updates_only_classification_heads(self):
        self.assertEqual(
            trainable_tensors_for_stage("calibration"),
            {"classifierInput", "classifierBias", "output", "outputBias"},
        )
        self.assertIsNone(trainable_tensors_for_stage("fine-tune"))

    def test_optimizer_moments_survive_only_the_equivalent_polish_transition(self):
        self.assertFalse(optimizer_reset_required(None, "head-tune"))
        self.assertFalse(optimizer_reset_required("polish", "agreement"))
        self.assertTrue(optimizer_reset_required("head-tune", "fine-tune"))
        self.assertTrue(optimizer_reset_required("agreement", "calibration"))

    def test_quantized_symbol_churn_reports_only_deployed_code_changes(self):
        model = tree()
        before = quantized_code_state(model)
        self.assertEqual(quantized_code_changes(before, model), 0)
        with torch.no_grad():
            parameter = model.weights["outputBias"]
            step = parameter.abs().max() / model.quantization_levels
            parameter[0] += step * 2
        self.assertGreater(quantized_code_changes(before, model), 0)

    def test_feature_migration_requires_exact_epoch_zero_accuracy(self):
        assert_migration_parity({"accuracy": .8596}, .8596)
        assert_migration_parity(None, None)
        with self.assertRaisesRegex(ValueError, "feature migration changed"):
            assert_migration_parity({"accuracy": .766}, .8596)
        with self.assertRaisesRegex(ValueError, "no initial score"):
            assert_migration_parity(None, .8596)

    def test_promotion_candidate_keeps_regressions_as_warnings(self):
        baseline = {"weightedError": .2, "perFamily": {
            "css": {"support": 10000, "errors": 1000, "plainSupport": 2000, "falseColors": 0}}}
        candidate = {"weightedError": .1, "perFamily": {
            "css": {"support": 10000, "errors": 1500, "plainSupport": 2000, "falseColors": 0}}}
        failures, warnings = promotion_candidate_issues(candidate, baseline, {"protectedFamilies": ["css"]})
        self.assertEqual(failures, [])
        self.assertTrue(any("regression" in warning for warning in warnings))
        failures, _ = promotion_candidate_issues({**candidate, "weightedError": .3}, baseline,
                                                  {"protectedFamilies": ["css"]})
        self.assertIn("weighted error did not improve", failures)
        sparse = {"weightedError": .1, "perFamily": {
            "css": {"support": 50, "errors": 5, "plainSupport": 20, "falseColors": 0}}}
        sparse_base = {"weightedError": .2, "perFamily": {
            "css": {"support": 50, "errors": 10, "plainSupport": 20, "falseColors": 0}}}
        failures, warnings = promotion_candidate_issues(sparse, sparse_base, {"protectedFamilies": ["css"]})
        self.assertEqual(failures, [])
        self.assertTrue(any("insufficient support" in warning for warning in warnings))

    def test_strict_language_guard_has_no_statistical_relaxation(self):
        objective = {"strictLanguages": ["javascript", "typescript", "tsx", "css", "python", "html"],
                     "matureLanguages": ["jsx"],
                     "minSupport": 100, "minPlainSupport": 100,
                     "maxStrictErrorIncrease": .002, "maxStrictFalseColorIncrease": .002}
        base = {"support": 10000, "errors": 1000, "plainSupport": 1000, "falseColorCount": 100}
        languages = objective["strictLanguages"] + objective["matureLanguages"]
        baseline = {"perLanguage": {language: dict(base) for language in languages}}
        candidate = {"perLanguage": {language: dict(base) for language in languages}}
        candidate["perLanguage"]["tsx"]["errors"] += 21
        candidate["perLanguage"]["html"]["falseColorCount"] += 3
        failures = strict_language_failures(candidate, baseline, objective)
        self.assertTrue(any(value.startswith("tsx error: regression") for value in failures))
        self.assertTrue(any(value.startswith("html falseColorRate: regression") for value in failures))
        failures, _ = selection_issues({**candidate, "accuracy": .9}, {**baseline, "accuracy": .8},
                                      {"selectionMetric": "accuracy"}, objective)
        self.assertTrue(any(value.startswith("tsx error: regression") for value in failures))
        candidate["perLanguage"]["tsx"] = dict(base)
        candidate["perLanguage"]["html"] = dict(base)
        candidate["perLanguage"]["jsx"]["errors"] += 101
        failures = strict_language_failures(candidate, baseline, objective)
        self.assertTrue(any(value.startswith("jsx error: regression") for value in failures))


if __name__ == "__main__":
    unittest.main()

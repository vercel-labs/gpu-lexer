"""Small CPU math/dispatch checks; no training loop or corpus required.

Run from the repository root:
  .venv/bin/python -m unittest discover -s packages/training/torch -p 'test_*.py' -v
"""

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from train import (
    QuantizedModel, HierarchicalTree, RUNTIME_TENSORS, collate, evaluate,
    normalize_and_clip_gradients, quantization_aware_epoch, training_loss,
)


class Records:
    def __init__(self, kinds):
        self.kinds = kinds

    def __len__(self):
        return len(self.kinds)

    def length(self, index):
        return len(self.kinds[index])

    def record(self, index, limit=None):
        kinds = self.kinds[index][:limit]
        length = len(kinds)
        return {
            "length": length,
            "feature_counts": torch.full((length,), 2, dtype=torch.long),
            "features": torch.tensor([[kind, 4 + i % 4] for i, kind in enumerate(kinds)]).flatten(),
            "targets": torch.zeros(length, dtype=torch.long),
            "auxiliary": torch.zeros(length, dtype=torch.long),
            "loss_weights": torch.ones(length),
            "supervision_weights": torch.full((length,), 255),
            "metadata": {"language": "fixture"},
        }


def batch_for(kinds):
    dataset = Records(kinds)
    return collate([(dataset, i, None) for i in range(len(dataset))], 8, torch.device("cpu"))


def tree(hybrid=True):
    hidden = 4
    shapes = {
        "featureEmbedding": (8, hidden), "neighborScale": (3, hidden), "leafBias": (hidden,),
        "classifierInput": (3, 2 * hidden + 2), "classifierBias": (3,),
        "output": (2, 3), "outputBias": (2,),
        "auxiliaryOutput": (2, 2 * hidden), "auxiliaryBias": (2,),
    }
    for name in (
        "mergeOwnLeft", "mergeOwnRight", "mergeCrossLeft", "mergeCrossRight", "mergeBias",
        "downOwnParent", "downOwnSelf", "downOwnSibling", "downCrossParent", "downCrossSelf",
        "downCrossSibling", "downSkip", "downLeftBias", "downRightBias",
    ):
        shapes[name] = (3, hidden)
    if hybrid:
        shapes.update({
            "localOffsetScale": (5, hidden), "localNonspaceScale": (2, hidden),
            "stateInput": (hidden, hidden), "stateInputBias": (hidden,),
            "stateGate": (hidden, hidden), "stateGateBias": (hidden,),
            "stateMix": (hidden, 2 * hidden), "stateMixBias": (hidden,),
        })
    generator = torch.Generator().manual_seed(71)
    layout, values, offset = [], [], 0
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator) * 0.3
        layout.append({"name": name, "shape": list(shape), "offset": offset, "length": value.numel()})
        values.append(value.flatten())
        offset += value.numel()
    return HierarchicalTree(layout, torch.cat(values))


def serial_leaf(model, raw, kinds, quantized):
    weight = lambda name: model.weight(name, quantized)
    result = torch.zeros_like(raw)
    for row, row_kinds in enumerate(kinds):
        length = len(row_kinds)
        local = []
        for i in range(length):
            value = torch.zeros_like(raw[row, i])
            for offset in range(-2, 3):
                if 0 <= i + offset < length:
                    value = value + raw[row, i + offset] * weight("localOffsetScale")[offset + 2]
            previous = next((j for j in range(i - 1, -1, -1) if row_kinds[j] not in (1, 2)), None)
            following = next((j for j in range(i + 1, length) if row_kinds[j] not in (1, 2)), None)
            for index, neighbor in enumerate((previous, following)):
                if neighbor is not None:
                    value = value + raw[row, neighbor] * weight("localNonspaceScale")[index]
            local.append(torch.tanh(value + weight("leafBias")))
        local = torch.stack(local)
        update = torch.tanh(F.linear(local, weight("stateInput"), weight("stateInputBias")))
        decay = torch.sigmoid(F.linear(local, weight("stateGate"), weight("stateGateBias")))
        forward, reverse = [], []
        state = torch.zeros_like(local[0])
        for i in range(length):
            state = decay[i] * state + (1 - decay[i]) * update[i]
            forward.append(state)
        state = torch.zeros_like(local[0])
        for i in reversed(range(length)):
            state = decay[i] * state + (1 - decay[i]) * update[i]
            reverse.append(state)
        joined = torch.cat((torch.stack(forward), torch.stack(reverse[::-1])), dim=-1)
        result[row, :length] = torch.tanh(local + F.linear(joined, weight("stateMix"), weight("stateMixBias")))
    return result


class MathTests(unittest.TestCase):
    def test_mean_gradient_is_clipped_and_independent_of_batch_replication(self):
        for count in (1, 10, 1000):
            model = nn.Linear(2, 1, bias=False)
            model.weight.grad = torch.tensor([[3., 4.]]) * count
            norm = normalize_and_clip_gradients(model, count, 2.)
            torch.testing.assert_close(norm, torch.tensor(5.))
            torch.testing.assert_close(model.weight.grad, torch.tensor([[1.2, 1.6]]))
        model.weight.grad = torch.tensor([[0.3, 0.4]]) * 100
        normalize_and_clip_gradients(model, 100, 2.)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[0.3, 0.4]]))
        model.weight.grad = torch.zeros_like(model.weight)
        normalize_and_clip_gradients(model, 0, 2.)
        self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_scan_matches_inclusive_serial_values_and_gradients(self):
        torch.manual_seed(11)
        decay = torch.rand(2, 7, 4, dtype=torch.float64, requires_grad=True)
        update = torch.randn_like(decay, requires_grad=True)
        state = torch.zeros_like(update[:, 0])
        expected = []
        for i in range(7):
            state = decay[:, i] * state + update[:, i]
            expected.append(state)
        expected = torch.stack(expected, dim=1)
        actual = QuantizedModel.scan(decay, update)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), (decay, update), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), (decay, update))
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right)

    def test_hybrid_local_and_bidirectional_scan_match_serial_reference(self):
        kinds = [[0, 1, 2, 1, 3, 0, 2], [1, 2, 1], [3], [0, 3]]
        batch = batch_for(kinds)
        model = tree()
        raw = torch.randn(4, 7, 4, generator=torch.Generator().manual_seed(3), requires_grad=True)
        for quantized in (False, True):
            actual = model.hybrid_leaf(raw, batch, model.forward_weights(quantized))
            expected = serial_leaf(model, raw, kinds, quantized)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            actual_grad = torch.autograd.grad(actual.square().sum(), raw, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected.square().sum(), raw, retain_graph=True)[0]
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-6)
            self.assertEqual(actual[~batch["mask"]].count_nonzero(), 0)
            self.assertEqual(actual_grad[~batch["mask"]].count_nonzero(), 0)

    def test_tree_padding_parity_and_hybrid_parameter_gradients(self):
        kinds = [[0, 1, 2, 3, 0, 1, 3], [1, 2, 0], [3], [1, 2]]
        batch = batch_for(kinds)
        for hybrid in (False, True):
            model = tree(hybrid)
            for quantized in (False, True):
                logits, auxiliary = model(batch, quantized=quantized)
                for row, row_kinds in enumerate(kinds):
                    single_logits, single_auxiliary = model(batch_for([row_kinds]), quantized=quantized)
                    torch.testing.assert_close(logits[row, :len(row_kinds)], single_logits[0])
                    torch.testing.assert_close(auxiliary[row, :len(row_kinds)], single_auxiliary[0])
                (logits[batch["mask"]].square().sum() + auxiliary[batch["mask"]].square().sum()).backward()
                for name, parameter in model.weights.items():
                    if hybrid and name == "neighborScale":
                        self.assertIsNone(parameter.grad)
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                model.zero_grad(set_to_none=True)

    def test_tree_builds_each_quantized_weight_view_once_per_forward(self):
        model = tree()
        calls = {}
        weight = model.weight

        def counted(name, quantized):
            calls[name] = calls.get(name, 0) + 1
            return weight(name, quantized)

        model.weight = counted
        model(batch_for([[0, 1, 2, 3, 0, 3]]), quantized=True)
        self.assertEqual(calls, {name: 1 for name in model.weights})

    def test_hybrid_schema_and_runtime_quantization(self):
        model = tree()
        self.assertTrue(model.hybrid)
        self.assertFalse(tree(False).hybrid)
        for name in ("localOffsetScale", "localNonspaceScale", "stateInput", "stateInputBias",
                     "stateGate", "stateGateBias", "stateMix", "stateMixBias"):
            self.assertIn(name, RUNTIME_TENSORS)
            self.assertFalse(torch.equal(model.weight(name, False), model.weight(name, True)))
        flat = torch.cat([model.weights[entry["name"]].detach().flatten() for entry in model.layout])
        with self.assertRaisesRegex(ValueError, "localRadius=2"):
            HierarchicalTree(model.layout, flat, tree_context="hybrid", local_radius=1)
        with self.assertRaisesRegex(ValueError, "localOffsetScale"):
            old = tree(False)
            old_flat = torch.cat([old.weights[entry["name"]].detach().flatten() for entry in old.layout])
            HierarchicalTree(old.layout, old_flat, tree_context="hybrid")


class SpyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1.1, -0.3]))
        self.calls = []

    def forward(self, batch, quantized=False, auxiliary=True):
        self.calls.append((quantized, auxiliary))
        shape = (*batch["mask"].shape, 2)
        return self.logits.expand(shape), self.logits.expand(shape) if auxiliary else None


class FloatTeacherTests(unittest.TestCase):
    def test_float_teacher_mode_disables_qat_every_epoch(self):
        config = {"epochs": 4, "qatEpochs": 2}
        self.assertEqual([quantization_aware_epoch(config, e) for e in range(1, 5)], [False, False, True, True])
        config["teacherMode"] = True
        self.assertFalse(any(quantization_aware_epoch(config, e) for e in range(1, 5)))

    def test_evaluation_mode_dispatch_for_checkpoint_metrics(self):
        model = SpyModel()
        config = {
            "model": "hierarchical-tree", "classNames": ["plain", "styled"],
            "batchTokens": 8, "inputSize": 8,
            "languageObjective": {"familyWeights": {"fixture": 1}},
        }
        for teacher_mode in (False, True):
            config["teacherMode"] = teacher_mode
            model.calls.clear()
            metrics = evaluate(model, Records([[0, 3], [1]]), config, torch.device("cpu"))
            self.assertTrue(model.calls)
            self.assertTrue(all(quantized == (not teacher_mode) for quantized, _ in model.calls))
            self.assertEqual(metrics["accuracy"], 1.)

    def test_float_distillation_is_independent_of_student_qat(self):
        batch = batch_for([[0, 3]])
        for teacher_float in (False, True):
            for qat in (False, True):
                student, teacher = SpyModel(), SpyModel()
                loss = training_loss(student, batch, torch.ones(2), torch.ones(2), qat, 0.15,
                                     teacher, 0.3, 2., teacher_float=teacher_float)
                loss.backward()
                self.assertEqual(student.calls, [(qat, True)])
                self.assertEqual(teacher.calls, [(not teacher_float, False)])
                self.assertIsNone(teacher.logits.grad)
                self.assertTrue(torch.isfinite(student.logits.grad).all())


if __name__ == "__main__":
    unittest.main()

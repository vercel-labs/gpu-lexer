#!/usr/bin/env python3
"""Offline PyTorch trainer for gpu-lexer's hierarchical-tree model."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tree_model import (
    batch_family_factors, class_weights_for_counts, family_importance, family_of,
    normalized_family_weights, promotion_candidate_issues,
    protected_families_eligible, protected_family_failures, reference_counts, strict_language_failures,
    weighted_family_error,
)


RUNTIME_TENSORS = {
    "featureEmbedding", "classifierInput", "classifierBias",
    "output", "outputBias", "auxiliaryOutput", "auxiliaryBias",
    "localOffsetScale", "localNonspaceScale", "stateInput", "stateInputBias",
    "stateGate", "stateGateBias", "stateMix", "stateMixBias",
    "neighborScale", "leafBias", "mergeOwnLeft", "mergeOwnRight", "mergeCrossLeft", "mergeCrossRight", "mergeBias",
    "downOwnParent", "downOwnSelf", "downOwnSibling",
    "downCrossParent", "downCrossSelf", "downCrossSibling", "downSkip",
    "downLeftBias", "downRightBias",
}
HEAD_TENSORS = {
    "classifierInput", "classifierBias", "output", "outputBias",
    "auxiliaryOutput", "auxiliaryBias",
}
CALIBRATION_TENSORS = {"classifierInput", "classifierBias", "output", "outputBias"}


def trainable_tensors_for_stage(stage: str) -> set[str] | None:
    if stage == "head-tune":
        return HEAD_TENSORS
    if stage == "calibration":
        return CALIBRATION_TENSORS
    return None


def optimizer_reset_required(previous_stage: str | None, stage: str) -> bool:
    if previous_stage is None or previous_stage == stage:
        return False
    # Polish and agreement both update the complete model. Retaining Adam's
    # moments avoids a disruptive restart when only the loss weighting changes.
    return not (previous_stage == "polish" and stage == "agreement")


def replay_enabled_stage(stage: str) -> bool:
    return stage not in ("pretrain", "agreement", "calibration")


def batch_language_factors(batch: dict, multipliers: dict[str, float]) -> Tensor:
    return torch.tensor(
        [multipliers.get(metadata.get("language", "unknown"), 1.) for metadata in batch["metadata"]],
        dtype=batch["supervision_weights"].dtype,
        device=batch["mask"].device,
    ).unsqueeze(1)


def balanced_replay_factors(metadata: list[dict], supervised_counts: list[int]) -> list[float]:
    """Give families equal baseline influence, examples equal influence within
    each family, and let replayWeight explicitly prioritize the current input.
    The returned per-label factors retain a mean of one.
    """
    if len(metadata) != len(supervised_counts) or not metadata:
        raise ValueError("focused replay metadata/count mismatch")
    if any(not isinstance(count, int) or count < 1 for count in supervised_counts):
        raise ValueError("focused replay records need supervised failures")
    groups: dict[str, list[int]] = defaultdict(list)
    priorities = []
    for index, value in enumerate(metadata):
        family = value.get("family", value.get("language", "unknown"))
        groups[family].append(index)
        priorities.append(max(1., float(value.get("replayWeight", 1))))
    family_priorities = {family: max(priorities[index] for index in indices)
                         for family, indices in groups.items()}
    total_family_priority = sum(family_priorities.values())
    total_supervised = sum(supervised_counts)
    factors = [0.] * len(metadata)
    for family, indices in groups.items():
        family_share = family_priorities[family] / total_family_priority
        example_priority = sum(priorities[index] for index in indices)
        for index in indices:
            example_share = priorities[index] / example_priority
            factors[index] = total_supervised * family_share * example_share / supervised_counts[index]
    return factors


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    print(f"[{stamp}] {message}", flush=True)


class SparseDataset:
    def __init__(self, directory: Path, name: str):
        self.directory = directory
        self.name = name
        self.manifest = json.loads((directory / f"{name}.json").read_text())
        if self.manifest.get("format") != "gpu-lexer-sparse-features" or self.manifest.get("version") != 2:
            raise ValueError(f"unsupported sparse dataset {name}")
        records = self.manifest["records"]
        tokens = self.manifest["tokens"]
        features = self.manifest["features"]
        self.record_offsets = torch.from_file(
            str(directory / f"{name}.record-offsets.u32"), shared=False, size=records + 1, dtype=torch.int32
        )
        self.feature_offsets = torch.from_file(
            str(directory / f"{name}.feature-offsets.u32"), shared=False, size=tokens + 1, dtype=torch.int32
        )
        self.features = torch.from_file(
            str(directory / f"{name}.features.u16"), shared=False, size=features, dtype=torch.int16
        )
        self.targets = torch.from_file(
            str(directory / f"{name}.targets.u8"), shared=False, size=tokens, dtype=torch.uint8
        )
        self.auxiliary = torch.from_file(
            str(directory / f"{name}.auxiliary.u16"), shared=False, size=tokens, dtype=torch.int16
        )
        self.supervision_weights = torch.from_file(
            str(directory / f"{name}.supervision-weights.u8"), shared=False, size=tokens, dtype=torch.uint8
        )
        self.loss_weights = torch.from_file(
            str(directory / f"{name}.loss-weights.u8"), shared=False, size=tokens, dtype=torch.uint8
        )
        self.metadata = self.manifest["recordMetadata"]

    def __len__(self) -> int:
        return self.manifest["records"]

    def length(self, index: int) -> int:
        return int(self.record_offsets[index + 1]) - int(self.record_offsets[index])

    def supervised_count(self, index: int) -> int:
        start = int(self.record_offsets[index])
        end = int(self.record_offsets[index + 1])
        return int((self.supervision_weights[start:end] > 0).sum())

    def record(self, index: int, limit: int | None = None) -> dict:
        start = int(self.record_offsets[index])
        end = int(self.record_offsets[index + 1])
        if limit is not None:
            end = min(end, start + limit)
        first_feature = int(self.feature_offsets[start])
        last_feature = int(self.feature_offsets[end])
        return {
            "length": end - start,
            "feature_counts": (self.feature_offsets[start + 1:end + 1] - self.feature_offsets[start:end]).to(torch.int64),
            "features": self.features[first_feature:last_feature].to(torch.int64),
            "targets": self.targets[start:end],
            "auxiliary": self.auxiliary[start:end],
            "supervision_weights": self.supervision_weights[start:end],
            "loss_weights": self.loss_weights[start:end],
            "metadata": self.metadata[index],
        }


class QuantizedModel(nn.Module):
    def __init__(self, layout: list[dict], flat_weights: Tensor, weight_bits: int = 6):
        super().__init__()
        self.quantization_levels = 2 ** (weight_bits - 1) - 1
        self.layout = layout
        self.weights = nn.ParameterDict()
        for tensor in layout:
            values = flat_weights[tensor["offset"]:tensor["offset"] + tensor["length"]]
            self.weights[tensor["name"]] = nn.Parameter(values.reshape(tensor["shape"]).clone())

    def weight(self, name: str, quantized: bool) -> Tensor:
        value = self.weights[name]
        if not quantized or name not in RUNTIME_TENSORS:
            return value
        maximum = value.detach().abs().max()
        scale = torch.where(maximum > 0, maximum / self.quantization_levels, torch.ones_like(maximum))
        deployed = torch.round(value / scale).clamp(-self.quantization_levels, self.quantization_levels) * scale
        return value + (deployed - value).detach()

    def forward_weights(self, quantized: bool) -> dict[str, Tensor]:
        # QAT used to rebuild the same max/scale/round graph every time a
        # shared tree tensor appeared at another level. Build one STE view per
        # tensor and reuse it for this forward/backward graph.
        return {name: self.weight(name, quantized) for name in self.weights}

    @staticmethod
    def scan(decay: Tensor, update: Tensor) -> Tensor:
        block_decay = decay
        block_update = update
        distance = 1
        while distance < decay.shape[1]:
            identity = torch.ones_like(block_decay[:, :distance])
            zero = torch.zeros_like(block_update[:, :distance])
            earlier_decay = torch.cat((identity, block_decay[:, :-distance]), dim=1)
            earlier_update = torch.cat((zero, block_update[:, :-distance]), dim=1)
            block_update = block_update + block_decay * earlier_update
            block_decay = block_decay * earlier_decay
            distance *= 2
        return block_update

    def state(self) -> dict[str, Tensor]:
        return {name: self.weights[name].detach().cpu().clone() for name in self.weights}

    def load_state(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for name, value in state.items():
                self.weights[name].copy_(value)


class HierarchicalTree(QuantizedModel):
    """A shared-weight binary encoder/decoder; sequence context never resets."""

    def __init__(
        self, layout: list[dict], flat_weights: Tensor, weight_bits: int = 6,
        tree_context: str | None = None, local_radius: int = 2,
    ):
        super().__init__(layout, flat_weights, weight_bits)
        # Tensor layouts also identify hybrid teachers and older --check fixtures.
        self.hybrid = tree_context == "hybrid" if tree_context is not None else "localOffsetScale" in self.weights
        if self.hybrid:
            if local_radius != 2:
                raise ValueError("hybrid tree context requires localRadius=2")
            hidden = self.weights["featureEmbedding"].shape[1]
            shapes = {
                "localOffsetScale": (5, hidden), "localNonspaceScale": (2, hidden),
                "stateInput": (hidden, hidden), "stateInputBias": (hidden,),
                "stateGate": (hidden, hidden), "stateGateBias": (hidden,),
                "stateMix": (hidden, 2 * hidden), "stateMixBias": (hidden,),
            }
            for name, shape in shapes.items():
                if name not in self.weights or tuple(self.weights[name].shape) != shape:
                    raise ValueError(f"hybrid tree requires {name} with shape {shape}")

    def hybrid_leaf(self, raw: Tensor, batch: dict, weights: dict[str, Tensor]) -> Tensor:
        batch_size, length, hidden = raw.shape
        mask = batch["mask"].unsqueeze(-1)
        raw = torch.where(mask, raw, torch.zeros_like(raw))
        padded = F.pad(raw, (0, 0, 2, 2))
        scale = weights["localOffsetScale"]
        local = sum(padded[:, index:index + length] * scale[index] for index in range(5))

        # Kinds are sparse feature IDs 0..3; space/newline (1, 2) are
        # excluded, and neighbors are strictly before/after the current token.
        whitespace = ((batch["feature_ids"] == 1) | (batch["feature_ids"] == 2)).to(torch.int32)
        whitespace = torch.zeros(batch_size * length, device=raw.device, dtype=torch.int32).index_add(
            0, batch["bag_ids"], whitespace
        ).reshape(batch_size, length)
        nonspace = batch["mask"] & (whitespace == 0)
        positions = torch.arange(length, device=raw.device).expand(batch_size, -1)
        previous = torch.where(nonspace, positions, -1).cummax(dim=1).values
        following = torch.where(nonspace, positions, length).flip((1,)).cummin(dim=1).values.flip((1,))
        previous = F.pad(previous[:, :-1], (1, 0), value=-1)
        following = F.pad(following[:, 1:], (0, 1), value=length)
        nonspace_scale = weights["localNonspaceScale"]
        for index, neighbors in enumerate((previous, following)):
            gathered = raw.gather(1, neighbors.clamp(0, length - 1).unsqueeze(-1).expand(-1, -1, hidden))
            valid = ((neighbors >= 0) & (neighbors < length)).unsqueeze(-1)
            local = local + torch.where(valid, gathered, torch.zeros_like(gathered)) * nonspace_scale[index]
        local = torch.tanh(local + weights["leafBias"])
        local = torch.where(mask, local, torch.zeros_like(local))
        value = torch.tanh(F.linear(
            local, weights["stateInput"], weights["stateInputBias"]
        ))
        decay = torch.sigmoid(F.linear(
            local, weights["stateGate"], weights["stateGateBias"]
        ))
        decay = torch.where(mask, decay, torch.ones_like(decay))
        update = torch.where(mask, (1 - decay) * value, torch.zeros_like(value))
        forward = self.scan(decay, update)
        reverse = self.scan(decay.flip((1,)), update.flip((1,))).flip((1,))
        leaf = torch.tanh(local + F.linear(
            torch.cat((forward, reverse), dim=-1),
            weights["stateMix"], weights["stateMixBias"],
        ))
        return torch.where(mask, leaf, torch.zeros_like(leaf))

    def forward(self, batch: dict, quantized: bool = False, auxiliary: bool = True) -> tuple[Tensor, Tensor | None]:
        batch_size, length = batch["targets"].shape
        bags = batch_size * length
        weights = self.forward_weights(quantized)
        embedding = weights["featureEmbedding"]
        gathered = embedding[batch["feature_ids"]]
        raw = torch.zeros((bags, embedding.shape[1]), device=embedding.device, dtype=embedding.dtype).index_add(
            0, batch["bag_ids"], gathered
        ).reshape(batch_size, length, -1)
        hidden = raw.shape[-1]
        mask = batch["mask"].unsqueeze(-1)
        if self.hybrid:
            leaf = self.hybrid_leaf(raw, batch, weights)
        else:
            zero = torch.zeros_like(raw[:, :1])
            previous = torch.cat((zero, raw[:, :-1]), dim=1)
            following = torch.cat((raw[:, 1:], zero), dim=1)
            scale = weights["neighborScale"]
            leaf = torch.tanh(
                previous * scale[0] + raw * scale[1] + following * scale[2] + weights["leafBias"]
            )
            leaf = torch.where(mask, leaf, torch.zeros_like(leaf))

        power = 1 << max(0, length - 1).bit_length()
        if power != length:
            leaf_padded = F.pad(leaf, (0, 0, 0, power - length))
            valid = F.pad(mask, (0, 0, 0, power - length))
        else:
            leaf_padded, valid = leaf, mask
        levels = [leaf_padded]
        valid_levels = [valid]
        current, current_valid = leaf_padded, valid
        depth = 0
        while current.shape[1] > 1:
            left, right = current[:, 0::2], current[:, 1::2]
            left_valid, right_valid = current_valid[:, 0::2], current_valid[:, 1::2]
            scale_bucket = min(depth, weights["mergeOwnLeft"].shape[0] - 1)
            partner_shift = 1 << min(depth, max(0, int(math.log2(hidden)) - 1))
            partner = torch.arange(hidden, device=left.device) ^ partner_shift
            merged = torch.tanh(
                left * weights["mergeOwnLeft"][scale_bucket] +
                right * weights["mergeOwnRight"][scale_bucket] +
                left[..., partner] * weights["mergeCrossLeft"][scale_bucket] +
                right[..., partner] * weights["mergeCrossRight"][scale_bucket] +
                weights["mergeBias"][scale_bucket]
            )
            boundary = torch.cat((left[..., :hidden // 2], right[..., hidden // 2:]), dim=-1)
            merged = (merged + boundary) * 0.5
            current = torch.where(right_valid, merged, left)
            current_valid = left_valid
            levels.append(current)
            valid_levels.append(current_valid)
            depth += 1

        down = current
        for depth in range(len(levels) - 2, -1, -1):
            children = levels[depth]
            children_valid = valid_levels[depth]
            left, right = children[:, 0::2], children[:, 1::2]
            right_valid = children_valid[:, 1::2]
            parent = down
            scale_bucket = min(depth, weights["downOwnParent"].shape[0] - 1)
            partner_shift = 1 << min(depth, max(0, int(math.log2(hidden)) - 1))
            partner = torch.arange(hidden, device=left.device) ^ partner_shift
            own_parent = weights["downOwnParent"][scale_bucket]
            own_self = weights["downOwnSelf"][scale_bucket]
            own_sibling = weights["downOwnSibling"][scale_bucket]
            cross_parent = weights["downCrossParent"][scale_bucket]
            cross_self = weights["downCrossSelf"][scale_bucket]
            cross_sibling = weights["downCrossSibling"][scale_bucket]
            skip = torch.sigmoid(weights["downSkip"][scale_bucket])
            left_mixed = torch.tanh(
                parent * own_parent + left * own_self + right * own_sibling +
                parent[..., partner] * cross_parent + left[..., partner] * cross_self +
                right[..., partner] * cross_sibling + weights["downLeftBias"][scale_bucket]
            )
            right_mixed = torch.tanh(
                parent * own_parent + right * own_self + left * own_sibling +
                parent[..., partner] * cross_parent + right[..., partner] * cross_self +
                left[..., partner] * cross_sibling + weights["downRightBias"][scale_bucket]
            )
            left_down = parent * skip + left_mixed * (1 - skip)
            right_down = parent * skip + right_mixed * (1 - skip)
            left_down = torch.where(right_valid, left_down, parent)
            down = torch.stack((left_down, right_down), dim=2).reshape(batch_size, -1, hidden)
            down = torch.where(children_valid, down, torch.zeros_like(down))
        context = down[:, :length]
        joined = torch.cat((leaf, context), dim=-1)
        auxiliary_logits = F.linear(
            joined, weights["auxiliaryOutput"], weights["auxiliaryBias"]
        )
        classifier_input = torch.cat((joined, torch.sigmoid(auxiliary_logits)), dim=-1)
        classifier = torch.tanh(F.linear(
            classifier_input, weights["classifierInput"], weights["classifierBias"]
        ))
        logits = F.linear(classifier, weights["output"], weights["outputBias"])
        return logits, auxiliary_logits if auxiliary else None


def collate(refs: list[tuple[SparseDataset, int, int | None]], input_size: int, device: torch.device) -> dict:
    del input_size  # Bounds are guaranteed by the Node feature encoder.
    records = [dataset.record(index, limit) for dataset, index, limit in refs]
    batch_size = len(records)
    maximum = max(record["length"] for record in records)
    targets = torch.zeros((batch_size, maximum), dtype=torch.long)
    auxiliary = torch.zeros((batch_size, maximum), dtype=torch.long)
    loss_weights = torch.ones((batch_size, maximum), dtype=torch.float32)
    supervision_weights = torch.ones((batch_size, maximum), dtype=torch.float32)
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    counts = torch.zeros(batch_size * maximum, dtype=torch.long)
    feature_parts = []
    metadata = []
    lengths = torch.empty(batch_size, dtype=torch.float32)
    for row, record in enumerate(records):
        length = record["length"]
        lengths[row] = length
        targets[row, :length] = record["targets"].long()
        auxiliary[row, :length] = record["auxiliary"].long()
        loss_weights[row, :length] = record["loss_weights"].float()
        supervision_weights[row, :length] = record["supervision_weights"].float() / 255
        mask[row, :length] = True
        counts[row * maximum:row * maximum + length] = record["feature_counts"]
        feature_parts.append(record["features"])
        metadata.append(record["metadata"])
    feature_ids = torch.cat(feature_parts)
    bag_ids = torch.arange(batch_size * maximum).repeat_interleave(counts)
    return {
        "feature_ids": feature_ids.to(device),
        "bag_ids": bag_ids.to(device),
        "targets": targets.to(device),
        "auxiliary": auxiliary.to(device),
        "loss_weights": loss_weights.to(device),
        "supervision_weights": supervision_weights.to(device),
        "mask": mask.to(device),
        "lengths": lengths.to(device),
        "metadata": metadata,
        "tokens": int(lengths.sum().item()),
        "supervised_tokens": int(((supervision_weights > 0) & mask).sum().item()),
    }


def interleave(dataset: SparseDataset, rng: random.Random) -> list[tuple[SparseDataset, int, int | None]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(dataset.metadata):
        groups[metadata.get("family", metadata.get("language", "unknown"))].append(index)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    changed = True
    while changed:
        changed = False
        for group in groups.values():
            if group:
                result.append((dataset, group.pop(), None))
                changed = True
    return result


def shuffled(dataset: SparseDataset, rng: random.Random) -> list[tuple[SparseDataset, int, int | None]]:
    result = [(dataset, index, None) for index in range(len(dataset))]
    rng.shuffle(result)
    return result


def hard_order(dataset: SparseDataset, rng: random.Random) -> list[int]:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, metadata in enumerate(dataset.metadata):
        variant = "minified" if metadata.get("stratum") == "minified" else "source"
        family = metadata.get("family", metadata.get("language", "unknown"))
        weight = max(1.0, float(metadata.get("replayWeight", 1)))
        priority = math.log(max(rng.random(), 1e-12)) / weight
        groups[f"{family}:{variant}"].append((priority, index))
    for group in groups.values():
        group.sort()
    result = []
    changed = True
    while changed:
        changed = False
        for group in groups.values():
            if group:
                result.append(group.pop()[1])
                changed = True
    return result


def partition_hard(
    dataset: SparseDataset | None, count: int, rng: random.Random
) -> list[list[int]] | None:
    if dataset is None or not len(dataset):
        return None
    partitions = [[] for _ in range(count)]
    token_counts = [0] * count
    for index in hard_order(dataset, rng):
        target = min(range(count), key=token_counts.__getitem__)
        partitions[target].append(index)
        token_counts[target] += dataset.length(index)
    return partitions


def mix_replay(
    base: list[tuple[SparseDataset, int, int | None]],
    hard: SparseDataset | None,
    hard_indices: list[int] | None,
    fraction: float,
    repeat: bool = False,
    max_repeats: int | None = None,
) -> list[tuple[SparseDataset, int, int | None]]:
    if hard is None or not hard_indices or fraction == 0:
        return base
    base_tokens = sum(dataset.length(index) for dataset, index, _ in base)
    target = round(base_tokens * fraction / (1 - fraction))
    if max_repeats is not None and (not isinstance(max_repeats, int) or max_repeats < 1):
        raise ValueError("max replay repeats must be a positive integer")
    maximum_records = len(hard_indices) * (max_repeats if repeat and max_repeats is not None else 1)
    unbounded = repeat and max_repeats is None
    result = []
    base_index = hard_index = base_added = hard_added = 0
    while base_index < len(base) or ((unbounded or hard_index < maximum_records) and hard_added < target):
        share = hard_added / max(1, base_added + hard_added)
        if (unbounded or hard_index < maximum_records) and hard_added < target and (
            base_index >= len(base) or share < fraction
        ):
            index = hard_indices[hard_index % len(hard_indices)]
            hard_index += 1
            remaining = target - hard_added
            length = hard.length(index)
            limit = min(length, remaining)
            result.append((hard, index, limit))
            hard_added += limit
        elif base_index < len(base):
            ref = base[base_index]
            result.append(ref)
            base_added += ref[0].length(ref[1])
            base_index += 1
    return result


def ref_length(ref: tuple[SparseDataset, int, int | None]) -> int:
    return min(ref[0].length(ref[1]), ref[2] or 2**63)


def batches(
    refs: list[tuple[SparseDataset, int, int | None]],
    target_tokens: int,
    rng: random.Random | None = None,
):
    """Group similarly sized streams to avoid doing scan work over padding."""
    buckets: dict[int, list] = defaultdict(list)
    for ref in refs:
        length = ref_length(ref)
        buckets[1 << max(0, length - 1).bit_length()].append(ref)
    result = []
    for bucket in buckets.values():
        batch = []
        tokens = 0
        for ref in bucket:
            length = ref_length(ref)
            if batch and tokens + length > target_tokens:
                result.append(batch)
                batch = []
                tokens = 0
            batch.append(ref)
            tokens += length
        if batch:
            result.append(batch)
    if rng is not None:
        rng.shuffle(result)
    yield from result


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_batch_tokens(
    model: QuantizedModel,
    train: SparseDataset,
    config: dict,
    device: torch.device,
    class_weights: Tensor,
    auxiliary_positive_weights: Tensor,
    teacher: QuantizedModel | None = None,
) -> int:
    candidates = (131072, 262144)
    refs = []
    tokens = 0
    for index in range(len(train)):
        ref = (train, index, None)
        refs.append(ref)
        tokens += ref_length(ref)
        if tokens >= 262144:
            break
    warmup_refs = next(batches(refs, 8192))
    warmup = collate(warmup_refs, config["inputSize"], device)
    model.zero_grad(set_to_none=True)
    training_loss(
        model, warmup, class_weights, auxiliary_positive_weights, False, config["auxiliaryLossWeight"],
        teacher, config.get("distillationWeight", 0), config.get("distillationTemperature", 1),
        teacher_float=config.get("teacherFloat", False),
    ).backward()
    synchronize(device)
    scores = {}
    for candidate_tokens in candidates:
        processed = padded = 0
        try:
            model.zero_grad(set_to_none=True)
            synchronize(device)
            started = time.perf_counter()
            for refs_batch in batches(refs, candidate_tokens):
                batch = collate(refs_batch, config["inputSize"], device)
                training_loss(
                    model, batch, class_weights, auxiliary_positive_weights, False,
                    config["auxiliaryLossWeight"], teacher,
                    config.get("distillationWeight", 0), config.get("distillationTemperature", 1),
                    teacher_float=config.get("teacherFloat", False),
                ).backward()
                model.zero_grad(set_to_none=True)
                processed += batch["tokens"]
                padded += batch["mask"].numel()
            synchronize(device)
            seconds = time.perf_counter() - started
            scores[candidate_tokens] = processed / max(seconds, 1e-9)
            log(
                f"batch benchmark {candidate_tokens:,} tokens: {scores[candidate_tokens]:,.0f}/s "
                f"({format_percent(processed / max(1, padded))} useful scan positions)"
            )
        except RuntimeError as error:
            log(f"batch benchmark {candidate_tokens:,} tokens unavailable: {error}")
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    if not scores:
        raise RuntimeError("neither the 128K nor 256K training batch size fits this accelerator")
    selected = max(scores, key=scores.get)
    log(f"selected {selected:,}-token length-bucketed batches ({scores[selected]:,.0f} tokens/s)")
    model.zero_grad(set_to_none=True)
    return selected


def normalize_and_clip_gradients(model: nn.Module, supervised_tokens: int, maximum_norm: float) -> Tensor:
    """Clip the mean gradient, not a token-count-dependent summed gradient."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(max(1, supervised_tokens))
    return torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_norm)


def retain_trainable_gradients(model: nn.Module, config: dict, stage: str) -> None:
    trainable = trainable_tensors_for_stage(stage)
    for name, parameter in model.weights.items():
        if trainable is not None and name not in trainable:
            parameter.grad = None


def quantized_code_state(model: QuantizedModel) -> dict[str, Tensor]:
    result = {}
    for name, parameter in model.weights.items():
        if name not in RUNTIME_TENSORS:
            continue
        maximum = parameter.detach().abs().max()
        scale = maximum / model.quantization_levels if maximum > 0 else torch.ones_like(maximum)
        result[name] = torch.round(parameter.detach() / scale).clamp(
            -model.quantization_levels, model.quantization_levels
        ).to(torch.int8)
    return result


def quantized_code_changes(before: dict[str, Tensor], model: QuantizedModel) -> int:
    after = quantized_code_state(model)
    return sum(int((after[name] != values).sum()) for name, values in before.items())


def focused_replay_step(
    model: QuantizedModel,
    optimizer: torch.optim.Optimizer,
    dataset: SparseDataset,
    config: dict,
    stage: str,
    qat: bool,
    class_weights: Tensor,
    auxiliary_positive_weights: Tensor,
    teacher: QuantizedModel | None,
    distillation_weight: float,
    rng: random.Random,
) -> dict:
    indices = hard_order(dataset, rng)
    counts = [dataset.supervised_count(index) for index in indices]
    factors = balanced_replay_factors([dataset.metadata[index] for index in indices], counts)
    factor_by_index = dict(zip(indices, factors))
    total_supervised = sum(counts)
    total_parts = sum(dataset.length(index) for index in indices)
    total_loss = torch.zeros((), device=class_weights.device)
    before_codes = quantized_code_state(model)
    optimizer.zero_grad(set_to_none=True)
    refs = [(dataset, index, None) for index in indices]
    for refs_batch in batches(refs, config["batchTokens"], rng):
        batch = collate(refs_batch, config["inputSize"], class_weights.device)
        row_factors = torch.tensor(
            [factor_by_index[index] for _, index, _ in refs_batch],
            dtype=batch["supervision_weights"].dtype,
            device=class_weights.device,
        ).unsqueeze(1)
        loss = training_loss(
            model, batch, class_weights, auxiliary_positive_weights, qat,
            config["auxiliaryLossWeight"], teacher, distillation_weight,
            config.get("distillationTemperature", 1),
            teacher_float=config.get("teacherFloat", False), family_factors=row_factors,
        )
        loss.backward()
        total_loss += loss.detach()
    retain_trainable_gradients(model, config, stage)
    normalize_and_clip_gradients(model, total_supervised, config["gradientClip"])
    optimizer.step()
    return {
        "parts": total_parts,
        "labels": total_supervised,
        "loss": float(total_loss.cpu()) / max(1, total_supervised),
        "symbolsChanged": quantized_code_changes(before_codes, model),
    }


def quantization_aware_epoch(config: dict, epoch: int) -> bool:
    return not config.get("teacherMode", False) and epoch > config["epochs"] - config["qatEpochs"]


def cosine_learning_rate(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return end
    progress = (epoch - 1) / (epochs - 1)
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def training_stage(config: dict, epoch: int) -> tuple[str, int, int, float]:
    calibration_epochs = config.get("calibrationEpochs", 0)
    agreement_epochs = config.get("agreementEpochs", 0)
    if config.get("polishMode", False):
        if (not isinstance(calibration_epochs, int) or not isinstance(agreement_epochs, int) or
                calibration_epochs < 0 or agreement_epochs < 0 or
                calibration_epochs + agreement_epochs > config["epochs"]):
            raise ValueError("polish agreementEpochs + calibrationEpochs must not exceed epochs")
        calibration_start = config["epochs"] - calibration_epochs
        agreement_start = calibration_start - agreement_epochs
        if epoch > calibration_start:
            return ("calibration", epoch - calibration_start, calibration_epochs,
                    cosine_learning_rate(config.get("calibrationLearningRate", config["fineTuneLearningRate"]),
                                         config.get("calibrationFinalLearningRate", config["fineTuneFinalLearningRate"]),
                                         epoch - calibration_start, calibration_epochs))
        if epoch > agreement_start:
            stage_epoch = epoch - agreement_start
            return ("agreement", stage_epoch, agreement_epochs,
                    cosine_learning_rate(config["agreementLearningRate"], config["agreementFinalLearningRate"],
                                         stage_epoch, agreement_epochs))
        return ("polish", epoch, agreement_start,
                cosine_learning_rate(config["learningRate"], config["finalLearningRate"], epoch, agreement_start))
    if (not isinstance(calibration_epochs, int) or not isinstance(agreement_epochs, int) or
            calibration_epochs < 0 or agreement_epochs < 0 or
            calibration_epochs > config["fineTuneEpochs"] or
            config["fineTuneEpochs"] + agreement_epochs > config["epochs"]):
        raise ValueError("invalid agreement/calibration epoch schedule")
    calibration_start = config["epochs"] - calibration_epochs
    agreement_start = calibration_start - agreement_epochs
    if epoch > calibration_start:
        return ("calibration", epoch - calibration_start, calibration_epochs,
                cosine_learning_rate(config.get("calibrationLearningRate", config["fineTuneLearningRate"]),
                                     config.get("calibrationFinalLearningRate", config["fineTuneFinalLearningRate"]),
                                     epoch - calibration_start, calibration_epochs))
    if epoch > agreement_start:
        stage_epoch = epoch - agreement_start
        return ("agreement", stage_epoch, agreement_epochs,
                cosine_learning_rate(config["agreementLearningRate"], config["agreementFinalLearningRate"],
                                     stage_epoch, agreement_epochs))
    fine_tune_epochs = config["fineTuneEpochs"] - calibration_epochs
    pretrain_epochs = agreement_start - fine_tune_epochs
    if epoch <= pretrain_epochs:
        return (
            "pretrain", epoch, pretrain_epochs,
            cosine_learning_rate(config["learningRate"], config["finalLearningRate"], epoch, pretrain_epochs),
        )
    fine_epoch = epoch - pretrain_epochs
    head_epochs = config.get("headTuneEpochs", 0) if config.get("stagedFineTune", False) else 0
    if fine_epoch <= head_epochs:
        return (
            "head-tune", fine_epoch, head_epochs,
            cosine_learning_rate(
                config["fineTuneLearningRate"], config["fineTuneFinalLearningRate"],
                fine_epoch, head_epochs,
            ),
        )
    if config.get("stagedFineTune", False):
        scan_epoch = fine_epoch - head_epochs
        scan_epochs = fine_tune_epochs - head_epochs
        return (
            "fine-tune", scan_epoch, scan_epochs,
            cosine_learning_rate(
                config["scanFineTuneLearningRate"], config["scanFineTuneFinalLearningRate"],
                scan_epoch, scan_epochs,
            ),
        )
    return (
        "fine-tune", fine_epoch, fine_tune_epochs,
        cosine_learning_rate(
            config["fineTuneLearningRate"], config["fineTuneFinalLearningRate"],
            fine_epoch, fine_tune_epochs,
        ),
    )


def assert_migration_parity(initial_metrics: dict | None, expected: float | None) -> None:
    if expected is None:
        return
    actual = initial_metrics["accuracy"] if initial_metrics is not None else None
    if actual is None or abs(actual - expected) > 1e-12:
        raise ValueError(
            f"feature migration changed untouched verification accuracy: "
            f"expected {format_percent(expected)}, got "
            f"{format_percent(actual) if actual is not None else 'no initial score'}"
        )


def training_loss(
    model: QuantizedModel,
    batch: dict,
    class_weights: Tensor,
    auxiliary_positive_weights: Tensor,
    qat: bool,
    auxiliary_weight: float,
    teacher: QuantizedModel | None = None,
    distillation_weight: float = 0,
    distillation_temperature: float = 1,
    teacher_float: bool = False,
    family_factors: Tensor | None = None,
    boundary_class_indices: tuple[int, ...] | None = None,
    boundary_multiplier: float = 1,
    unit_part_weights: bool = False,
) -> Tensor:
    logits, auxiliary_logits = model(batch, quantized=qat, auxiliary=True)
    mask = batch["mask"]
    targets = batch["targets"][mask]
    confidence = batch["supervision_weights"][mask]
    if unit_part_weights:
        confidence = (confidence > 0).to(confidence.dtype)
    # Keep label eligibility/confidence untouched; language importance is an independent factor.
    language_factor = family_factors.expand_as(batch["targets"])[mask] if family_factors is not None else 1
    loss_weights = 1 if unit_part_weights else batch["loss_weights"][mask]
    token_weights = class_weights[targets] * loss_weights * confidence
    if boundary_class_indices and boundary_multiplier > 1:
        supervised_mask = mask & (batch["supervision_weights"] > 0)
        token_weights = token_weights * boundary_loss_factors(
            batch["targets"], supervised_mask, boundary_class_indices, boundary_multiplier
        )[mask]
    student_logits = logits[mask]
    classification = F.cross_entropy(student_logits, targets, reduction="none")
    hard_losses = classification * token_weights * language_factor
    supervised = hard_losses.sum()
    if teacher is not None and distillation_weight > 0:
        with torch.no_grad():
            teacher_logits, _ = teacher(batch, quantized=not teacher_float, auxiliary=False)
            selected_teacher_logits = teacher_logits[mask]
            teacher_probabilities = F.softmax(selected_teacher_logits / distillation_temperature, dim=-1)
            teacher_agrees = (selected_teacher_logits.argmax(dim=-1) == targets).to(hard_losses.dtype)
        divergence = F.kl_div(
            F.log_softmax(student_logits / distillation_temperature, dim=-1),
            teacher_probabilities,
            reduction="none",
        ).sum(dim=-1) * distillation_temperature**2
        # Failure-window weights belong to Shiki's hard target: applying them
        # to the teacher term would reinforce precisely the teacher mistakes
        # those windows were mined to correct.
        distilled = divergence * class_weights[targets] * confidence * language_factor
        blend = distillation_weight * teacher_agrees
        supervised = (hard_losses * (1 - blend) + distilled * blend).sum()
    bits = torch.arange(auxiliary_logits.shape[-1], device=logits.device)
    auxiliary_targets = ((batch["auxiliary"][mask, None] >> bits) & 1).float()
    auxiliary_weights = torch.where(
        auxiliary_targets > 0, auxiliary_positive_weights.unsqueeze(0), torch.ones_like(auxiliary_targets)
    ) * (confidence * language_factor).unsqueeze(1)
    auxiliary = F.binary_cross_entropy_with_logits(
        auxiliary_logits[mask], auxiliary_targets, weight=auxiliary_weights, reduction="sum"
    )
    return supervised + auxiliary_weight * auxiliary


def boundary_loss_factors(
    targets: Tensor,
    supervised_mask: Tensor,
    emphasized_classes: tuple[int, ...],
    multiplier: float = 2,
) -> Tensor:
    """Weight both supervised sides of quote/comment or plain/style edges."""
    factors = torch.ones_like(targets, dtype=torch.float32)
    if targets.shape[1] < 2 or multiplier <= 1 or not emphasized_classes:
        return factors
    emphasized = torch.zeros_like(supervised_mask)
    for class_index in emphasized_classes:
        emphasized |= targets == class_index
    positions = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
    previous = torch.where(supervised_mask, positions, -1).cummax(dim=1).values
    previous = F.pad(previous[:, :-1], (1, 0), value=-1)
    valid_previous = previous >= 0
    previous_targets = targets.gather(1, previous.clamp_min(0))
    previous_emphasized = emphasized.gather(1, previous.clamp_min(0))
    previous_transition = (
        supervised_mask & valid_previous & (targets != previous_targets) &
        (emphasized | previous_emphasized)
    )
    following = torch.where(supervised_mask, positions, targets.shape[1]).flip((1,)).cummin(dim=1).values.flip((1,))
    following = F.pad(following[:, 1:], (0, 1), value=targets.shape[1])
    valid_following = following < targets.shape[1]
    following_targets = targets.gather(1, following.clamp_max(targets.shape[1] - 1))
    following_emphasized = emphasized.gather(1, following.clamp_max(targets.shape[1] - 1))
    following_transition = (
        supervised_mask & valid_following & (targets != following_targets) &
        (emphasized | following_emphasized)
    )
    boundary = previous_transition | following_transition
    return torch.where(boundary, multiplier, 1.0)


def empty_metrics(classes: int) -> dict:
    return {
        "confusion": torch.zeros(classes, classes, dtype=torch.int64),
        "loss": 0.0, "total": 0, "boundaryTp": 0, "boundaryFp": 0, "boundaryFn": 0,
    }


def add_metrics(metrics: dict, expected: Tensor, predicted: Tensor, losses: Tensor, classes: int) -> None:
    metrics["confusion"] += torch.bincount(expected * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    metrics["loss"] += float(losses.sum())
    metrics["total"] += expected.numel()
    if expected.numel() > 1:
        expected_boundaries = expected[1:] != expected[:-1]
        predicted_boundaries = predicted[1:] != predicted[:-1]
        metrics["boundaryTp"] += int((expected_boundaries & predicted_boundaries).sum())
        metrics["boundaryFp"] += int((~expected_boundaries & predicted_boundaries).sum())
        metrics["boundaryFn"] += int((expected_boundaries & ~predicted_boundaries).sum())


def finish_metrics(raw: dict, class_names: list[str], tree_metrics: bool = False) -> dict:
    confusion = raw["confusion"]
    total = raw["total"]
    per_class = {}
    styled_f1 = 0.0
    for target, name in enumerate(class_names):
        true_positive = int(confusion[target, target])
        false_positive = int(confusion[:, target].sum()) - true_positive
        false_negative = int(confusion[target, :].sum()) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-30, precision + recall)
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": true_positive + false_negative,
        }
        if target:
            styled_f1 += f1
    expected_counts = confusion.sum(dim=1)
    majority_index = int(expected_counts.argmax()) if total else 0
    boundary_precision = raw["boundaryTp"] / max(1, raw["boundaryTp"] + raw["boundaryFp"])
    boundary_recall = raw["boundaryTp"] / max(1, raw["boundaryTp"] + raw["boundaryFn"])
    plain_support = int(expected_counts[0])
    plain_correct = int(confusion[0, 0])
    correct = int(confusion.diag().sum())
    diagnostics = {
        "support": total, "errors": total - correct,
        "plainSupport": plain_support, "falseColorCount": plain_support - plain_correct,
        "styledSupport": total - plain_support,
        "styledErrors": total - plain_support - (correct - plain_correct),
        "styledAccuracy": (correct - plain_correct) / max(1, total - plain_support),
    } if tree_metrics else {}
    return {
        **diagnostics,
        "loss": raw["loss"] / max(1, total),
        "accuracy": float(confusion.diag().sum()) / max(1, total),
        "macroF1": styled_f1 / (len(class_names) - 1),
        "majorityBaseline": {
            "class": class_names[majority_index],
            "accuracy": int(expected_counts[majority_index]) / max(1, total),
        },
        "falseColorRate": (plain_support - plain_correct) / max(1, plain_support),
        "boundary": {
            "precision": boundary_precision,
            "recall": boundary_recall,
            "f1": 2 * boundary_precision * boundary_recall / max(1e-30, boundary_precision + boundary_recall),
        },
        "perClass": per_class,
    }


def length_bucket(length: int) -> str:
    if length <= 64:
        return "1-64"
    if length <= 256:
        return "65-256"
    if length <= 1024:
        return "257-1024"
    return "1025+"


@torch.no_grad()
def evaluate(model: QuantizedModel, dataset: SparseDataset, config: dict, device: torch.device) -> dict:
    model.eval()
    classes = len(config["classNames"])
    overall = empty_metrics(classes)
    mixed = empty_metrics(classes)
    embedded = empty_metrics(classes)
    lengths = {name: empty_metrics(classes) for name in ("1-64", "65-256", "257-1024", "1025+")}
    languages: dict[str, dict] = {}
    families: dict[str, dict] = {}
    tree_metrics = True
    refs = [(dataset, index, None) for index in range(len(dataset))]
    for refs_batch in batches(refs, config["batchTokens"]):
        batch = collate(refs_batch, config["inputSize"], device)
        logits, _ = model(batch, quantized=not config.get("teacherMode", False), auxiliary=False)
        for row, metadata in enumerate(batch["metadata"]):
            length = int(batch["lengths"][row].item())
            row_logits = logits[row, :length]
            expected = batch["targets"][row, :length]
            selected = batch["supervision_weights"][row, :length] > 0
            row_logits = row_logits[selected]
            expected = expected[selected]
            predicted = row_logits.argmax(dim=-1)
            losses = F.cross_entropy(row_logits, expected, reduction="none")
            expected_cpu = expected.cpu()
            predicted_cpu = predicted.cpu()
            losses_cpu = losses.cpu()
            add_metrics(overall, expected_cpu, predicted_cpu, losses_cpu, classes)
            add_metrics(lengths[length_bucket(length)], expected_cpu, predicted_cpu, losses_cpu, classes)
            language = metadata.get("language", "unknown")
            if language not in languages:
                languages[language] = empty_metrics(classes)
            add_metrics(languages[language], expected_cpu, predicted_cpu, losses_cpu, classes)
            if tree_metrics:
                family = family_of(metadata)
                if family not in families:
                    families[family] = empty_metrics(classes)
                add_metrics(families[family], expected_cpu, predicted_cpu, losses_cpu, classes)
            if metadata.get("mixedLanguage", False):
                add_metrics(mixed, expected_cpu, predicted_cpu, losses_cpu, classes)
            if metadata.get("embeddedLanguage", False):
                add_metrics(embedded, expected_cpu, predicted_cpu, losses_cpu, classes)
    result = finish_metrics(overall, config["classNames"], tree_metrics)
    result["perLanguage"] = {
        language: finish_metrics(metrics, config["classNames"], tree_metrics) for language, metrics in languages.items()
    }
    result["mixedLanguage"] = finish_metrics(mixed, config["classNames"], tree_metrics)
    result["embeddedLanguage"] = finish_metrics(embedded, config["classNames"], tree_metrics)
    result["byLength"] = {
        name: finish_metrics(metrics, config["classNames"], tree_metrics) for name, metrics in lengths.items()
    }
    if tree_metrics:
        result["perFamily"] = {
            family: finish_metrics(metrics, config["classNames"], True) for family, metrics in families.items()
        }
        weights = normalized_family_weights(config.get("languageObjective", {}), families)
        result["familyWeights"] = weights
        result["languageObjective"] = config["languageObjective"]
        result["weightedError"] = weighted_family_error(result["perFamily"], weights)
        result["objectiveComplete"] = True
        result["missingFamilies"] = []
        result["unknownFamilies"] = []
        for value in result["perFamily"].values():
            value["error"] = value["errors"] / value["support"] if value["support"] else None
            value["falseColors"] = value["falseColorCount"]
            value["falseColorRate"] = value["falseColors"] / value["plainSupport"] if value["plainSupport"] else None
    model.train()
    return result


@torch.no_grad()
def evaluate_required(model: QuantizedModel, dataset: SparseDataset | None, config: dict, device: torch.device) -> dict | None:
    """Track the originally failing snippet parts against deployed quantized weights."""
    if dataset is None:
        return None
    model.eval()
    correct = total = 0
    mismatches = []
    refs = [(dataset, index, None) for index in range(len(dataset))]
    for refs_batch in batches(refs, config["batchTokens"]):
        batch = collate(refs_batch, config["inputSize"], device)
        logits, _ = model(batch, quantized=True, auxiliary=False)
        predicted = logits.argmax(dim=-1)
        for row in range(len(refs_batch)):
            length = int(batch["lengths"][row].item())
            selected = batch["supervision_weights"][row, :length] > 0
            expected = batch["targets"][row, :length]
            actual = predicted[row, :length]
            correct += int(((actual == expected) & selected).sum())
            total += int(selected.sum())
            for part in torch.nonzero(selected & (actual != expected), as_tuple=False).flatten().cpu().tolist():
                mismatches.append({
                    "record": refs_batch[row][1], "part": part,
                    "expected": int(expected[part]), "predicted": int(actual[part]),
                })
    model.train()
    return {"passed": total > 0 and correct == total, "correct": correct, "total": total,
            "accuracy": correct / max(1, total), "mismatches": mismatches}


def checkpoint_score(metrics: dict, selection_metric: str = "weightedError") -> float:
    if selection_metric == "accuracy":
        return metrics["accuracy"]
    if "weightedError" in metrics:
        return -metrics["weightedError"]
    return metrics["accuracy"] + metrics["macroF1"] * 0.25 + metrics["mixedLanguage"]["macroF1"] * 0.05


def candidate(epoch: int, model: QuantizedModel, metrics: dict, source: str = "raw", required: dict | None = None) -> dict:
    return {"epoch": epoch, "source": source, "state": model.state(), "metrics": metrics, "required": required}


def candidate_summary(value: dict, selection_metric: str) -> dict:
    return {
        "epoch": value["epoch"], "source": value.get("source", "raw"),
        "score": checkpoint_score(value["metrics"], selection_metric), "metrics": value["metrics"],
        "required": value.get("required"),
    }


def selection_issues(metrics: dict, baseline: dict | None, config: dict, objective: dict) -> tuple[list[str], list[str]]:
    if config.get("selectionMetric") != "accuracy":
        return promotion_candidate_issues(metrics, baseline, objective)
    issues = protected_family_failures(metrics, baseline, objective)
    failures = [issue for issue in issues if "missing family metrics" in issue or "support mismatch" in issue]
    failures.extend(strict_language_failures(metrics, baseline, objective))
    warnings = [issue for issue in issues if issue not in failures]
    if baseline is not None and not metrics["accuracy"] > baseline["accuracy"]:
        failures.append("verification accuracy did not improve")
    return failures, warnings


def selection_eligible(metrics: dict, baseline: dict | None, config: dict, objective: dict) -> bool:
    return not selection_issues(metrics, baseline, config, objective)[0]


def device_state(model: QuantizedModel) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.weights.items()}


def update_ema(state: dict[str, Tensor] | None, model: QuantizedModel, decay: float) -> dict[str, Tensor]:
    if state is None:
        return device_state(model)
    with torch.no_grad():
        for name, parameter in model.weights.items():
            state[name].mul_(decay).add_(parameter.detach(), alpha=1 - decay)
    return state


def export_state(path: Path, state: dict[str, Tensor], layout: list[dict]) -> None:
    flat = torch.cat([state[tensor["name"]].reshape(-1).float() for tensor in layout])
    values = flat.tolist()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(struct.pack(f"<{len(values)}f", *values))
    temporary.replace(path)


def save_training_result(directory, config, best, best_overall, best_mixed, history, initial_metrics):
    """Export diagnostics even when no candidate qualifies for promotion."""
    selected = best if best is not None else best_overall
    if selected is None or best_mixed is None:
        raise ValueError("no evaluated checkpoint is available to save")
    selection = {"status": "eligible" if best is not None else "diagnostic-only", "failures": []}
    selection["failures"], selection["warnings"] = selection_issues(
        selected["metrics"], config.get("fixedBaselineMetrics") or initial_metrics,
        config, config["languageObjective"],
    )
    if selection["failures"]:
        selection["status"] = "diagnostic-only"
    required = selected.get("required")
    selection["required"] = required
    if best is None and not selection["failures"]:
        selection["failures"] = ["no candidate passed checkpoint selection guards"]
    for name, value in (("selected", selected), ("best-overall", best_overall), ("best-mixed", best_mixed)):
        export_state(directory / f"{name}.f32", value["state"], config["tensorLayout"])
    result = {
        "teacherMode": config.get("teacherMode", False),
        "evaluationMode": "float" if config.get("teacherMode", False) else "quantized",
        "batchTokens": config["batchTokens"], "history": history,
        "initialMetrics": initial_metrics, "selection": selection,
        "selected": candidate_summary(selected, config.get("selectionMetric", "weightedError")),
        "bestOverall": candidate_summary(best_overall, config.get("selectionMetric", "weightedError")),
        "bestMixed": candidate_summary(best_mixed, config.get("selectionMetric", "weightedError")),
    }
    result.update(languageObjective=config["languageObjective"], selectionMetric=config.get("selectionMetric", "weightedError"),
                  classWeightPower=config.get("classWeightPower", .5),
                  calibrationEpochs=config.get("calibrationEpochs", 0))
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(directory / "result.json")
    return selection


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = args.directory
    config = json.loads((directory / "config.json").read_text())
    if config.get("model") != "hierarchical-tree":
        raise ValueError("only the hierarchical-tree trainer is supported")
    device = torch.device(config["device"])
    torch.manual_seed(config.get("seed", 1337))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.get("seed", 1337))

    flat = torch.frombuffer(bytearray((directory / "initial.f32").read_bytes()), dtype=torch.float32).clone()
    model = HierarchicalTree(
        config["tensorLayout"], flat, config.get("weightBits", 6),
        tree_context=config.get("treeContext"), local_radius=config.get("localRadius", 2),
    ).to(device)
    evaluation_mode = "float teacher diagnostic" if config.get("teacherMode", False) else "deployed"
    if config.get("teacherMode", False):
        log("offline float teacher diagnostic: QAT disabled; all checkpoint selection/evaluation uses float weights")
    teacher = None
    if config.get("hasTeacher", False):
        teacher_flat = torch.frombuffer(
            bytearray((directory / "teacher.f32").read_bytes()), dtype=torch.float32
        ).clone()
        if config.get("teacherModel") != "hierarchical-tree":
            raise ValueError("only a hierarchical-tree distillation teacher is supported")
        teacher = HierarchicalTree(
            config["teacherTensorLayout"], teacher_flat, config.get("teacherWeightBits", 6),
            tree_context=config.get("teacherTreeContext"), local_radius=2,
        ).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        log(
            f"loaded frozen {'float' if config.get('teacherFloat', False) else 'int' + str(config.get('teacherWeightBits', 6))} "
            f"{config['teacherModel']} distillation teacher; "
            f"weight={format_percent(config['distillationWeight'])} "
            f"temperature={config['distillationTemperature']}"
        )
    train = SparseDataset(directory, "train")
    fine = SparseDataset(directory, "fine") if config.get("hasFineDataset", config.get("hasPretrain", False)) else train
    tree_objective = True
    objective = config.get("languageObjective", {})
    # Resolve defaults once, independently of each evaluation's observed families.
    weights = normalized_family_weights(objective, [family_of(metadata) for metadata in train.metadata + fine.metadata])
    objective = {**objective, "familyWeights": weights}
    config["languageObjective"] = objective
    if args.check:
        refs = [(train, index, None) for index in range(len(train))]
        batch = collate(refs, config["inputSize"], device)
        with torch.no_grad():
            logits, _ = model(batch, quantized=False, auxiliary=False)
            probabilities = []
            for row in range(len(refs)):
                length = int(batch["lengths"][row].item())
                probabilities.append(torch.softmax(logits[row, :length], dim=-1).cpu().tolist())
        class_weights = torch.tensor(config["classWeights"], dtype=torch.float32, device=device)
        if tree_objective and "classWeightPower" in config:
            class_weights = class_weights_for_counts(
                reference_counts(refs, len(config["classNames"]))[1], config["classWeightPower"]
            ).to(device)
        auxiliary_positive_weights = torch.tensor(
            config.get("auxiliaryPositiveWeights", [1] * model.weights["auxiliaryBias"].numel()),
            dtype=torch.float32,
            device=device,
        )
        loss = training_loss(
            model, batch, class_weights, auxiliary_positive_weights, False, config["auxiliaryLossWeight"],
            teacher, config.get("distillationWeight", 0), config.get("distillationTemperature", 1),
            teacher_float=config.get("teacherFloat", False),
            family_factors=batch_family_factors(batch, family_importance(reference_counts(refs)[0], weights))
            if tree_objective else None,
        )
        loss.backward()
        gradients = {
            name: parameter.grad.detach().cpu() if parameter.grad is not None else torch.zeros_like(parameter).cpu()
            for name, parameter in model.weights.items()
        }
        export_state(directory / "gradients.f32", gradients, config["tensorLayout"])
        (directory / "check.json").write_text(json.dumps({
            "loss": float(loss.detach().cpu()), "probabilities": probabilities,
        }, separators=(",", ":")) + "\n")
        return
    verification = SparseDataset(directory, "verification")
    hard = SparseDataset(directory, "hard") if config["hasHardRecords"] else None
    required = SparseDataset(directory, "required") if config.get("hasRequiredRecords", False) else None
    pretrain_class_weights = torch.tensor(config["pretrainClassWeights"], dtype=torch.float32, device=device)
    fine_class_weights = torch.tensor(config["fineTuneClassWeights"], dtype=torch.float32, device=device)
    pretrain_auxiliary_positive_weights = torch.tensor(
        config["pretrainAuxiliaryPositiveWeights"], dtype=torch.float32, device=device
    )
    fine_auxiliary_positive_weights = torch.tensor(
        config["fineTuneAuxiliaryPositiveWeights"], dtype=torch.float32, device=device
    )
    if not config["batchTokens"]:
        config["batchTokens"] = benchmark_batch_tokens(
            model, train, config, device, pretrain_class_weights, pretrain_auxiliary_positive_weights, teacher
        )
    if config.get("focusedReplay", False):
        steps = config.get("focusedReplaySteps", 1)
        if hard is None or not isinstance(steps, int) or steps < 1 or steps > 4:
            raise ValueError("focused replay requires hard records and 1-4 steps")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], eps=1e-8)
    focused_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get("focusedReplayLearningRate", .001), eps=1e-8
    ) if config.get("focusedReplay", False) else None
    rng = random.Random(config["seed"])
    replay_epochs = [
        epoch for epoch in range(1, config["epochs"] + 1)
        if replay_enabled_stage(training_stage(config, epoch)[0])
    ]
    replay_epoch_indices = {epoch: index for index, epoch in enumerate(replay_epochs)}
    hard_partitions = partition_hard(hard, len(replay_epochs), rng) if replay_epochs else None
    boundary_classes = tuple(
        config["classNames"].index(name) for name in ("plain", "string", "comment")
        if name in config["classNames"]
    )

    measured_initial_metrics = evaluate(model, verification, config, device) if config["hasInitial"] else None
    migration_expected = config.get("migrationExpectedAccuracy")
    assert_migration_parity(measured_initial_metrics, migration_expected)
    if migration_expected is not None:
        log(f"feature-v2 to v3 migration parity verified at {format_percent(measured_initial_metrics['accuracy'])} accuracy")
    initial_metrics = (
        config["fixedBaselineMetrics"]
        if measured_initial_metrics is not None and config.get("initialIsBaseline", False)
        else measured_initial_metrics
    )
    if measured_initial_metrics is not None and config.get("initialIsBaseline", False):
        log(f"epoch 0 anchored to exact deployed baseline at {format_percent(initial_metrics['accuracy'])} accuracy")
    initial_required = evaluate_required(model, required, config, device) if initial_metrics else None
    initial_candidate = candidate(0, model, initial_metrics, "initial", initial_required) if initial_metrics else None
    best = None if config.get("initialIsBaseline", False) else initial_candidate
    if initial_candidate and config.get("fixedBaselineMetrics") is not None and not selection_eligible(
        initial_metrics, config["fixedBaselineMetrics"], config, objective
    ):
        best = None
    best_overall = initial_candidate
    best_mixed = best
    if initial_metrics:
        log(
            f"initial {evaluation_mode} checkpoint verified "
            f"accuracy={format_percent(initial_metrics['accuracy'])} "
            f"styled-macro-f1={format_percent(initial_metrics['macroF1'])} "
            f"mixed-f1={format_percent(initial_metrics['mixedLanguage']['macroF1'])}"
        )
        if initial_required is not None:
            log(
                f"initial target snippet {initial_required['correct']}/{initial_required['total']} parts correct "
                f"({'pass' if initial_required['passed'] else 'fail'})"
            )

    history = []
    stale_epochs = 0
    ema_state = None
    previous_stage = None
    for epoch in range(1, config["epochs"] + 1):
        stage, stage_epoch, stage_epochs, learning_rate = training_stage(config, epoch)
        if stage_epoch == 1 and optimizer_reset_required(previous_stage, stage):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        qat = quantization_aware_epoch(config, epoch)
        distillation_progress = min(1, max(0, epoch - config.get("distillationWarmupEpochs", 0)) /
                                   max(1, config.get("distillationRampEpochs", 1)))
        epoch_distillation_weight = 0 if stage in ("agreement", "calibration") else \
            config.get("distillationWeight", 0) * distillation_progress
        if stage_epoch == 1 or (qat and epoch == config["epochs"] - config["qatEpochs"] + 1):
            stale_epochs = 0
        # Calibration observes the original corpus distribution. The fine set
        # deliberately balances and duplicates families, which is useful for
        # representation learning but wrong for probability calibration.
        active = train if stage in ("pretrain", "polish", "agreement", "calibration") else fine
        class_weights = pretrain_class_weights if stage in ("pretrain", "polish", "agreement") else fine_class_weights
        auxiliary_positive_weights = (
            pretrain_auxiliary_positive_weights if stage in ("pretrain", "polish") else fine_auxiliary_positive_weights
        )
        base_refs = shuffled(active, rng) if stage in ("pretrain", "polish", "agreement", "calibration") else interleave(active, rng)
        hard_indices = None
        focused_replay = bool(config.get("focusedReplay", False) and
                              hard is not None and replay_enabled_stage(stage))
        if hard is not None and replay_enabled_stage(stage) and not focused_replay:
            hard_indices = hard_order(hard, rng) if config.get("repeatHardReplay", False) \
                else hard_partitions[replay_epoch_indices[epoch]]
        refs = base_refs if focused_replay else mix_replay(
            base_refs, hard, hard_indices, config["replayFraction"], config.get("repeatHardReplay", False),
            config.get("maxHardReplayRepeats"),
        )
        importance = None
        family_support = None
        if tree_objective:
            family_support, class_counts = reference_counts(refs, len(config["classNames"]))
            # Accuracy-selected runs optimize the natural part distribution;
            # per-language curriculum weights below replace popularity-family
            # factors. Weighted-error experiments retain the older objective.
            if (stage not in ("polish", "agreement", "calibration") and
                    config.get("selectionMetric") != "accuracy"):
                importance = family_importance(family_support, weights)
            if stage in ("polish", "agreement", "calibration") or "classWeightPower" in config:
                class_weights = class_weights_for_counts(
                    class_counts, 0 if stage in ("polish", "agreement", "calibration") else config["classWeightPower"]
                ).to(device)
        token_target = sum(min(dataset.length(index), limit or 2**63) for dataset, index, limit in refs)
        replay_tokens = sum(ref_length(ref) for ref in refs if hard is not None and ref[0] is hard)
        started = time.perf_counter()
        processed = 0
        epoch_loss = torch.zeros((), device=device)
        focused_stats = None
        if focused_replay:
            focused_stats = {"parts": 0, "labels": 0, "loss": 0., "symbolsChanged": 0}
            steps = config.get("focusedReplaySteps", 1)
            focused_epoch = replay_epoch_indices[epoch] + 1
            focused_learning_rate = cosine_learning_rate(
                config.get("focusedReplayLearningRate", .001),
                config.get("focusedReplayFinalLearningRate", .0005),
                focused_epoch, len(replay_epochs),
            )
            for group in focused_optimizer.param_groups:
                group["lr"] = focused_learning_rate
            for _ in range(steps):
                step = focused_replay_step(
                    model, focused_optimizer, hard, config, stage, qat, class_weights,
                    auxiliary_positive_weights, teacher, epoch_distillation_weight, rng,
                )
                focused_stats["parts"] += step["parts"]
                focused_stats["labels"] += step["labels"]
                focused_stats["loss"] += step["loss"]
                focused_stats["symbolsChanged"] += step["symbolsChanged"]
            focused_stats["loss"] /= steps
            focused_stats["learningRate"] = focused_learning_rate
            focused_stats["required"] = evaluate_required(model, required, config, device)
            replay_tokens = focused_stats["parts"]
            log(
                f"epoch {epoch}/{config['epochs']} {stage} balanced failure update "
                f"labels={focused_stats['labels']:,} parts={focused_stats['parts']:,} "
                f"loss={focused_stats['loss']:.4f} lr {focused_learning_rate:.2e} "
                f"symbols={focused_stats['symbolsChanged']:,}"
                f"{' target=' + str(focused_stats['required']['correct']) + '/' + str(focused_stats['required']['total']) if focused_stats['required'] else ''}"
            )
        batch_number = 0
        for refs_batch in batches(refs, config["batchTokens"], rng):
            batch_number += 1
            batch = collate(refs_batch, config["inputSize"], device)
            training_factors = batch_family_factors(batch, importance) if importance is not None else None
            if stage not in ("agreement", "calibration") and config.get("trainingLanguageMultipliers"):
                language_factors = batch_language_factors(batch, config["trainingLanguageMultipliers"])
                training_factors = language_factors if training_factors is None else training_factors * language_factors
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                model, batch, class_weights, auxiliary_positive_weights, qat,
                0 if stage == "agreement" else config["auxiliaryLossWeight"],
                teacher, epoch_distillation_weight, config.get("distillationTemperature", 1),
                teacher_float=config.get("teacherFloat", False),
                family_factors=training_factors,
                boundary_class_indices=boundary_classes,
                boundary_multiplier=1 if stage in ("agreement", "calibration") else config.get("boundaryLossMultiplier", 1),
                unit_part_weights=stage == "agreement",
            )
            loss.backward()
            retain_trainable_gradients(model, config, stage)
            normalize_and_clip_gradients(model, batch["supervised_tokens"], config["gradientClip"])
            optimizer.step()
            if epoch >= config["emaStartEpoch"]:
                ema_state = update_ema(ema_state, model, config["emaDecay"])
            processed += batch["tokens"]
            epoch_loss += loss.detach()
            if batch_number == 1 or batch_number % config["progressEvery"] == 0 or processed == token_target:
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 0.001)
                mean_loss = float(epoch_loss.cpu()) / max(1, processed)
                log(
                    f"epoch {epoch}/{config['epochs']} {stage} {stage_epoch}/{stage_epochs} batch {batch_number} "
                    f"tokens {processed:,}/{token_target:,} ({format_percent(processed / token_target)}) "
                    f"loss {mean_loss:.4f} lr {learning_rate:.2e}"
                    f"{' distill=' + format_percent(epoch_distillation_weight) if teacher is not None else ''}"
                    f"{' qat-int' + str(config.get('weightBits', 6)) if qat else ''} rate {round(rate):,}/s "
                    f"eta {format_duration((token_target - processed) / max(rate, 1e-9))}"
                )

        metrics = evaluate(model, verification, config, device)
        required_metrics = evaluate_required(model, required, config, device)
        current = candidate(epoch, model, metrics, required=required_metrics)
        epoch_candidates = [current]
        ema_metrics = None
        if ema_state is not None:
            current_state = device_state(model)
            model.load_state(ema_state)
            ema_metrics = evaluate(model, verification, config, device)
            ema_required = evaluate_required(model, required, config, device)
            epoch_candidates.append(candidate(epoch, model, ema_metrics, "ema", ema_required))
            model.load_state(current_state)
        seconds = time.perf_counter() - started
        epoch_result = {
            "epoch": epoch,
            "trainLoss": float(epoch_loss.cpu()) / max(1, processed),
            "verificationLoss": metrics["loss"],
            "verificationAccuracy": metrics["accuracy"],
            "verificationMacroF1": metrics["macroF1"],
            "verificationMixedAccuracy": metrics["mixedLanguage"]["accuracy"],
            "verificationMixedMacroF1": metrics["mixedLanguage"]["macroF1"],
            "verificationPerLanguage": {
                language: {"accuracy": value["accuracy"], "macroF1": value["macroF1"]}
                for language, value in metrics["perLanguage"].items()
            },
            "learningRate": learning_rate,
            "stage": stage,
            "trainingTokens": token_target,
            "replayTokens": replay_tokens,
            "replayFraction": replay_tokens / max(1, token_target + replay_tokens),
            "focusedReplay": focused_stats,
            "seconds": seconds,
            "required": required_metrics,
        }
        if tree_objective:
            epoch_result["verificationFalseColorRate"] = metrics["falseColorRate"]
            epoch_result["verificationWeightedError"] = metrics["weightedError"]
            epoch_result["verificationPerFamily"] = metrics["perFamily"]
            epoch_result["familySupport"] = family_support
            epoch_result["familyImportance"] = importance
        if ema_metrics is not None:
            if tree_objective:
                epoch_result["emaVerificationWeightedError"] = ema_metrics["weightedError"]
                epoch_result["emaVerificationPerFamily"] = ema_metrics["perFamily"]
            epoch_result["emaVerificationLoss"] = ema_metrics["loss"]
            epoch_result["emaVerificationAccuracy"] = ema_metrics["accuracy"]
            epoch_result["emaVerificationMacroF1"] = ema_metrics["macroF1"]
            epoch_result["emaVerificationMixedMacroF1"] = ema_metrics["mixedLanguage"]["macroF1"]
        history.append(epoch_result)
        previous_stage = stage
        for value in epoch_candidates:
            if best_overall is None or checkpoint_score(
                value["metrics"], config.get("selectionMetric", "weightedError")
            ) > checkpoint_score(best_overall["metrics"], config.get("selectionMetric", "weightedError")):
                best_overall = value
            if best_mixed is None or value["metrics"]["mixedLanguage"]["macroF1"] > best_mixed["metrics"]["mixedLanguage"]["macroF1"]:
                best_mixed = value
        mixed_floor = (
            initial_metrics["mixedLanguage"]["macroF1"] - config["mixedF1Tolerance"]
            if initial_metrics else -math.inf
        )
        def eligible_candidate(value):
            if tree_objective:
                return selection_eligible(
                    value["metrics"], config.get("fixedBaselineMetrics") or initial_metrics, config, objective
                )
            return value["metrics"]["mixedLanguage"]["macroF1"] >= mixed_floor

        eligible = max(
            (value for value in epoch_candidates if eligible_candidate(value)),
            key=lambda value: checkpoint_score(value["metrics"], config.get("selectionMetric", "weightedError")),
            default=None,
        )
        current_failures = selection_issues(
            metrics, config.get("fixedBaselineMetrics") or initial_metrics, config, objective
        )[0] if tree_objective else []
        epoch_result["selectionFailures"] = current_failures
        improved = (
            eligible is not None and
            (best is None or checkpoint_score(
                eligible["metrics"], config.get("selectionMetric", "weightedError")
            ) > checkpoint_score(best["metrics"], config.get("selectionMetric", "weightedError")) + config["minDelta"])
        )
        if improved:
            best = eligible
            stale_epochs = 0
            suffix = f"best checkpoint: {best['source']}"
        else:
            stale_epochs += 1
            suffix = (f"guarded: {'; '.join(current_failures)}"
                      if eligible is None and current_failures
                      else f"no improvement {stale_epochs}/{config['patience']}")
        log(
            f"epoch {epoch}/{config['epochs']} {evaluation_mode} loss={metrics['loss']:.4f} "
            f"accuracy={format_percent(metrics['accuracy'])} "
            f"styled-macro-f1={format_percent(metrics['macroF1'])} "
            f"mixed-f1={format_percent(metrics['mixedLanguage']['macroF1'])}"
            f"{' weighted-error=' + format_percent(metrics['weightedError']) if tree_objective else ''}"
            f"{' ema=' + format_percent(ema_metrics['accuracy']) + '/' + format_percent(ema_metrics['macroF1']) if ema_metrics else ''}"
            f"{' target=' + str(required_metrics['correct']) + '/' + str(required_metrics['total']) if required_metrics else ''} "
            f"[{suffix}]"
        )
        saved_selection = save_training_result(
            directory, config, best, best_overall, best_mixed, history, initial_metrics,
        )
        if saved_selection["status"] == "diagnostic-only":
            log("saved diagnostic checkpoint; selection guards: " + "; ".join(saved_selection["failures"]))
        calibration_pending = tree_objective and config.get("calibrationEpochs", 0) > 0 and stage != "calibration"
        if best is not None and stale_epochs >= config["patience"] and not calibration_pending and (
            tree_objective or config.get("teacherMode", False) or qat
        ):
            log(f"early stopping at epoch {epoch}; restoring best checkpoint from epoch {best['epoch']}")
            break

    save_training_result(directory, config, best, best_overall, best_mixed, history, initial_metrics)
    if best is None:
        log("no checkpoint passed selection guards; exported best weighted-error candidate for diagnostics only")


if __name__ == "__main__":
    main()

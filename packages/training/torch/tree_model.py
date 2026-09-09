"""Tree-only language objective helpers (no model/training side effects)."""

from __future__ import annotations

import math
from collections import defaultdict

import torch


def family_of(metadata: dict) -> str:
    return metadata.get("family") or metadata.get("language") or "unknown"


def normalized_family_weights(objective: dict, families=()) -> dict[str, float]:
    configured = objective.get("familyWeights")
    if configured is None:
        raise ValueError("languageObjective.familyWeights is required; no observed-family fallback")
    weights = dict(configured)
    if not weights or any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
                          or not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("languageObjective.weights must contain finite nonnegative family weights")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("languageObjective.weights must have positive total weight")
    return {family: weight / total for family, weight in weights.items()}


def weighted_family_error(per_family: dict, weights: dict[str, float]) -> float:
    unknown = set(per_family) - set(weights)
    if unknown:
        raise ValueError("unknown evaluation families: " + ", ".join(sorted(unknown)))
    missing = [family for family, weight in weights.items()
               if weight > 0 and per_family.get(family, {}).get("support", 0) <= 0]
    if missing:
        raise ValueError("language objective has no supervised support for: " + ", ".join(sorted(missing)))
    return sum(weight * (1 - per_family[family]["accuracy"])
               for family, weight in weights.items() if weight > 0)


def reference_counts(refs, classes: int = 0) -> tuple[dict[str, int], torch.Tensor]:
    """Count positions, not confidence mass; include duplicates and truncation."""
    families = defaultdict(int)
    counts = torch.zeros(classes, dtype=torch.float64)
    cache = {}
    for dataset, index, limit in refs:
        key = (id(dataset), index, limit)
        if key not in cache:
            # SparseDataset supports cheap target/mask slicing without materializing features.
            if hasattr(dataset, "record_offsets"):
                start = int(dataset.record_offsets[index])
                end = int(dataset.record_offsets[index + 1])
                if limit is not None:
                    end = min(end, start + limit)
                metadata = dataset.metadata[index]
                supervised = dataset.supervision_weights[start:end] > 0
                targets = dataset.targets[start:end]
            else:
                record = dataset.record(index, limit)
                metadata = record["metadata"]
                supervised = record["supervision_weights"] > 0
                targets = record["targets"]
            class_counts = torch.bincount(targets[supervised].long(), minlength=classes) if classes else counts
            cache[key] = family_of(metadata), int(supervised.sum()), class_counts
        family, support, class_counts = cache[key]
        families[family] += support
        if classes:
            counts += class_counts
    return dict(families), counts


def family_importance(support: dict[str, int], weights: dict[str, float]) -> dict[str, float]:
    total = sum(support.values())
    unknown = set(support) - set(weights)
    if unknown:
        raise ValueError("unknown training families: " + ", ".join(sorted(unknown)))
    missing = [family for family, weight in weights.items() if weight > 0 and support.get(family, 0) <= 0]
    if missing:
        raise ValueError("training stage has no supervised support for: " + ", ".join(sorted(missing)))
    return {family: weights.get(family, 0.0) * total / count if count else 0.0
            for family, count in support.items()}


def batch_family_factors(batch: dict, importance: dict[str, float]) -> torch.Tensor:
    return torch.tensor([importance.get(family_of(metadata), 0.0) for metadata in batch["metadata"]],
                        dtype=batch["supervision_weights"].dtype, device=batch["mask"].device).unsqueeze(1)


def class_weights_for_counts(counts: torch.Tensor, power: float) -> torch.Tensor:
    if not math.isfinite(power) or power < 0:
        raise ValueError("classWeightPower must be finite and nonnegative")
    if power == 0:
        return torch.ones_like(counts, dtype=torch.float32)
    values = (counts.sum() / (len(counts) * counts.clamp_min(1))).pow(power).clamp_max(4)
    return (values / values.mean().clamp_min(1e-30)).float()


def protected_family_failures(metrics: dict, initial: dict | None, objective: dict) -> list[str]:
    """Report every fixed-reference guard failure, not just a boolean veto."""
    failures = []
    if initial is None:
        return failures
    for family in objective.get("protectedFamilies", []):
        before = initial.get("perFamily", {}).get(family)
        after = metrics.get("perFamily", {}).get(family)
        if not before or not after:
            failures.append(f"{family}: missing family metrics")
            continue
        for metric, denominator, numerator, minimum, allowance in (
            ("error", "support", "errors", "minSupport", "maxErrorIncrease"),
            ("falseColorRate", "plainSupport", "falseColors", "minPlainSupport", "maxFalseColorIncrease"),
        ):
            n = after[denominator]
            required = objective.get(minimum, 100)
            if n != before[denominator]:
                failures.append(f"{family} {metric}: support mismatch ({n} vs {before[denominator]})")
                continue
            if n < required:
                failures.append(f"{family} {metric}: insufficient support ({n} < {required})")
                continue
            pc = (after[numerator] + 1) / (n + 2)
            pb = (before[numerator] + 1) / (n + 2)
            tolerance = max(objective.get(allowance, .01), objective.get("z", 1.96) *
                            math.sqrt((pc * (1 - pc) + pb * (1 - pb)) / n))
            delta = (after[numerator] - before[numerator]) / n
            if delta > tolerance:
                failures.append(f"{family} {metric}: regression (delta={delta:.6g}, tolerance={tolerance:.6g})")
    return failures


def protected_families_eligible(metrics: dict, initial: dict | None, objective: dict) -> bool:
    return not protected_family_failures(metrics, initial, objective)


def strict_language_failures(metrics: dict, initial: dict | None, objective: dict) -> list[str]:
    """Apply absolute, non-statistical regression caps to product-critical languages."""
    failures = []
    if initial is None:
        return failures
    tiers = (
        (objective.get("strictLanguages", []),
         objective.get("maxStrictErrorIncrease", .002), objective.get("maxStrictFalseColorIncrease", .002)),
        (objective.get("matureLanguages", []),
         objective.get("maxMatureErrorIncrease", .01), objective.get("maxMatureFalseColorIncrease", .01)),
    )
    for languages, error_allowance, false_color_allowance in tiers:
      for language in languages:
        before = initial.get("perLanguage", {}).get(language)
        after = metrics.get("perLanguage", {}).get(language)
        if not before or not after:
            failures.append(f"{language}: missing language metrics")
            continue
        for metric, denominator, numerator, minimum, tolerance in (
            ("error", "support", "errors", "minSupport", error_allowance),
            ("falseColorRate", "plainSupport", "falseColorCount", "minPlainSupport", false_color_allowance),
        ):
            n = after[denominator]
            required = objective.get(minimum, 100)
            if n != before[denominator]:
                failures.append(f"{language} {metric}: support mismatch ({n} vs {before[denominator]})")
            elif n < required:
                failures.append(f"{language} {metric}: insufficient support ({n} < {required})")
            else:
                after_count = after.get(numerator, after.get("falseColors"))
                before_count = before.get(numerator, before.get("falseColors"))
                if after_count is None or before_count is None:
                    failures.append(f"{language} {metric}: missing language metrics")
                    continue
                delta = (after_count - before_count) / n
                if delta > tolerance:
                    failures.append(
                        f"{language} {metric}: regression (delta={delta:.6g}, tolerance={tolerance:.6g})"
                    )
    return failures


def promotion_candidate_issues(metrics: dict, initial: dict | None, objective: dict) -> tuple[list[str], list[str]]:
    """Keep coverage failures fatal while reporting per-family regressions as warnings."""
    issues = protected_family_failures(metrics, initial, objective)
    failures = [issue for issue in issues if "missing family metrics" in issue or "support mismatch" in issue]
    failures.extend(strict_language_failures(metrics, initial, objective))
    warnings = [issue for issue in issues if issue not in failures]
    if initial is not None:
        improvement = initial["weightedError"] - metrics["weightedError"]
        if not improvement > objective.get("minWeightedImprovement", 0):
            failures.append("weighted error did not improve")
    return failures, warnings


def promotion_candidate_eligible(metrics: dict, initial: dict | None, objective: dict) -> bool:
    return not promotion_candidate_issues(metrics, initial, objective)[0]

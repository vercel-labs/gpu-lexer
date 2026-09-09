# Training and promotion policy

The current trainer is tree-only and uses the tracked promoted checkpoint as epoch zero. `pnpm train` and `pnpm train:tree` are equivalent. Use `--fresh` only for an intentional random baseline.

The language objective is built from the pinned popularity snapshot plus explicit supplemental web-language weights. Natural-distribution agreement selects checkpoints. Weak, unguarded languages can receive gentle training multipliers, while strict and mature languages have direct per-language regression caps. Languages with enough support and less than 10% error can graduate into the mature guard set.

Training combines natural rehearsal, boundary-weighted loss, auxiliary lexical-state supervision, quantization-aware epochs, optional teacher/active-model distillation, and final natural-frequency classifier calibration. Independent mining contributes a small error-ranked replay fraction for weak families. Verification is never replayed into training.

Targeted correction uses:

```sh
pnpm fine-tune -- --file ./failure.tsx --lang tsx
```

The snippet is labeled by Shiki, added to an ignored local failure bank, and trained with balanced failure-only updates followed by ordinary rehearsal. The exact promoted int6 model remains epoch zero. A snippet match alone is not sufficient for promotion.

Every run writes float weights, packed weights, metadata, and history under ignored `packages/training/runs/<timestamp>/`. A candidate is eligible only when its deployed quantized accuracy strictly improves the fixed promoted baseline on the same direct-label verification shard and every strict/mature language guard passes. Eligible runs auto-promote; manual review can invoke `pnpm model:promote <run-id-or-path>`.

Promotion regenerates the browser model, minified shader, website statistics, and the compact tracked checkpoint under `packages/training/active/`. Offline float teachers are never promotable.

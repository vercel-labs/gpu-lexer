# Training and promotion policy

The current trainer is tree-only and uses the tracked promoted checkpoint as epoch zero. `pnpm train` and `pnpm train:tree` are equivalent. Use `--fresh` only for an intentional random baseline.

Ordinary continuations use the short polish schedule only when the checkpoint records the current label taxonomy version. After a taxonomy change, or for older checkpoints without a recorded version, compatible weights start the full 32-epoch schedule: learning rate `0.002` initially, with quantization-aware training in the final six epochs. Runs record the target taxonomy in their configuration and the selected checkpoint's taxonomy in model metadata; selecting epoch zero preserves its original version.

The language objective is built from the pinned popularity snapshot plus explicit supplemental web-language weights. Natural-distribution agreement selects checkpoints. Weak, unguarded languages can receive gentle training multipliers, while strict and mature languages have direct per-language regression caps for promotion. Languages with enough support and less than 10% error can graduate into the mature guard set. Automatic membership carries forward only within the same taxonomy; after a taxonomy change it is rebuilt from the baseline evaluated on the new labels. Explicitly configured mature languages remain guarded.

Training combines natural rehearsal, boundary-weighted loss, auxiliary lexical-state supervision, quantization-aware epochs, optional teacher/active-model distillation, and final natural-frequency classifier calibration. Independent mining contributes a small error-ranked replay fraction for weak families. Verification is never replayed into training.

Targeted correction uses:

```sh
pnpm fine-tune -- --file ./failure.tsx --lang tsx
```

The snippet is labeled by Shiki, added to an ignored local failure bank, and trained with balanced failure-only updates followed by ordinary rehearsal. The exact promoted int6 model remains epoch zero. A snippet match alone is not sufficient for promotion.

Every run writes float weights, packed weights, metadata, and history under ignored `packages/training/runs/<timestamp>/`. Training saves the checkpoint with the highest held-out quantized accuracy across epoch zero and every raw/EMA candidate, retaining the earlier checkpoint on ties. Explicit weighted-error experiments use their configured score instead. Language guards do not filter checkpoint selection or reset the progress counter; a best checkpoint that fails them is still saved with diagnostic-only status.

The promoted baseline is evaluated on the run's current verification labels and stays fixed throughout the run. A selected checkpoint is eligible for promotion only when its deployed quantized accuracy strictly improves that baseline and every strict/mature language guard passes. Eligible runs auto-promote; manual review can invoke `pnpm model:promote <run-id-or-path>`.

Promotion regenerates the browser model, minified shader, website statistics, and the compact tracked checkpoint under `packages/training/active/`. Offline float teachers are never promotable.

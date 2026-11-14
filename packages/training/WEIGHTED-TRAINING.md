# Training and promotion policy

The current trainer is tree-only and uses the tracked promoted checkpoint as epoch zero. `pnpm train` and `pnpm train:tree` are equivalent. Use `--fresh` only for an intentional random baseline.

A checkpoint that records the current label taxonomy version continues with 48 full-model agreement epochs: unit part weights on the natural corpus, int6 quantization-aware training throughout, 65,536-part batches, and a cosine learning rate from `0.0064` to `0.00001`. Every epoch evaluates the raw weights and an exponential average of the latent weights (decay `0.99`), and either can be selected. After a taxonomy change, or for older checkpoints without a recorded version, compatible weights start the full 32-epoch schedule: learning rate `0.002` initially, with quantization-aware training in the final six epochs. Runs record the target taxonomy in their configuration and the selected checkpoint's taxonomy in model metadata; selecting epoch zero preserves its original version.

The language objective is built from the pinned popularity snapshot plus explicit supplemental web-language weights. Natural-distribution agreement selects checkpoints. Weak, unguarded languages can receive gentle training multipliers, while strict and mature languages have direct per-language regression caps for promotion. Languages with enough support and less than 10% error can graduate into the mature guard set. Automatic membership carries forward only within the same taxonomy; after a taxonomy change it is rebuilt from the baseline evaluated on the new labels. Explicitly configured mature languages remain guarded.

Training combines natural rehearsal, boundary-weighted loss, auxiliary lexical-state supervision, quantization-aware epochs, optional teacher/active-model distillation, and final natural-frequency classifier calibration. Independent mining contributes a small error-ranked replay fraction for weak families. Verification is never replayed into training.

Continuations skip replay, distillation, and calibration by default. They return when the schedule leaves epochs before agreement, for example `--epochs 10 --agreement-epochs 6 --calibration-epochs 2`; those replay-polish epochs use `--lr` and `--final-lr`.

Targeted correction uses:

```sh
pnpm fine-tune -- --file ./failure.tsx --lang tsx
```

The snippet is labeled by Shiki, added to an ignored local failure bank, and trained with balanced failure-only updates followed by ordinary rehearsal. The exact promoted int6 model remains epoch zero. A snippet match alone is not sufficient for promotion.

Every run writes float weights, packed weights, metadata, and history under ignored `packages/training/runs/<timestamp>/`. Training saves the checkpoint with the highest held-out quantized accuracy across epoch zero and every raw/EMA candidate, retaining the earlier checkpoint on ties. Explicit weighted-error experiments use their configured score instead. Language guards do not filter checkpoint selection or reset the progress counter; a best checkpoint that fails them is still saved with diagnostic-only status.

The promoted baseline is evaluated on the run's current verification labels and stays fixed throughout the run. A selected checkpoint is eligible for promotion only when its deployed quantized accuracy strictly improves that baseline and every strict/mature language guard passes. Eligible runs auto-promote; manual review can invoke `pnpm model:promote <run-id-or-path>`.

Promotion regenerates the browser model, minified shader, website statistics, and the compact tracked checkpoint under `packages/training/active/`. Offline float teachers are never promotable.

## Continuation settings

A deployed int6 code changes only when its latent weight crosses a rounding threshold, so a continuation needs a rate and a step count that can cross one. The earlier defaults used rate `0.000001` with 131,072- or 262,144-part batches, which is 49 or 31 optimizer steps per epoch. That rate is fifty times below the lowest one in the September comparison in `MODEL_CARD.md`, which gained 0.11 points in 24 epochs.

Each row is one run from checkpoint `20260916T055435.785Z` (83.02%) with seed 1337 on MPS, a cosine decay to `0.00001`, and no replay, distillation, or calibration. Accuracy is deployed-int6 agreement of the selected epoch on the verification shard.

| Batch parts | Starting rate | Epochs | Weight average | Accuracy |
| --- | --- | --- | --- | --- |
| 262,144 | 0.0004 | 24 | none | 84.49% |
| 65,536 | 0.0002 | 24 | none | 84.75% |
| 65,536 | 0.0004 | 24 | none | 85.28% |
| 65,536 | 0.0008 | 24 | none | 85.91% |
| 65,536 | 0.0008 | 24 | 0.99 | 85.97% |
| 32,768 | 0.0008 | 24 | none | 86.33% |
| 65,536 | 0.0016 | 24 | none | 86.54% |
| 65,536 | 0.0032 | 24 | none | 86.91% |
| 65,536 | 0.0064 | 24 | none | 87.73% |
| 65,536 | 0.0128 | 24 | 0.99 | 87.67% |
| 65,536 | 0.0064 | 48 | 0.99 | 88.41% |

At `0.0064`, consecutive raw epochs after the tenth still differ by up to 0.8 points while the averaged weights move by 0.23 or less, which is why averaging is on by default. Hiding the lexeme hashes of 10% of parts during training scored 85.85% against 85.91% at rate `0.0008` and was not kept.

These are single runs, MPS is not bit-reproducible (two runs with the same seed and settings differed by up to 0.35 points at individual epochs), and the verification shard that scores a run also selects its epoch. A second run of the last row through `pnpm train`'s defaults, `20261114T092944.130Z`, selected 88.20% and is the promoted checkpoint.

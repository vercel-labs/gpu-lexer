# Training and promotion policy

The current trainer is tree-only and uses the tracked promoted checkpoint as epoch zero. `pnpm train` and `pnpm train:tree` are equivalent. Use `--fresh` only for an intentional random baseline.

A checkpoint that records the current label taxonomy version continues with 48 full-model agreement epochs: unit part weights on the natural corpus, int6 quantization-aware training throughout, 65,536-part batches, and a cosine learning rate from `0.0064` to `0.00001`. Every epoch evaluates the raw weights and an exponential average of the latent weights (decay `0.99`), and either can be selected. Half of the loss is divided among the 25 most popular language families in proportion to their popularity (`--top-language-loss-share`, `0.5` for continuations), so training data added for other languages does not dilute them. After a taxonomy change, or for older checkpoints without a recorded version, compatible weights start the full 32-epoch schedule: learning rate `0.002` initially, with quantization-aware training in the final six epochs. Runs record the target taxonomy in their configuration and the selected checkpoint's taxonomy in model metadata; selecting epoch zero preserves its original version.

The language objective is built from the pinned popularity snapshot plus explicit supplemental web-language weights. Natural-distribution agreement selects checkpoints. Weak, unguarded languages can receive gentle training multipliers, while strict and mature languages have direct per-language regression caps for promotion. Languages with enough support and less than 10% error can graduate into the mature guard set. Automatic membership carries forward only within the same taxonomy; after a taxonomy change it is rebuilt from the baseline evaluated on the new labels. Explicitly configured mature languages remain guarded.

Training combines natural rehearsal, boundary-weighted loss, auxiliary lexical-state supervision, quantization-aware epochs, optional teacher/active-model distillation, and final natural-frequency classifier calibration. Independent mining contributes a small error-ranked replay fraction for weak families. Verification is never replayed into training.

Continuations skip replay, distillation, and calibration by default. They return when the schedule leaves epochs before agreement, for example `--epochs 10 --agreement-epochs 6 --calibration-epochs 2`; those replay-polish epochs use `--lr` and `--final-lr`.

Targeted correction uses:

```sh
pnpm fine-tune -- --file ./failure.tsx --lang tsx
```

The snippet is labeled by Shiki, added to an ignored local failure bank, and trained with balanced failure-only updates followed by ordinary rehearsal. The exact promoted int6 model remains epoch zero. A snippet match alone is not sufficient for promotion.

Every run writes float weights, packed weights, metadata, and history under ignored `packages/training/runs/<timestamp>/`. A continuation against the promoted baseline scores each candidate by the website's Top-25 weighted agreement, less however far any measurable language fell below the baseline beyond five points. It saves the highest score across epoch zero and every raw/EMA candidate, retaining the earlier checkpoint on ties. Other accuracy-selected runs save the highest held-out quantized accuracy. Explicit weighted-error experiments use their configured score instead. Language guards do not filter checkpoint selection or reset the progress counter; a best checkpoint that fails them is still saved with diagnostic-only status.

The promoted baseline is evaluated on the run's current verification labels and stays fixed throughout the run. A selected checkpoint is eligible for promotion only when its deployed quantized accuracy strictly improves that baseline, its Top-25 weighted agreement does not fall, and every strict/mature language guard passes. Eligible runs auto-promote; manual review can invoke `pnpm model:promote <run-id-or-path>`.

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

These are single runs, MPS is not bit-reproducible (two runs with the same seed and settings differed by up to 0.35 points at individual epochs), and the verification shard that scores a run also selects its epoch. A second run of the last row through `pnpm train`'s defaults, `20261114T092944.130Z`, selected 88.20%.

## Build and script language sources

Shell, Makefile, Dockerfile, and Batchfile carry 21% of the Top-25 weight, but the training sources held far fewer of their files than the sampling targets asked for. Twenty-four pinned sources were added to the training split, each limited to those four families, which raised their share of the 3,000,000 scored training parts from 6% to 14%. Verification is unchanged.

Each row is one 48-epoch continuation of `20261114T092944.130Z` with the default settings; Top-25 is the website's popularity-weighted agreement.

| Training data | Run | Top-25 | All languages |
| --- | --- | --- | --- |
| Starting checkpoint | `20261114T092944.130Z` | 92.20% | 88.20% |
| Unchanged | `20261116T131125.232Z` | 92.51% | 88.76% |
| With the added sources | `20261116T131208.426Z` | 93.47% | 88.94% |

Makefile rose from 75.14% to 90.77%, Batchfile from 72.86% to 81.79%, Dockerfile from 89.64% to 92.84%, and Shell from 91.08% to 92.41%. The other languages gave up training parts to make room. `MODEL_CARD.md` lists what declined.

## Language coverage, classifier width, and selection

The previous checkpoint had no training parts for 36 of the verification languages, and the build-language sources had displaced parts of the others. Fifty-five more pinned sources, each limited to listed families, and file-extension mappings for 28 more languages raised the training shard to 4,200,000 scored parts. Seventy-three language families now have at least 5,000 parts, up from 33, and the ten most popular languages fell from 58% of the shard to 45%.

Each row is a 48-epoch continuation of `20261116T131208.426Z`. Figures are means of the averaged weights over the last eight epochs, because no checkpoint was saved for these runs.

| Setup | Runs | Top-25 | All languages |
| --- | --- | --- | --- |
| Starting checkpoint | | 93.47% | 88.94% |
| Added languages, 72-unit classifier | 2 | 92.91–93.12% | 89.0–89.6% |
| Also half the loss for the top languages | 1 | 93.60% | 89.6% |
| Also a 176-unit classifier | 3 | 93.70–94.04% | 89.9–90.1% |

The training shard gained a few sources between rows, so only the direction is comparable across them.

Added languages alone lowered Top-25: JavaScript's averaged agreement fell from 98.0% to 96.5%. Reserving half of the loss for the top languages recovered that. On the same training shard, the wider classifier then scored 93.98% against 93.60%. `--classifier-size 176` widens the promoted model at epoch zero: the new units read the leaf and tree context through small random weights and have zero output weights, so every existing int6 code and every label is unchanged until training moves them. The model grows from 41,609 to 50,969 parameters and the package from 28,256 to 31,062 bytes (31,011 after the runtime changes recorded in `architecture.md`).

Two more runs with these settings saved a checkpoint:

| Run | Seed | Selected | Top-25 | All languages | Plain false color |
| --- | --- | --- | --- | --- | --- |
| `20261117T071411.182Z` (promoted) | 1337 | epoch 29, averaged | 93.88% | 89.89% | 19.06% |
| `20261117T071530.670Z` | 7 | epoch 25, averaged | 93.98% | 90.04% | 19.17% |

The first was promoted although it scores 0.10 lower, a difference within run-to-run variation. Among the languages that can be measured it lost more than a point only in Markdown and Ruby, while the second also lost PHP (91.90% to 89.99%), Vue, HTML, CSS, and Svelte.

### Which languages can show a drop

Of the 76 verification languages with at least 400 supervised parts, 43 rest on what is effectively a single file: every website probe, YAML, Zig, and Haskell, and others such as GLSL where one file holds nearly all the parts. Their scores move by ten points or more between neighboring epochs of one run, and a different one is lowest each time. With all 76 counted, the drop penalty rejected every epoch of every run above.

The penalty therefore counts a language only if it has 400 parts spread over the equivalent of three or more files (the squared part total divided by the sum of squared per-file parts). Thirty languages qualify. The others are still reported, and still need more held-out files before a drop in them means anything.


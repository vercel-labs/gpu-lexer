# gpu-lexer model card

## Model

The package embeds checkpoint `20261117T071411.182Z`, a format-9 hierarchical tree classifier with 32 hidden channels and a 176-unit classifier. It has 50,969 training parameters; the browser projection retains 50,681 reachable int6 weights (38,011 packed bytes). The full minified JavaScript package, including shader and weights, is 30.28 KiB (31,011 bytes) with Brotli compression.

The model receives no language ID. It predicts one of nine visual labels for each mechanically split source part: plain, comment, string, number, keyword, type, function, constant, or operator. It also learns auxiliary lexical states used internally by the classifier.

## Intended use

The model is intended for experimental client-side syntax coloring when a language hint or grammar bundle is unavailable, especially for large or mixed-language source. It may also be useful as a compact feature extractor for code-viewing experiments.

It is not suitable for parsing, compilation, linting, source transformation, semantic analysis, access control, malware detection, or any decision where a wrong label has security or correctness consequences.

## Training data

The corpus consists of permissively licensed, pinned Git repositories and exact npm package versions listed in `packages/training/data/corpus.json`. Downloaded source and generated shards are not distributed in the repository. Train, verification, and mining sources are repository/package-disjoint. Website examples are reserved from training and included in verification.

Shiki 4.4.3 provides offline TextMate-scope supervision. Scopes are normalized to the nine display classes and auxiliary states. This makes the model an approximation of the chosen Shiki mapping, not ground truth for programming-language semantics.

The promoted run saw 6,357,620 prepared training parts per natural corpus pass, of which 4,200,000 were supervised non-whitespace parts, from 7,833 files. Of the 122 pinned training repositories, 79 are limited to listed language families: they were added for build and script languages, or for languages that previously had little or no training data, such as YAML, Zig, Haskell, Julia, Bicep, SystemVerilog, and Visual Basic. Seventy-three language families now have at least 5,000 training parts, up from 33. Training may add balanced or replayed parts depending on the stage.

## Evaluation

The promoted deployed-int6 checkpoint scores:

- 89.89% per-part agreement with normalized Shiki labels
- 84.28% styled macro F1
- 724,666 verification parts across 1,929 files, with 481,512 supervised parts

These figures come from the held-out verification corpus recorded in the checkpoint. Accuracy is dominated by the natural part distribution and is not equal across languages or classes. The website's Top-25 weighted agreement is a separate GitHub-popularity-weighted comparison and must not be substituted for the checkpoint score.

On the same taxonomy-v5 labels and 481,512 supervised verification parts, the original deployed checkpoint scored 72.85% accuracy and 70.57% styled macro F1. The first accuracy-focused continuation reached 81.21% and 75.94%. A controlled learning-rate comparison then reached 83.02% and 77.73%, and a continuation with a higher rate, smaller batches, and more epochs reached 88.20% and 82.32%, all without changing model size or training data. A further continuation on training data with added Shell, Makefile, Dockerfile, and Batchfile sources reached 88.94% and 82.84% at the same model size. The current checkpoint widened the classifier from 72 to 176 units, added training sources for the uncovered languages, and reserved half of the loss for the 25 most popular languages; it reached 89.89% and 84.28%, and is the first of these steps to add parameters. The original published 88.02% score used older labels and is not directly comparable.

The three September learning-rate trials started from checkpoint `20260916T051903.950Z` with seed 1337, fixed 262,144-token batches, 24 full-model agreement epochs, and cosine decay to 0.00001. Selection used the best checkpoint from each run, independently evaluated after int6 deployment quantization:

| Starting learning rate | Best epoch | Accuracy | Styled macro F1 | Plain-token false-color rate |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 81.21% | 75.94% | 37.23% |
| 0.00005 | 22 | 81.32% | 76.14% | 35.87% |
| 0.0001 | 23 | 81.92% | 76.52% | 35.02% |
| 0.0002 | 19 | 83.02% | 77.73% | 33.19% |

The winning rate improved accuracy by 1.81 percentage points over the fixed baseline and reduced plain-token false coloring. This establishes that 81.21% was not a fixed model-capacity ceiling; it does not establish that 0.0002 is globally optimal. The comparison uses one seed and the same verification set used for checkpoint selection.

The November continuation from that 83.02% checkpoint is compared across rates, batch sizes, and epoch counts in `packages/training/WEIGHTED-TRAINING.md`. The promoted run used the same settings: 65,536-part batches and a rate decaying from 0.0064 to 0.00001 over 48 epochs. The same file compares the wider classifier, the added languages, and the top-language loss share.

Prioritizing overall accuracy still leaves regressions compared with the original deployed model: plain-token false coloring is 19.06% versus 14.73%, Vala accuracy is 46.43% versus 83.51%, and Kotlin is 89.95% versus 92.51%. Relative to the previous 88.94% checkpoint, 61 of the 91 verification languages improved and 29 declined. Most of what the previous checkpoint had lost came back: YAML (28.18% to 43.35%, still below 74.17% three checkpoints ago), Bicep (59.50% to 72.83%), SystemVerilog (68.39% to 75.37%), Zig (74.58% to 77.89%), Vue (85.16% to 87.50%), Java (92.98% to 93.71%), and C (95.20% to 96.37%). Among the 25 most popular languages, PHP (91.90% to 91.07%), Dart (94.81% to 93.11%), Ruby (85.26% to 84.09%), C# (91.13% to 90.78%), Kotlin (90.29% to 89.95%), PowerShell (91.15% to 90.87%), and TypeScript (96.44% to 96.21%) declined, as did Procfile on 193 parts. Only 30 of the 91 languages have enough held-out files to show a real change: 43 of the 76 with at least 400 parts rest on what is effectively one file, and such a score moves by ten points or more between neighboring epochs. Among those 30, Markdown (80.41% to 77.93%) and Ruby are the only declines over one point. The largest declines outside them are Haskell (68.09% to 53.66%, one 423-part file), GLSL (91.27% to 86.59%), and Solidity (96.57% to 94.49%). Shiki colors brackets and semicolons as operators in GLSL but leaves them plain in HLSL and ShaderLab, which the model cannot tell apart without a language ID. These remain advisory diagnostics.

The website comparison regenerates Shiki labels with the current taxonomy for every benchmark run, scoring all engines, including the deployed model, against those fresh targets. The checkpoint's verification digest pins the original source selection; the generated comparison also records a separate digest of the regenerated labels. The checkpoint scores above describe the taxonomy used at promotion.

## Limitations

- Accuracy varies substantially by language, source style, and class.
- Ambiguous punctuation, Markdown/MDX, SQL, shell syntax, minified code, malformed source, and embedded-language boundaries remain difficult.
- Labels can be locally plausible but structurally wrong over long strings or comments.
- Shiki itself can be ambiguous or inconsistent after scope normalization.
- Unseen languages may work by structural similarity, but there is no guarantee of useful output.
- Quantization and browser GPU implementations may introduce differences unless parity is explicitly tested.
- WebGPU startup and dispatch overhead can make small snippets slower than CPU lexers.

## Reproducibility

`packages/training/active/` contains compact metadata, exact float weights, and exact deployed int6 weights for the promoted checkpoint. The metadata records tensor layout, feature/tokenizer versions, language objective, verification digest, class order, and quantization scales. Generated corpora remain local and can be reconstructed from the pinned manifest with `pnpm --filter @gpu-lexer/training corpus:prepare`.

New runs start from the deployed promoted weights by default. Promotion re-evaluates both candidate and fixed baseline on the pinned direct-label verification shard and requires a strict overall accuracy improvement; a run selected with the top-language guard must also not lower the Top-25 weighted agreement. Regular training uses advisory language guards, preserving per-language regressions in the diagnostics without vetoing an overall gain. `--language-guards strict` and targeted fine-tuning enforce the language limits. Corpus provenance, label coverage, and matching evaluation support remain required in both modes.

This checkpoint was selected at epoch 29 of a 48-epoch run on MPS, starting from `20261116T131208.426Z` with its classifier widened from 72 to 176 units; the added units start with zero output weights, so epoch zero labels exactly as that checkpoint did. All epochs updated the full model using 65,536-part batches and int6 quantization-aware training, with no replay, distillation, or classifier-only calibration. Half of the loss was divided among the 25 most popular language families in proportion to their popularity, and the rest used unit supervised-part weights. The learning rate decayed from 0.0064 to 0.00001. Each epoch also evaluated a 0.99 exponential average of the weights; the selected weights are the averaged epoch-29 weights, chosen by Top-25 weighted agreement less any drop over five points among the 30 measurable languages. Seed: 1337. Train shard SHA-256: `881825626c89497de7b604e59ec6b2de46010a1b9fe4d4257a0488957663ea0c`; verification shard SHA-256: `6757c82f635faef58fa7b9526ed1a0d7531755079004c0d46d5fa193cc36e0d2`.

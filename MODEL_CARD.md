# gpu-lexer model card

## Model

The published package embeds checkpoint `20260916T055435.785Z`, a format-9 hierarchical tree classifier. It has 41,609 training parameters; the browser projection retains 41,321 reachable int6 weights (30,991 packed bytes). The full minified JavaScript package, including shader and weights, is 27.64 KiB (28,305 bytes) with Brotli compression.

The model receives no language ID. It predicts one of nine visual labels for each mechanically split source part: plain, comment, string, number, keyword, type, function, constant, or operator. It also learns auxiliary lexical states used internally by the classifier.

## Intended use

The model is intended for experimental client-side syntax coloring when a language hint or grammar bundle is unavailable, especially for large or mixed-language source. It may also be useful as a compact feature extractor for code-viewing experiments.

It is not suitable for parsing, compilation, linting, source transformation, semantic analysis, access control, malware detection, or any decision where a wrong label has security or correctness consequences.

## Training data

The corpus consists of permissively licensed, pinned Git repositories and exact npm package versions listed in `packages/training/data/corpus.json`. Downloaded source and generated shards are not distributed in the repository. Train, verification, and mining sources are repository/package-disjoint. Website examples are reserved from training and included in verification.

Shiki 4.4.3 provides offline TextMate-scope supervision. Scopes are normalized to the nine display classes and auxiliary states. This makes the model an approximation of the chosen Shiki mapping, not ground truth for programming-language semantics.

The promoted run saw 4,679,585 prepared training parts per natural corpus pass, of which 3,079,424 were supervised non-whitespace parts. Training may add balanced or replayed parts depending on the stage.

## Evaluation

The promoted deployed-int6 checkpoint scores:

- 83.02% per-part agreement with normalized Shiki labels
- 77.73% styled macro F1
- 724,666 verification parts across 1,929 files, with 481,512 supervised parts

These figures come from the held-out verification corpus recorded in the checkpoint. Accuracy is dominated by the natural part distribution and is not equal across languages or classes. The website's Top-25 weighted agreement is a separate GitHub-popularity-weighted comparison and must not be substituted for the checkpoint score.

On the same taxonomy-v5 labels and 481,512 supervised verification parts, the original deployed checkpoint scored 72.85% accuracy and 70.57% styled macro F1. The first accuracy-focused continuation reached 81.21% and 75.94%. A controlled learning-rate comparison then reached 83.02% and 77.73%, without changing model size or training data. The original published 88.02% score used older labels and is not directly comparable.

All three learning-rate trials started from checkpoint `20260916T051903.950Z` with seed 1337, fixed 262,144-token batches, 24 full-model agreement epochs, and cosine decay to 0.00001. Selection used the best checkpoint from each run, independently evaluated after int6 deployment quantization:

| Starting learning rate | Best epoch | Accuracy | Styled macro F1 | Plain-token false-color rate |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 81.21% | 75.94% | 37.23% |
| 0.00005 | 22 | 81.32% | 76.14% | 35.87% |
| 0.0001 | 23 | 81.92% | 76.52% | 35.02% |
| 0.0002 | 19 | 83.02% | 77.73% | 33.19% |

The winning rate improved accuracy by 1.81 percentage points over the fixed baseline and reduced plain-token false coloring. This establishes that 81.21% was not a fixed model-capacity ceiling; it does not establish that 0.0002 is globally optimal. The comparison uses one seed and the same verification set used for checkpoint selection.

Prioritizing overall accuracy still leaves substantial regressions compared with the original deployed model: plain-token false coloring is 33.19% versus 14.73%, Vala accuracy is 48.56% versus 83.51%, diff is 66.08% versus 75.49%, and Kotlin is 73.67% versus 92.51%. Relative to the 81.21% continuation baseline, some languages also declined, including ShaderLab (5.62 points) and YAML (4.40 points). These remain advisory diagnostics.

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

New runs start from the deployed promoted weights by default. Promotion re-evaluates both candidate and fixed baseline on the pinned direct-label verification shard and requires a strict overall accuracy improvement. Regular training uses advisory language guards, preserving per-language regressions in the diagnostics without vetoing an overall gain. `--language-guards strict` and targeted fine-tuning enforce the language limits. Corpus provenance, label coverage, and matching evaluation support remain required in both modes.

This checkpoint was selected at epoch 19 of a 24-epoch run on MPS, starting from `20260916T051903.950Z`. All epochs updated the full model using unit supervised-part weights and int6 quantization-aware training, with no replay, distillation, or classifier-only calibration. The learning rate decayed from 0.0002 to 0.00001. Seed: 1337. Train shard SHA-256: `7e6e5131287c51c544d65661f2384b47a08dd061e036449f4c11b14e4d8d6268`; verification shard SHA-256: `6757c82f635faef58fa7b9526ed1a0d7531755079004c0d46d5fa193cc36e0d2`.

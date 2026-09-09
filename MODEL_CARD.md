# gpu-lexer model card

## Model

The published package embeds checkpoint `20260909T074331.018Z`, a format-9 hierarchical tree classifier. It has 41,609 training parameters; the browser projection retains 41,321 reachable int6 weights (30,991 packed bytes). The full minified JavaScript package, including shader and weights, is 27.46 KiB (28,115 bytes) with Brotli compression.

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

- 88.02% per-part agreement with normalized Shiki labels
- 79.09% styled macro F1
- 724,666 verification parts across 1,929 files, with 481,512 supervised parts

These figures come from the held-out verification corpus recorded in the checkpoint. Accuracy is dominated by the natural part distribution and is not equal across languages or classes. The website's Top-25 weighted agreement is a separate GitHub-popularity-weighted comparison and must not be substituted for the checkpoint score.

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

New runs start from the deployed promoted weights by default. Promotion re-evaluates both candidate and fixed baseline on the pinned direct-label verification shard and requires a strict accuracy improvement plus all configured language guards.

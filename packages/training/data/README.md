# Training data

`corpus.json` pins permissively licensed Git repositories to full commits and npm packages to exact versions. Train, verification, and mining sources are repository/package-disjoint. `language-popularity.json` records the GitHub Innovation Graph snapshot used for sampling and evaluation weights.

The builder selects by non-whitespace runtime parts, not file count. It prioritizes difficult lexical constructs, reserves compiled capacity for minified JavaScript/CSS/HTML, removes exact and structural duplicates, and stores direct UTF-16 Shiki source spans. Language names are offline sampling and supervision metadata; they never enter runtime features.

Generate local data with:

```sh
pnpm --filter @gpu-lexer/training corpus:prepare
```

Downloaded repositories, package archives, generated sources, summaries, and compressed JSONL shards live under ignored `data/generated/`. The tracked `corpus-summary.json` documents the last prepared corpus without redistributing source.

The selected training, verification, and mining sources have been relabeled with taxonomy version 5. The summary records the old and new shard digests and a source digest for each split. Original shards are retained locally under `data/generated/shards/snapshots/<sha256>.jsonl.gz`, so benchmarks for older checkpoints can keep their pinned source selection while regenerating current labels. The taxonomy version participates in the corpus cache key.

The current promoted checkpoint records 4,679,585 training parts and 724,666 verification parts. Verification includes the website examples, capped to the same 20,000 UTF-16 source units shown in the demo, and those paths are excluded from training and replay.

Shiki labeling failures are recorded rather than silently converted to plain labels. Website examples fail closed so an example cannot disappear from verification unnoticed. Existing shards without direct `sourceLabels` are rejected and must be rebuilt.

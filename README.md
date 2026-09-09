# gpu-lexer

`gpu-lexer` is an experimental, language-agnostic syntax lexer powered by WebGPU. It uses one small learned model for every language and returns display-oriented source spans without a language argument or bundled grammar packs.

```js
import { parse } from "gpu-lexer";

const spans = await parse("const answer = 42");
// [{ type: "keyword", start: 0, end: 5 }, ...]
```

The public package exports only `parse(code)`. It requires WebGPU and a secure browser context. Offsets are UTF-16 code-unit offsets, matching JavaScript string slicing.

## How it works

The CPU makes one mechanical pass over the source, splitting it into word runs, horizontal whitespace, newlines, and individual symbols. It records compact language-neutral features such as part kind, length, edge characters, hashes, and neighboring symbol pairs.

The GPU embeds those sparse features into 32 learned channels. A five-part neighborhood and exact bidirectional affine scans add ordered local context. A shared-weight binary tree then combines 32-part blocks bottom-up, merges block roots across the whole source, and propagates context back down. A small classifier assigns one of nine visual classes to every original part in parallel; adjacent equal classes become the returned spans.

The promoted runtime is format 9 (`local-affine-tree`), with 41,321 reachable browser weights encoded at six bits each. The current minified package is 27.46 KiB (28,115 bytes) with Brotli compression.

## Accuracy

The model is trained against Shiki labels, so reported quality is agreement with Shiki rather than objective semantic correctness. The promoted checkpoint `20260909T074331.018Z` reaches 88.02% agreement and 79.09% styled macro F1 on the held-out verification corpus. The verification split is repository/package-disjoint from training and also includes the website examples. See [MODEL_CARD.md](MODEL_CARD.md) for the evaluation contract and limitations.

`gpu-lexer` can label unfamiliar languages because no language ID is supplied, but it may confidently misclassify ambiguous syntax. It is not a parser, compiler, linter, or security tool.

## Development

Install dependencies with Node.js 20+ and pnpm 11:

```sh
pnpm install
pnpm test:core
pnpm build:core
```

The generated corpus, downloaded repositories, training runs, benchmark results, and local failure bank are intentionally ignored. To prepare data and train:

```sh
pnpm --filter @gpu-lexer/training corpus:prepare
pnpm train
```

`pnpm train` warm-starts the tracked promoted checkpoint by default. `pnpm fine-tune -- --file ./failure.tsx --lang tsx` adds a local failure example and performs guarded fine-tuning. Runs are written under `packages/training/runs/`. An eligible run that strictly improves untouched verification accuracy and passes language guards is promoted automatically; manual promotion is available through `pnpm model:promote <run-id-or-path>`.

The tracked `packages/training/active/` directory contains the compact float and deployed checkpoint needed to reproduce continuation training from a clean clone. Promotion refreshes it atomically after verification.

## Repository

- `packages/core`: publishable browser package and WGSL runtime
- `packages/training`: corpus, Shiki labeling, PyTorch training, evaluation, and promotion
- `packages/benchmark`: size, browser performance, and normalized agreement comparisons
- `apps/website`: project site and interactive demo
- `video`: explainer source and storyboard

Architecture details live in [architecture.md](architecture.md). Third-party attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT © Shu Ding. Source corpora and comparison libraries retain their own licenses.

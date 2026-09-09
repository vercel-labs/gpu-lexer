# Architecture

## Contract

`parse(source)` accepts one complete JavaScript string and returns non-overlapping `{ type, start, end }` spans. The runtime receives no language ID. Positions are UTF-16 code units. The only public output classes are `plain`, `comment`, `string`, `number`, `keyword`, `type`, `function`, `constant`, and `operator`.

The package keeps its WebGPU device, pipelines, weights, and grow-only buffers resident between calls. Calls are scheduled through one internal runtime so the one-function API does not imply stateless initialization.

## Source preparation

One CPU scan emits four mechanical part kinds:

- ASCII letters, digits, underscores, and non-ASCII code units form word runs.
- Horizontal whitespace forms runs.
- CRLF is one newline part; other newlines are individual parts.
- Every other code unit is an individual symbol.

For each part, the scan packs its kind, logarithmic length bucket, normalized first and last character, two word hashes, line-start and shape flags, and generic neighboring-symbol hashes into two `u32` values. No language grammar, regex collection, semantic segmenter, or token object graph runs on the CPU.

## Learned context

The promoted format-9 model has 32 hidden channels and a 72-unit classifier.

1. Sparse part features are summed into one learned vector per part.
2. A shared five-part local window and nearest non-whitespace neighbors mix immediate evidence.
3. Learned affine state updates scan the sequence in both directions. The browser kernel evaluates these scans in parallel 32-part blocks and carries exact block prefixes, so block boundaries do not reset context.
4. Enhanced leaves merge bottom-up through a scale-aware butterfly binary tree. Boundary-preserving residual channels retain information from both ends of a region.
5. Whole-source context propagates top-down through parent, self, sibling, cross-channel, and skip terms.
6. The classifier combines each enhanced leaf, returned tree context, and predicted auxiliary lexical states to produce nine logits.
7. Four byte-sized labels share each readback word. The CPU maps labels back to the prepared ranges and merges adjacent equal styled labels.

Large inputs are tiled into 32-part subtrees for WebGPU workgroup locality. Block roots continue through the global tree; tiling is an execution detail, not semantic segmentation.

## Model representation

Training retains 41,609 float parameters. The browser projection removes unreachable feature rows and a training-only neighbor tensor, leaving 41,321 weights. Each tensor uses symmetric per-tensor signed int6 quantization. Six-bit codes are stored as a compression-friendly one-character alphabet and expanded into GPU buffers once during initialization.

The shader is generated from the promoted model metadata and minified during the core build. Promotion validates tensor names, shapes, offsets, scales, packed length, model format, feature version, class order, and verification provenance before replacing generated runtime artifacts.

## Training and supervision

Shiki provides offline source spans. Its scopes are normalized into the nine visual classes plus auxiliary states for strings, comments, directives, markup, embedded regions, CSS regions, member access, clause regions, and bracket depth. Whitespace remains context-only.

The trainer uses repository/package-disjoint train, verification, and mining splits. It supports quantization-aware training, active-model consistency distillation, boundary-weighted loss, weak-language curriculum weights, natural-distribution calibration, and a small replay bank. Popular languages have strict or mature per-language regression guards.

The tracked active checkpoint contains the exact promoted float and int6 artifacts, tensor layout, feature contract, language objective, and verification digest. Historical runs and optimizer intermediates are not part of the repository.

## Performance boundaries

WebGPU is most useful for warm, large, or batched inputs. Device acquisition, pipeline compilation, buffer allocation, and model upload affect the first call. Warm calls reuse those resources. CPU preparation, queue submission, GPU inference, readback, span reconstruction, and DOM rendering are measured separately in the browser benchmark.

The runtime returns spans to JavaScript, so it necessarily pays one readback synchronization. It does not render text on the GPU. For tiny snippets, a CPU lexer can be faster because dispatch overhead dominates.

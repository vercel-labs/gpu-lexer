# Architecture

## Contract

`parse(source)` accepts one complete JavaScript string and returns non-overlapping `{ type, start, end, confidence }` spans. The runtime receives no language ID. Positions are UTF-16 code units. The only public output classes are `plain`, `comment`, `string`, `number`, `keyword`, `type`, `function`, `constant`, and `operator`.

The package keeps its WebGPU device, pipelines, weights, and grow-only buffers resident between calls. Calls are scheduled through one internal runtime so the one-function API does not imply stateless initialization.

## Source preparation

One CPU scan emits four mechanical part kinds:

- ASCII letters, digits, underscores, and non-ASCII code units form word runs.
- Horizontal whitespace forms runs.
- CRLF is one newline part; other newlines are individual parts.
- Every other code unit is an individual symbol.

For each part, the scan packs its kind, logarithmic length bucket, normalized first and last character, two word hashes, line-start and shape flags, and generic neighboring-symbol hashes into two `u32` values. No language grammar, regex collection, semantic segmenter, or token object graph runs on the CPU.

## Learned context

The promoted format-9 model has 32 hidden channels and a 176-unit classifier.

1. Sparse part features are summed into one learned vector per part.
2. A shared five-part local window and nearest non-whitespace neighbors mix immediate evidence.
3. Learned affine state updates scan the sequence in both directions. The browser kernel evaluates these scans in parallel 32-part blocks and carries exact block prefixes, so block boundaries do not reset context.
4. Enhanced leaves merge bottom-up through a scale-aware butterfly binary tree. Boundary-preserving residual channels retain information from both ends of a region.
5. Whole-source context propagates top-down through parent, self, sibling, cross-channel, and skip terms.
6. The classifier combines each enhanced leaf, returned tree context, and predicted auxiliary lexical states to produce nine logits.
7. Four byte-sized labels share each readback word. Each byte holds the class in its low four bits and, in its high four, the selected class's softmax probability in sixteenths. The CPU maps labels back to the prepared ranges, merges adjacent equal styled labels, and reports the lowest confidence in each span.

Large inputs are tiled into 32-part subtrees for WebGPU workgroup locality. The local scan and tree-up kernels process four parts concurrently with 128-thread workgroups, while retaining the original arithmetic order within each part and recurrent scan. Block roots continue through the global tree; tiling is an execution detail, not semantic segmentation.

## Model representation

Training retains 50,969 float parameters. The browser projection removes unreachable feature rows and a training-only neighbor tensor, leaving 50,681 weights. Each tensor uses symmetric per-tensor signed int6 quantization. Six-bit codes are stored as a compression-friendly one-character alphabet and expanded into GPU buffers once during initialization.

The runtime weight buffer starts with the classifier heads. Their matrices are stored four output units side by side, so each classifier thread computes four units from one contiguous run of vec4 loads; the matrices of the scan stages and the output layer stay column-major. The package entry imports the weights after the runtime, which places the weight string at the end of the bundle: Brotli then codes the code and the weights as two runs, about 180 bytes smaller than with the weights in the middle. The weights are 76% of the package, and Brotli already codes them within 0.5% of their per-tensor entropy; adaptive models conditioned on neighboring weights, rows, or columns were no smaller, so the weights cannot shrink without changing them.

The shader is generated from the promoted model metadata and minified during the core build. Promotion validates tensor names, shapes, offsets, scales, packed length, model format, feature version, class order, and verification provenance before replacing generated runtime artifacts.

## Training and supervision

Shiki provides offline source spans. Its scopes are normalized into the nine visual classes plus auxiliary states for strings, comments, directives, markup, embedded regions, CSS regions, member access, clause regions, and bracket depth. Whitespace remains context-only.

The trainer uses repository/package-disjoint train, verification, and mining splits. It supports quantization-aware training, active-model consistency distillation, boundary-weighted loss, natural-distribution calibration, and a small replay bank. Regular training selects by overall token accuracy with natural class and language weights. Per-language regression checks are advisory by default; strict mode and targeted fine-tuning retain the language guards and weak-language curriculum.

The tracked active checkpoint contains the exact promoted float and int6 artifacts, tensor layout, feature contract, language objective, and verification digest. Historical runs and optimizer intermediates are not part of the repository.

## Performance boundaries

WebGPU is most useful for warm, large, or batched inputs. Device acquisition, pipeline compilation, buffer allocation, and model upload affect the first call. Warm calls reuse those resources. CPU preparation, queue submission, GPU inference, readback, span reconstruction, and DOM rendering are measured separately in the browser benchmark.

On September 16, 2026, the four-part workgroups and direct last-span reuse reduced median production-bundle time from 668.25 ms to 557.85 ms for 10 concatenated copies of three.min.js (5,556,500 characters), and from 62.9 ms to 54.7 ms for one copy. Each variant had 18 measured calls across two dedicated workers, with three warm-ups per worker and ABBA ordering. The same checkpoint produced identical labels and spans across 117 cases (3,317,022 parts) in each of FP16 and FP32, including partial blocks. Batched requests also returned identical spans. These are single-device Chrome 152 measurements; timings vary with system load and garbage collection. Raw timings, bundle hashes, and parity results are recorded in [the benchmark report](packages/benchmark/reports/tree-runtime-2026-09-16.json).

On October 22, 2026, four exact changes reduced the same 10-copy input from about 290 ms to 212 ms in fresh benchmark workers, and GPU buffer allocation from 891 MiB to 220 MiB with FP16:

- Matrices read across output units (state input, gate and mix, classifier, auxiliary and output) are stored column-major, so adjacent threads read adjacent weights.
- Local states are recomputed in workgroup memory by the scan and mix passes instead of round-tripping a part × hidden FP32 buffer; each part's raw embedding is computed once per block.
- Nonspace-neighbor block prefixes use a parallel workgroup scan, and both halves of the classify workgroup share the block-local tree levels.
- CPU preparation uses lookup tables and buffer capacity is rounded to a sixteenth of the next power of two.

Labels and spans matched the previous runtime for 5,028 corpus files, one request at a time and batched, plus one and ten copies of three.min.js, in both FP16 and FP32. See [the browser benchmark report](packages/benchmark/reports/browser-performance-2026-10-22.json).

On November 17, 2026, the 176-unit classifier and per-span confidence raised the same 10-copy input from about 228 ms to 270 ms. Both builds ran in headless Chrome with FP16 on an Apple M4 Pro, alternating three times, each run the median of nine fresh-worker samples: the previous package gave 235, 214, and 236 ms, and the new one 270 ms each time. The two changes were not timed separately.

Later that day, four exact changes reduced it from 270 ms to 190 ms (187.0 ms in the seven-library comparison of the report below), and the package from 31,062 to 31,011 bytes:

- Each classifier thread reads a part's leaf and tree context once into private memory and computes four output units per vec4 load. Before, a thread computed one unit at a time and re-read the leaf state from storage for every unit.
- The classifier heads' weights start the weight buffer. On the Apple GPU the classifier loop runs about 40% slower whenever one of its vec4 indices reaches 4096. Binding the same buffer 64 KiB further into memory changed nothing, so the cause is the code Metal generates for small index ranges, not placement. A core test fails if the classifier matrix would end past that index.
- Batches over 4,096 blocks classify in up to eight chunks, each with its own label readback, so the CPU rebuilds one chunk's spans while the GPU classifies the next.
- Span rebuilding compares integer class codes and confidence levels, and creates names and fractions only when a span starts.

In the same profile the classify pass fell from 122 ms to 73 ms; the other passes are unchanged (scan 23 ms, block prefixes 5 ms, mix and tree-up 35 ms, global tree 10 ms), with 28 ms of CPU preparation before them.

On ten copies of three.min.js (1,559,165 spans), a hash over every span's type, range, and confidence matches the previous runtime. Types and ranges are identical to the committed runtime code running the same weights on the 1,929 verification files, one request at a time and batched (434,688 spans), on seven ordinary sources repeated to 2.17 million parts, and, in FP32, on three copies of three.min.js. Against the JS reference, a single copy of three.min.js differs in 4 of 168,813 spans, all near-ties at 0.44 to 0.5 confidence, before and after. See [the browser benchmark report](packages/benchmark/reports/browser-performance-2026-11-17.json).

Measured and not kept: unrolling the classifier loop (3 ms for about 200 bytes), `fma` (slower), sixteen units per thread (slower), more lanes with fewer parts per batch (slower), classifying only styled parts by compaction (slower, because the batch loop bound becomes data-dependent), and a vec4 form of the scan stages (the mix pass would need a ninth storage buffer).

The runtime returns spans to JavaScript, so it necessarily pays one readback synchronization. It does not render text on the GPU. For tiny snippets, a CPU lexer can be faster because dispatch overhead dominates.

# How gpu-lexer Sees Source Code

## Overview

- **Topic**: How gpu-lexer turns raw source into syntax-highlight spans with its active hierarchical tree model.
- **Hook**: A tiny language-agnostic model can label code without being told which language it is.
- **Target Audience**: Web developers; no machine-learning or GPU background required.
- **Estimated Length**: 51–55 seconds, silent, 16:9.
- **Key Insight**: Every source part receives both precise local evidence and compressed whole-file context: context travels up a learned tree, then back down before parallel classification.

## Narrative Arc

Begin with one familiar JavaScript snippet and physically decompose it into the units gpu-lexer observes. Preserve those same units as they become feature vectors, gather context upward into a root, receive context on the downward path, and finally turn into colored syntax spans. The visual payoff is that the original code reassembles with syntax colors even though the model was never given a language name.

---

## Scene 1: One Unlabeled String

**Duration**: ~5 seconds

**Purpose**: Establish the input and the central question immediately.

### Visual Elements

- Pure black 1920×1080 frame.
- Centered two-line monospace snippet in white:

  ```js
  const message = "hello, world!";
  console.log(message)
  ```

- Small top caption: `source in`.
- Bottom caption appears after a short pause: `no language id`.

### Content

The snippet types on rapidly, then holds. `javascript` never appears on screen. A thin white outline grows around the source and becomes the recurring data-flow container.

### Narration Notes

Silent. The juxtaposition of recognizable JavaScript and `no language id` is the hook.

### Technical Notes

- Use `Text` with a bundled monospace font rather than `Code`, so individual glyphs can transform cleanly.
- Keep the camera static and use generous negative space.
- Use a blinking cursor only during the initial type-on; remove it before the transformation.

---

## Scene 2: Characters Become Simple Parts

**Duration**: ~8 seconds

**Purpose**: Explain the only source-specific CPU operation without implying semantic tokenization.

### Visual Elements

- Every glyph separates into a small outlined character cell.
- Adjacent cells then gather into part containers:
  - word runs such as `const`, `message`, `hello`, and `console`;
  - horizontal-space runs shown as `·`;
  - newline shown as `↵`;
  - every symbol kept separate, including `=`, quotes, punctuation, and brackets.
- Caption sequence: `characters` → `one scan` → `simple parts`.
- Small lower caption: `words · spaces · newlines · symbols`.

### Content

First reveal individual characters for roughly half a second. White brackets then group only alphanumeric/underscore runs and whitespace runs; symbols remain one cell each. The result becomes one horizontal 1D strip. Briefly flash compact feature marks below three representative parts: kind, length, edge characters, hashes, flags, and neighbor-pair cues.

This must not visually claim that the implementation allocates an array of character objects. The character cells are an explanatory decomposition; the caption `one scan` makes clear that the CPU directly emits compact parts and features.

### Narration Notes

On-screen copy only. Do not call the parts “tokens”; that can suggest language-aware lexing.

### Technical Notes

- Use `TransformMatchingShapes` from glyphs into grouped part boxes.
- Represent whitespace visibly only during the explainer; preserve its real width.
- Exact part rule: word-like runs, horizontal-space runs, newline sequences, and individual symbols.
- Keep all part mobjects in a stable ordered array. Every later scene transforms these same objects.
- Place isolated source glyphs against a shared Geist Mono baseline; do not center punctuation by its individual bounding box.

---

## Scene 3: GPU Feature Pipeline

**Duration**: ~9 seconds

**Purpose**: Show CNN-like learned feature extraction and ordered local context before the tree.

### Visual Elements

- A large thin frame labeled `webgpu` encloses the part strip.
- Each part sprouts a short 8-cell heatmap column, with a nearby brace labeled `32 learned channels` to clarify that only representative channels are drawn.
- A five-part window slides once across `message = "hello`.
- One luminous five-part window completes a full left-to-right sweep, then a full right-to-left sweep.
- Captions appear in sequence:
  - `embed each part`
  - `mix nearby evidence`
  - `scan both directions`

### Content

Compact CPU features enter the GPU frame and become learned leaf vectors. A convolution-like local window shows that nearby parts and nearest non-space neighbors affect each leaf. Complete forward and reverse sweeps then visualize ordered context moving across the strip. The heatmap cells change intensity but the source-part ordering never changes.

### Narration Notes

The neural-network feeling comes from activation heatmaps and shared moving filters, not from a false 2D-image metaphor.

### Technical Notes

- Use grayscale-to-electric-blue activation values; reserve semantic colors for the output labels.
- The arrows represent the model’s learned bidirectional affine scan, not CPU iteration.
- Avoid drawing fully connected edges between every part and channel; it would be noisy and technically misleading.

---

## Scene 4: Context Climbs the Tree

**Duration**: ~10 seconds

**Purpose**: Make hierarchical whole-file context visually intuitive.

### Visual Elements

- The part strip becomes the leaf row of a binary tree.
- Adjacent pairs connect upward to learned parent nodes.
- Levels illuminate sequentially with a counter:
  `21 → 11 → 6 → 3 → 2 → 1`.
- Each node contains a tiny multi-channel activation bar rather than a scalar dot.
- Top caption: `pass 1 · bottom-up`.
- At the root, caption changes to `one whole-file context`.

### Content

Pairs merge simultaneously at each level. Odd unmatched nodes carry forward rather than disappearing. Boundary-preserving residuals are suggested by thin direct edge lines that continue around each merge, while brighter cross-links between a few feature channels imply butterfly channel mixing. The root pulses once when the whole snippet has been summarized.

### Narration Notes

This is the central slow beat. Allow approximately half a second after the root lights up.

### Technical Notes

- All nodes within one level animate together to communicate GPU parallelism.
- The concrete snippet fits within one 32-part execution block. Do not draw multiple global blocks for this example.
- A tiny corner annotation may read `large files: 32-part roots merge again`, but it must not compete with the main visual.

---

## Scene 5: Context Returns to Every Part

**Duration**: ~9 seconds

**Purpose**: Correct the common misconception that a single root vector can directly label all leaves.

### Visual Elements

- The root emits an electric-blue pulse downward.
- Tree edges illuminate top-to-bottom, one level at a time.
- At each branch, the parent pulse mixes with a dim retained child/sibling activation.
- Fine local skip lines remain visible from Scene 3.
- Caption: `pass 2 · top-down`.

### Content

Whole-snippet context flows back down the same hierarchy. Each leaf ends with two visibly distinct bundles: a local feature strip and a returned context strip. A brace joins them under the text `local detail + whole-file context`.

### Narration Notes

The “aha” is directional symmetry: context goes up to understand the whole, then down so every original part can use it.

### Technical Notes

- Reverse the visual rhythm of Scene 4, but do not simply reverse the animation: every glow must travel from a parent node toward its children, while node activations mix and change on descent.
- Use residual arcs to show that exact leaf evidence is retained rather than reconstructed solely from the root.

---

## Scene 6: Parallel Labels Become Highlight Spans

**Duration**: ~12 seconds

**Purpose**: Connect the network output to the final highlighted result.

### Visual Elements

- A compact classifier head appears above one representative leaf:
  `leaf + context` → small neural layer → nine score bars.
- Score-bar labels: `plain`, `comment`, `string`, `number`, `keyword`, `type`, `function`, `constant`, `operator`.
- The winning bar lights up; classifier heads then appear over all non-whitespace leaves simultaneously.
- Small colored type badges land on their parts:
  - `const` → `keyword`;
  - quoted content and quote/punctuation parts inside it → `string`;
  - `log` → `function`;
  - syntax symbols → `operator` where predicted;
  - ordinary names → `plain`.
- Adjacent equal labels fuse into longer colored spans.
- The tree and feature channels fade, leaving the highlighted original snippet.
- Final captions:
  - `classify every part in parallel`
  - `adjacent labels → highlight spans`

### Content

Demonstrate the classifier once, then multiply the result across every part. Whitespace remains plain/context-only. End on only the fully reconstructed, highlighted source. Lay out each complete source line as one Geist Mono text run before applying colors, preserving its natural spacing.

### Narration Notes

The last four seconds should be nearly still so the highlighted result is readable.

### Technical Notes

- Do not imply deterministic correctness for the illustrative labels; the model predicts the highest of nine scores.
- Animate all leaf classifications together after the single-leaf explanation.
- Use exact UTF-16 start/end boundaries conceptually; numeric offsets need not appear in this short version.

---

## Transitions & Flow

### Scene Connections

- Scene 1 → Scene 2: Preserve every source glyph; spread rather than replace it.
- Scene 2 → Scene 3: Keep part boxes fixed while the `webgpu` frame grows around them.
- Scene 3 → Scene 4: Heatmap columns shrink into tree leaves and remain visible as skip connections.
- Scene 4 → Scene 5: Hold the root pulse, then redirect attention downward with the same tree geometry.
- Scene 5 → Scene 6: Collapse each leaf’s two context strips into its classifier input, then dissolve the machinery behind the reconstructed code.

### Recurring Visual Motifs

- The same ordered part boxes persist from decomposition through final labeling.
- Left-to-right motion means source processing; bottom-to-top means context compression; top-to-bottom means context distribution.
- White outlines represent data structure. Blue activation intensity represents learned internal values. Semantic colors appear only when the model predicts labels.

## Color Palette

| Role | Color | Hex | Usage |
|---|---|---:|---|
| Background | Black | `#000000` | Entire frame |
| Primary | White | `#F5F5F5` | Source, boxes, tree edges, captions |
| Muted | Gray | `#737373` | Inactive nodes and secondary copy |
| Learned activation | Electric blue | `#3B82F6` | Feature heatmaps, scan pulses |
| Comment | Sage | `#6A9955` | `comment` labels |
| String | Green | `#A6E22E` | `string` labels |
| Number | Orange | `#F78C6C` | `number` labels |
| Keyword | Violet | `#C792EA` | `keyword` labels |
| Type | Gold | `#FFCB6B` | `type` labels |
| Function | Blue | `#82AAFF` | `function` labels |
| Constant | Coral | `#F07178` | `constant` labels |
| Operator | Cyan | `#89DDFF` | `operator` labels |
| Plain | Off-white | `#D4D4D4` | `plain` labels |

## Mathematical Content

No equations or LaTeX are necessary. Use only these compact structural notations:

1. `N → ⌈N/2⌉ → … → 1` for upward context aggregation.
2. `1 → … → N` for downward context propagation.
3. `local detail + whole-file context → 9 scores` for classification.

## Implementation Order

1. **Persistent source and part system** — exact glyph-to-part transforms are the continuity backbone.
2. **Reusable activation-strip and tree-node components** — shared by feature, upward, downward, and classifier scenes.
3. **Up/down tree geometry** — validate that 21 leaves remain legible at 1920×1080.
4. **Semantic label and span-merging animation** — establish the final color mapping before tuning intermediate colors.
5. **Captions, GPU enclosure, and scan overlays** — layer concise explanations onto the stable core animation.
6. **Timing and render polish** — target 51–55 seconds and verify readability at the website’s displayed size.

### Shared Components

- `PartCell`: source text, visible whitespace form, kind, stable center, and semantic label.
- `ActivationStrip`: compact multi-channel vector with animated intensity.
- `TreeNode`: activation strip plus parent/child edges.
- `Caption`: consistent Geist Mono annotation, matching the source typography.
- `SemanticPalette`: one mapping used by score bars, badges, and final highlighted source.

## Accuracy Guardrails

- The CPU does not perform language-aware tokenization; it performs one mechanical source scan into simple parts and compact features.
- Characters are shown separately only to explain the scan. They are not materialized as individual runtime objects.
- The active model includes learned local mixing and bidirectional ordered context before its hierarchical tree.
- Pairwise merging is learned and boundary-preserving; the root is not an average pool.
- Context must flow downward before per-part classification.
- Classification predicts one of nine visual classes. It does not build an AST and it can make mistakes.
- Adjacent parts with the same predicted class become returned `{ type, start, end }` spans; whitespace/newlines remain plain and break those runs.

## Reference Material

- [`packages/core/src/prepare-tree.js`](packages/core/src/prepare-tree.js): active simple-part scan and compact feature encoding.
- [`packages/core/src/tree-shader.js`](packages/core/src/tree-shader.js): local context, upward tree, downward propagation, auxiliary heads, and final classifier.
- [`packages/core/src/spans.js`](packages/core/src/spans.js): conversion from per-part labels into adjacent output spans.
- [`architecture.md`](architecture.md): project-level rationale and active hierarchical-tree description.

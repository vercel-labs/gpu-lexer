# gpu-lexer pipeline video

The storyboard is in [`scenes.md`](scenes.md). The animation follows the active hierarchical-tree runtime in `packages/core`: one mechanical CPU scan, learned local and bidirectional context, upward tree aggregation, downward context propagation, parallel nine-class prediction, and span merging.

Render a fast draft:

```sh
manim -ql gpu_lexer_pipeline.py GpuLexerPipeline
```

Render the 1920×1080, 30 fps site asset:

```sh
manim -qh --fps 30 gpu_lexer_pipeline.py GpuLexerPipeline
```

The animation bundles and explicitly registers Geist Mono for all typography. It has no audio track.

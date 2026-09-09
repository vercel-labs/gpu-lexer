# gpu-lexer

A tiny experimental language-agnostic WebGPU syntax lexer. One learned model parses source from any language into display-oriented spans; there are no language arguments or grammar packs.

```sh
pnpm add gpu-lexer
```

```js
import { parse } from "gpu-lexer";

const spans = await parse("const answer = 42");
// [{ type: "keyword", start: 0, end: 5 }, ...]

for (const span of spans) {
  console.log(span.type, source.slice(span.start, span.end));
}
```

`parse(code: string): Promise<SyntaxSpan[]>` is the package's only export. `start` and `end` are half-open UTF-16 code-unit offsets, and the spans cover the source in order.

WebGPU and a secure browser context are required. The first call acquires a device, compiles pipelines, uploads the model, and runs inference; later calls reuse those resources.

The output is probabilistic and trained for agreement with normalized Shiki labels. It is not a parser, compiler, linter, or security tool. See the [project repository](https://github.com/shuding/gpu-lexer) for architecture, benchmarks, and model limitations.

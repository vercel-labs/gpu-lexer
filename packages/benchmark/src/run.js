import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { packUtf16, packUtf8, RawInputWorkspace } from "./raw-input.js";
import hljs from "highlight.js";
import Prism from "prismjs";
import { codeToTokens } from "shiki";
import { highlight as sugarHigh } from "sugar-high";

const samples = [
  { name: "tree-features", path: new URL("../../core/src/tree-features.js", import.meta.url), language: "javascript" },
  { name: "gpu-runner", path: new URL("../../core/src/gpu.js", import.meta.url), language: "javascript" },
];

async function time(name, operation, iterations = 50) {
  for (let index = 0; index < 5; index++) await operation();
  const started = performance.now();
  for (let index = 0; index < iterations; index++) await operation();
  const elapsed = performance.now() - started;
  return { name, milliseconds: elapsed / iterations };
}

const rows = [];
for (const sample of samples) {
  const code = await readFile(sample.path, "utf8");
  const rawWorkspace = new RawInputWorkspace();
  const cases = [
    ["gpu-lexer/simple-tree-prepare", () => {
      const prepared = prepareTreeSource(code);
      releaseTreePrepared(prepared);
    }, 50],
    ["transport/packed-utf16", () => packUtf16(code, rawWorkspace), 50],
    ["transport/utf8-encodeInto", () => packUtf8(code, rawWorkspace), 50],
    ["shiki", () => codeToTokens(code, { lang: sample.language, theme: "github-dark-default" }), 10],
    ["highlight.js", () => hljs.highlight(code, { language: sample.language }).value, 50],
    ["prism", () => Prism.highlight(code, Prism.languages.javascript, "javascript"), 50],
    ["sugar-high", () => sugarHigh(code, { lang: "javascript" }), 50],
  ];
  const benchmarks = [];
  for (const [name, operation, iterations] of cases) benchmarks.push(await time(name, operation, iterations));
  for (const result of benchmarks) {
    rows.push({ sample: sample.name, library: result.name, ms: result.milliseconds.toFixed(3) });
  }
}

console.table(rows);
console.log("Open /packages/benchmark/browser/ after `pnpm --filter @gpu-lexer/benchmark browser` for WebGPU phase, raw-input parity, and readback benchmarks.");

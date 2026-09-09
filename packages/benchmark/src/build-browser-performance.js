import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const source = fileURLToPath(new URL("../browser/performance/", import.meta.url));
const output = resolve(process.argv[2] ?? "/tmp/gpu-lexer-browser-performance");
await mkdir(output, { recursive: true });
await build({
  bundle: true,
  entryNames: "[name]",
  entryPoints: {
    runner: resolve(source, "runner.js"),
    "gpu-lexer": resolve(source, "gpu-lexer.js"),
    prism: resolve(source, "prism.js"),
    highlight: resolve(source, "highlight.js"),
    "sugar-high": resolve(source, "sugar-high.js"),
    "starry-night": resolve(source, "starry-night.js"),
    shiki: resolve(source, "shiki.js"),
    profile: resolve(source, "profile.js"),
    "profile-runner": resolve(source, "profile-runner.js"),
  },
  format: "esm",
  outdir: output,
  platform: "browser",
  target: ["chrome120"],
});
await import("node:fs/promises").then(({ copyFile }) => Promise.all([
  copyFile(resolve(source, "index.html"), resolve(output, "index.html")),
  copyFile(resolve(source, "profile.html"), resolve(output, "profile.html")),
]));
console.log(output);

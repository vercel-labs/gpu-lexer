import assert from "node:assert/strict";
import test from "node:test";

import { build } from "esbuild";

test("each public module exports only parse", async () => {
  const { metafile } = await build({
    bundle: true,
    entryPoints: ["index", "lite"].map((entry) => new URL(`../src/${entry}.js`, import.meta.url).pathname),
    format: "esm",
    loader: { ".wgsl": "text" },
    metafile: true,
    outdir: "out",
    write: false,
  });
  for (const output of Object.values(metafile.outputs)) assert.deepEqual(output.exports, ["parse"]);
});

import assert from "node:assert/strict";
import test from "node:test";

import { build } from "esbuild";

test("public modules export only parse and bundle their own models", async () => {
  const { metafile } = await build({
    bundle: true,
    entryPoints: ["index", "lite"].map((entry) => new URL(`../src/${entry}.js`, import.meta.url).pathname),
    format: "esm",
    loader: { ".wgsl": "text" },
    metafile: true,
    outdir: "out",
    write: false,
  });
  for (const output of Object.values(metafile.outputs)) {
    assert.deepEqual(output.exports, ["parse"]);
    const inputs = Object.keys(output.inputs);
    const isLite = output.entryPoint.endsWith("/lite.js");
    assert.equal(inputs.some((path) => path.endsWith("/model.runtime.generated.js")), !isLite);
    assert.equal(inputs.some((path) => path.endsWith("/shader.min.generated.js")), !isLite);
    assert.equal(inputs.some((path) => path.includes("/lex/lite/")), isLite);
    assert.equal(inputs.some((path) => path.endsWith("/lex/runtime.js")), isLite);
  }
});

import { copyFile, mkdir, writeFile } from "node:fs/promises";
import { brotliCompressSync, constants } from "node:zlib";

import { build } from "esbuild";
import { promotedModel } from "../src/model.generated.js";
import { generatedShaderContents } from "./generate-shader.js";
import { promotedModelContents, runtimeModelContents } from "./generate-runtime-model.js";

const generatedShader = await generatedShaderContents(promotedModel);
const generatedRuntimeModel = runtimeModelContents(promotedModel);
const generatedPromotedModel = promotedModelContents(promotedModel);

await Promise.all([
  writeFile(new URL("../src/shader.min.generated.js", import.meta.url), generatedShader),
  writeFile(new URL("../src/model.runtime.generated.js", import.meta.url), generatedRuntimeModel),
  writeFile(new URL("../src/model.generated.js", import.meta.url), generatedPromotedModel),
]);

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });
await copyFile(
  new URL("../src/index.d.ts", import.meta.url),
  new URL("../dist/index.d.ts", import.meta.url),
);
const result = await build({
  bundle: true,
  entryPoints: [new URL("../src/index.js", import.meta.url).pathname],
  format: "esm",
  minify: true,
  metafile: true,
  outfile: new URL("../dist/index.js", import.meta.url).pathname,
  platform: "browser",
  target: ["es2022"],
  write: true,
});

const output = result.metafile?.outputs;
const bytes = output ? Object.values(output)[0].bytes : null;
const contents = await import("node:fs/promises").then(({ readFile }) =>
  readFile(new URL("../dist/index.js", import.meta.url)),
);
const minifiedBytes = bytes ?? contents.byteLength;
const brotliBytes = brotliCompressSync(contents, {
  params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
}).byteLength;
console.log(`core minified: ${formatBytes(minifiedBytes)} (${minifiedBytes} bytes)`);
console.log(`core minified+brotli: ${formatBytes(brotliBytes)} (${brotliBytes} bytes)`);

function formatBytes(value) {
  return value < 1024 ? `${value} B` : `${(value / 1024).toFixed(2)} KiB`;
}

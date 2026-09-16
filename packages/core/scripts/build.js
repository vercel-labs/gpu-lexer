import { copyFile, mkdir, readFile, rm } from "node:fs/promises";
import { brotliCompressSync, constants } from "node:zlib";

import { build } from "esbuild";
import { initialize, minify, validate } from "wgslender";

// Each model is its own entry; `gpu-lexer` re-exports the default model.
// Code splitting puts the shared runtime in a chunk, so importing one model
// never bundles another.
const ENTRIES = ["index", "lite"];

const wgsl = {
  name: "wgsl",
  setup(build) {
    build.onStart(initialize);
    build.onLoad({ filter: /\.wgsl$/ }, async ({ path }) => {
      const result = minify(await readFile(path, "utf8"), {
        minifyWhitespace: true,
        minifyIdentifiers: true,
        minifySyntax: true,
        treeShaking: true,
      });
      if (result.errors.length) {
        throw new Error(`WGSL minification failed: ${result.errors.map(({ message }) => message).join("; ")}`);
      }
      const validation = validate(result.code);
      if (!validation.valid) {
        throw new Error(`Minified WGSL is invalid: ${validation.diagnostics.map(({ message }) => message).join("; ")}`);
      }
      return { contents: `export default ${JSON.stringify(result.code)};`, loader: "js" };
    });
  },
};

const dist = new URL("../dist/", import.meta.url);
await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await Promise.all(ENTRIES.map((entry) => copyFile(
  new URL("../src/index.d.ts", import.meta.url),
  new URL(`${entry}.d.ts`, dist),
)));
const result = await build({
  bundle: true,
  entryPoints: ENTRIES.map((entry) => new URL(`../src/${entry}.js`, import.meta.url).pathname),
  chunkNames: "[name]-[hash]",
  format: "esm",
  minify: true,
  metafile: true,
  outdir: dist.pathname,
  platform: "browser",
  plugins: [wgsl],
  splitting: true,
  target: ["es2022"],
  write: true,
});

// Reports each entry with every chunk it statically imports, as a bundler
// would ship it.
const outputs = result.metafile.outputs;
for (const entry of ENTRIES) {
  const files = new Set();
  const visit = (file) => {
    if (files.has(file)) return;
    files.add(file);
    for (const { path, kind } of outputs[file].imports) if (kind === "import-statement") visit(path);
  };
  visit(Object.keys(outputs).find((file) => outputs[file].entryPoint?.endsWith(`src/${entry}.js`)));
  const contents = Buffer.concat(await Promise.all([...files].map((file) => readFile(file))));
  const brotliBytes = brotliCompressSync(contents, {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
  }).byteLength;
  console.log(`${entry} minified: ${formatBytes(contents.byteLength)} (${contents.byteLength} bytes)`);
  console.log(`${entry} minified+brotli: ${formatBytes(brotliBytes)} (${brotliBytes} bytes)`);
}

function formatBytes(value) {
  return value < 1024 ? `${value} B` : `${(value / 1024).toFixed(2)} KiB`;
}

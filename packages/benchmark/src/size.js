import { build } from "esbuild";
import { createRequire } from "node:module";
import { dirname } from "node:path";
import { readFile } from "node:fs/promises";
import { brotliCompressSync, constants } from "node:zlib";

const web = ["javascript", "typescript", "css", "html", "json", "markdown"];
const demoLanguages = ["javascript", "typescript", "css", "html", "python", "c", "ruby", "rust", "markdown", "shell"];
const require = createRequire(import.meta.url);
const starryNightPath = require.resolve("@wooorm/starry-night");
const onigurumaPath = require.resolve("vscode-oniguruma/release/onig.wasm", {
  paths: [dirname(starryNightPath)],
});
const onigurumaBytes = compressedBytes(await readFile(onigurumaPath));
const prismAll = Object.keys(require("prismjs/components.js").languages)
  .filter((name) => name !== "meta");
const sugarAll = [
  "javascript", "typescript", "css", "python", "c", "go", "java", "rust", "json", "diff", "shell",
  "cpp", "csharp", "sql", "html", "yaml", "markdown", "plaintext", "ruby", "kotlin", "swift", "php",
  "toml", "powershell", "dockerfile", "graphql", "hcl", "zig", "lua",
];

const targets = [
  { name: "gpu-lexer", scopes: { one: gpuLexer(), web: gpuLexer(), all: gpuLexer() } },
  { name: "Shiki", scopes: {
    one: shiki(["javascript"]), web: shiki(web),
    all: "import { bundledLanguages } from 'shiki/langs'; import * as core from 'shiki/core'; " +
      "import wasm from 'shiki/wasm'; globalThis.__size = [core, wasm, bundledLanguages];",
  } },
  { name: "Highlight.js", scopes: {
    one: highlightJs(["javascript"]), web: highlightJs(web),
    all: "import value from 'highlight.js'; globalThis.__size = value;",
  } },
  { name: "Prism.js", scopes: {
    one: prism(["javascript"]), web: prism(web), all: prism(prismAll),
  } },
  { name: "Sugar High", scopes: {
    one: sugar(["javascript"]), web: sugar(web), all: sugar(sugarAll),
  } },
  { name: "Starry Night", wasm: true, scopes: {
    one: starry(["source.js"]),
    web: starry(["source.js", "source.ts", "source.css", "text.html.basic", "source.json", "text.md"]),
    all: "import { all, createStarryNight } from '@wooorm/starry-night'; " +
      "globalThis.__size = [createStarryNight, all];",
  } },
];

console.log("minified+Brotli bytes");
console.log("library\tone-language\tmajor-web\tall-languages");
for (const target of targets) {
  const sizes = [];
  for (const source of Object.values(target.scopes)) {
    sizes.push(await bundledBytes(source) + (target.wasm ? onigurumaBytes : 0));
  }
  console.log(`${target.name}\t${sizes.join("\t")}`);
}
console.log("\none-language selected-file sizes");
for (const language of demoLanguages) {
  const values = {
    "gpu-lexer": await bundledBytes(gpuLexer()),
    Shiki: await bundledBytes(shiki([language])),
    "Highlight.js": await bundledBytes(highlightJs([language])),
    "Prism.js": await bundledBytes(prism([language])),
    "Sugar High": await bundledBytes(sugar([language])),
    "Starry Night": await bundledBytes(starry([starryScope(language)])) + onigurumaBytes,
  };
  console.log(`${language}\t${Object.entries(values).map(([name, bytes]) => `${name}=${bytes}`).join("\t")}`);
}

async function bundledBytes(source) {
  const result = await build({
    bundle: true, format: "esm", minify: true, platform: "browser",
    stdin: { contents: source, resolveDir: process.cwd() }, write: false,
  });
  return result.outputFiles.reduce((total, file) => total + compressedBytes(file.contents), 0);
}

function compressedBytes(value) {
  return brotliCompressSync(value, {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
  }).byteLength;
}

function gpuLexer() {
  return "import { parse } from 'gpu-lexer'; globalThis.__size = parse;";
}

function shiki(languages) {
  const names = { shell: "shellscript" };
  const imports = languages.map((name, index) => `import l${index} from 'shiki/dist/langs/${names[name] ?? name}.mjs';`).join("\n");
  return `${imports}\nimport * as core from 'shiki/core'; import wasm from 'shiki/wasm'; ` +
    `globalThis.__size = [core, wasm, ${languages.map((_, index) => `l${index}`).join(",")}];`;
}

function highlightJs(languages) {
  const names = { html: "xml", shell: "bash" };
  const imports = languages.map((name, index) => `import l${index} from 'highlight.js/lib/languages/${names[name] ?? name}';`).join("\n");
  return `import core from 'highlight.js/lib/core';\n${imports}\n` +
    languages.map((name, index) => `core.registerLanguage('${name}', l${index});`).join("\n") +
    "\nglobalThis.__size = core;";
}

function prism(languages) {
  const names = { html: "markup", shell: "bash" };
  const imports = languages.filter((name) => name !== "javascript")
    .map((name) => `import 'prismjs/components/prism-${names[name] ?? name}';`).join("\n");
  return `import value from 'prismjs';\n${imports}\nglobalThis.__size = value;`;
}

function sugar(languages) {
  const imports = languages.map((name, index) => `import * as l${index} from 'sugar-high/lang/${name}';`).join("\n");
  return `import * as core from 'sugar-high/core';\n${imports}\nglobalThis.__size = [core, ${languages.map((_, index) => `l${index}`).join(",")}];`;
}

function starry(scopes) {
  const imports = scopes.map((scope, index) =>
    `import l${index} from '@wooorm/starry-night/${scope}';`).join("\n");
  return `import { createStarryNight } from '@wooorm/starry-night';\n${imports}\n` +
    `globalThis.__size = [createStarryNight, ${scopes.map((_, index) => `l${index}`).join(",")}];`;
}

function starryScope(language) {
  return {
    c: "source.c", cpp: "source.c++", css: "source.css", html: "text.html.basic",
    javascript: "source.js", markdown: "text.md", python: "source.python",
    ruby: "source.ruby", rust: "source.rust", shell: "source.shell", typescript: "source.ts",
  }[language];
}

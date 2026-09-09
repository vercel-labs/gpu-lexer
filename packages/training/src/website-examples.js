import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export const WEBSITE_EXAMPLE_SOURCE_LIMIT = 20_000;

const demoSourceUrl = new URL("../../../apps/website/app/demo.tsx", import.meta.url);
const apiExampleUrl = new URL("../../../apps/website/app/api-example.tsx", import.meta.url);
const probesUrl = new URL("../data/language-probes.json", import.meta.url);
let cachedExamples;
let cachedSourceKeys;

const shikiLanguages = Object.freeze({
  javascript: "javascript", jsx: "jsx", tsx: "tsx", vue: "vue", svelte: "svelte",
  typescript: "typescript", css: "css", scss: "scss", html: "html", xsl: "xsl",
  python: "python", go: "go", rust: "rust", zig: "zig", java: "java", kotlin: "kotlin",
  swift: "swift", "objective-c": "objective-c", php: "php", c: "c", csharp: "csharp",
  ruby: "ruby", dart: "dart", solidity: "solidity", cuda: "cpp", hlsl: "hlsl",
  glsl: "glsl", svg: "xml", json: "json", yaml: "yaml", cpp: "cpp", dockerfile: "docker",
  makefile: "make", cmake: "cmake", hcl: "hcl", astro: "astro", solidjs: "tsx",
  haskell: "haskell", markdown: "markdown", shell: "shellscript", powershell: "powershell",
});

const familyAliases = Object.freeze({
  jsx: "javascript", tsx: "typescript", scss: "css", xsl: "xslt",
  solidjs: "typescript", "angular-html": "html",
  "angular-ts": "typescript", less: "css",
});

export function readWebsiteExamples() {
  if (cachedExamples) return cachedExamples;
  const demoSource = readFileSync(demoSourceUrl, "utf8");
  const start = demoSource.indexOf("const demoFolders");
  const end = demoSource.indexOf("] as const", start);
  if (start < 0 || end < 0) throw new Error("cannot locate website demo file list");
  const block = demoSource.slice(start, end);
  const staticFiles = [...block.matchAll(
    /\{\s*id:\s*'([^']+)',\s*name:\s*'([^']+)',\s*language:\s*'([^']+)',\s*url:\s*'([^']+)',?\s*\}/g,
  )].map((match) => {
    const [, id, fileName, displayLanguage, url] = match;
    const shiki = shikiLanguages[displayLanguage];
    if (!shiki) throw new Error(`website demo ${id} has no Shiki language mapping for ${displayLanguage}`);
    return example({ id: `demo-${id}`, fileName, url, shiki,
      family: familyAliases[displayLanguage] ?? displayLanguage });
  });
  const quotedUrls = [...block.matchAll(/\burl:\s*'/g)].length;
  if (staticFiles.length !== quotedUrls) {
    throw new Error(`parsed ${staticFiles.length}/${quotedUrls} hard-coded website examples`);
  }

  const probes = JSON.parse(readFileSync(probesUrl, "utf8")).map((probe) => example({
    id: `probe-${probe.id}`, fileName: probe.fileName, url: probe.url, shiki: probe.shiki,
    family: familyAliases[probe.id] ?? probe.id,
  }));

  const apiSource = readFileSync(apiExampleUrl, "utf8").match(/const source = `([\s\S]*?)`/)?.[1];
  if (!apiSource) throw new Error("cannot locate highlighted website API example");
  const examples = [...staticFiles, ...probes, example({
    id: "api-example", fileName: "api-example.ts", inlineSource: apiSource,
    shiki: "typescript", family: "typescript",
  })];
  const ids = new Set(), locations = new Set();
  for (const item of examples) {
    if (ids.has(item.id)) throw new Error(`duplicate website example id: ${item.id}`);
    ids.add(item.id);
    const location = item.url ?? `inline:${item.id}`;
    if (locations.has(location)) throw new Error(`duplicate website example source: ${location}`);
    locations.add(location);
  }
  cachedExamples = Object.freeze(examples.map(Object.freeze));
  return cachedExamples;
}

export function websiteExampleFamilies() {
  return [...new Set(readWebsiteExamples().map(({ family }) => family))].sort();
}

export function websiteExampleFileName(item) {
  const safe = item.id.replaceAll(/[^a-z0-9_-]/gi, "-");
  const key = item.url ?? item.inlineSource;
  const digest = createHash("sha256").update(key).digest("hex").slice(0, 12);
  return `${safe}-${digest}.source`;
}

export function websiteExampleMatchesCandidate(item, candidate) {
  const source = item.provenance;
  return source && source.kind === candidate.origin && source.sourceName === candidate.sourceName &&
    source.path === candidate.path;
}

export function websiteExampleSourceKeys() {
  cachedSourceKeys ??= new Set(readWebsiteExamples().flatMap(({ provenance }) => provenance && provenance.kind !== "url"
    ? [`${provenance.kind}\0${provenance.sourceName}\0${provenance.path}`]
    : []));
  return cachedSourceKeys;
}

export function corpusRecordSourceKey(record) {
  return `${record.origin}\0${record.sourceName}\0${record.path}`;
}

export function isWebsiteVerificationSource(record) {
  return websiteExampleSourceKeys().has(corpusRecordSourceKey(record));
}

function example(value) {
  if (!value.id || !value.fileName || !value.shiki || !value.family || !(value.url || value.inlineSource)) {
    throw new Error(`invalid website example: ${value.id ?? "unknown"}`);
  }
  return { ...value, provenance: value.url ? parseSourceUrl(value.url) : null };
}

function parseSourceUrl(value) {
  const url = new URL(value);
  const parts = url.pathname.slice(1).split("/").map(decodeURIComponent);
  if (url.hostname === "raw.githubusercontent.com" && parts.length >= 4) {
    return { kind: "git", sourceName: `${parts[0]}/${parts[1]}`, revision: parts[2],
      path: parts.slice(3).join("/") };
  }
  if (url.hostname === "unpkg.com") {
    const slash = url.pathname.indexOf("/", 1);
    const secondSlash = url.pathname.startsWith("/@") ? url.pathname.indexOf("/", slash + 1) : slash;
    const specifier = decodeURIComponent(url.pathname.slice(1, secondSlash));
    const at = specifier.lastIndexOf("@");
    if (at > 0) return { kind: "npm", sourceName: specifier.slice(0, at), revision: specifier.slice(at + 1),
      path: decodeURIComponent(url.pathname.slice(secondSlash + 1)) };
  }
  return { kind: "url", sourceName: url.hostname, revision: "url", path: decodeURIComponent(url.pathname.slice(1)) };
}

import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";

import { readWebsiteExamples } from "./website-examples.js";
import { TREE_TOKENIZER_VERSION } from "../../core/src/constants.js";
import { taxonomyVersion } from "./classes.js";

export const corpusRoot = new URL("../data/generated/", import.meta.url);
export const manifestUrl = new URL("../data/corpus.json", import.meta.url);
export const popularityUrl = new URL("../data/language-popularity.json", import.meta.url);

const languagesByExtension = new Map([
  [".js", "javascript"], [".mjs", "javascript"], [".cjs", "javascript"], [".jsx", "jsx"],
  [".ts", "typescript"], [".mts", "typescript"], [".cts", "typescript"], [".tsx", "tsx"],
  [".py", "python"], [".java", "java"], [".cc", "cpp"], [".cpp", "cpp"], [".cxx", "cpp"],
  [".h", "c"], [".c", "c"], [".cs", "csharp"], [".php", "php"], [".sh", "shell"],
  [".bash", "shell"], [".zsh", "shell"], [".go", "go"], [".rs", "rust"], [".kt", "kotlin"],
  [".kts", "kotlin"], [".swift", "swift"], [".rb", "ruby"], [".html", "html"], [".htm", "html"],
  [".css", "css"], [".scss", "scss"], [".md", "markdown"], [".mdx", "mdx"],
  [".vue", "vue"], [".svelte", "svelte"], [".bat", "bat"], [".cmd", "bat"],
  [".ps1", "powershell"], [".psm1", "powershell"], [".psd1", "powershell"],
  [".cmake", "cmake"], [".lua", "lua"], [".dart", "dart"], [".asm", "asm"], [".s", "asm"],
  [".hcl", "hcl"], [".tf", "hcl"], [".tfvars", "hcl"], [".nix", "nix"],
  [".pl", "perl"], [".pm", "perl"], [".r", "r"], [".hh", "hack"], [".mako", "jinja"],
  [".hlsl", "hlsl"], [".glsl", "glsl"], [".vert", "glsl"], [".frag", "glsl"],
  [".shader", "shaderlab"], [".xsl", "xsl"], [".xslt", "xsl"], [".vim", "viml"],
  [".pyx", "python"], [".pxd", "python"], [".cu", "cpp"], [".cuh", "cpp"], [".mm", "objective-cpp"],
  [".m", "objective-c"], [".sql", "sql"], [".mao", "jinja"],
  [".f", "fortran-fixed-form"], [".for", "fortran-fixed-form"], [".f77", "fortran-fixed-form"],
  [".f90", "fortran-free-form"], [".f95", "fortran-free-form"], [".f03", "fortran-free-form"],
  [".sol", "solidity"], [".m4", "shellscript"], [".el", "emacs-lisp"], [".feature", "gherkin"],
  [".bzl", "python"], [".star", "python"], [".awk", "awk"], [".astro", "astro"],
  [".diff", "diff"], [".patch", "diff"],
  [".yml", "yaml"], [".yaml", "yaml"], [".zig", "zig"], [".hs", "haskell"], [".jl", "julia"],
  [".vb", "vb"], [".adb", "ada"], [".ads", "ada"], [".nim", "nim"], [".d", "d"], [".scala", "scala"],
  [".pp", "puppet"], [".hx", "haxe"], [".bicep", "bicep"], [".sv", "system-verilog"],
  [".svh", "system-verilog"], [".prisma", "prisma"], [".graphql", "graphql"], [".gql", "graphql"],
  [".proto", "proto"], [".c3", "c3"], [".c3i", "c3"], [".coffee", "coffeescript"],
  [".toml", "toml"], [".cshtml", "razor"], [".razor", "razor"],
  [".gd", "gdscript"], [".wgsl", "wgsl"], [".groovy", "groovy"], [".gradle", "groovy"],
  [".less", "less"], [".cls", "apex"], [".vala", "vala"], [".json", "json"], [".xml", "xml"], [".svg", "xml"],
]);

const languageFamilies = new Map([
  ["jsx", "javascript"], ["tsx", "typescript"], ["scss", "css"], ["mdx", "markdown"],
  ["shellscript", "shell"], ["docker", "dockerfile"], ["make", "makefile"], ["bat", "batchfile"],
  ["sql", "plpgsql"], ["asm", "assembly"], ["jinja", "mako"], ["xsl", "xslt"],
  ["viml", "vim-script"], ["fortran-fixed-form", "fortran"], ["fortran-free-form", "fortran"],
  ["system-verilog", "systemverilog"], ["proto", "protobuf"], ["less", "css"],
]);

const namedLanguages = new Map([
  ["dockerfile", ["docker", "dockerfile"]], ["makefile", ["make", "makefile"]],
  ["cmakelists.txt", ["cmake", "cmake"]], ["procfile", ["shellscript", "procfile"]],
  ["meson.build", ["python", "meson"]],
  ["build", ["python", "starlark"]], ["build.bazel", ["python", "starlark"]],
  ["workspace", ["python", "starlark"]], ["workspace.bzlmod", ["python", "starlark"]],
]);

const sampleLanguages = new Map([
  ["javascript", ["javascript", "javascript"]], ["python", ["python", "python"]],
  ["typescript", ["typescript", "typescript"]], ["shell", ["shellscript", "shell"]],
  ["dockerfile", ["docker", "dockerfile"]], ["docker", ["docker", "dockerfile"]],
  ["java", ["java", "java"]], ["c++", ["cpp", "cpp"]], ["cpp", ["cpp", "cpp"]], ["c", ["c", "c"]],
  ["makefile", ["make", "makefile"]], ["make", ["make", "makefile"]],
  ["batchfile", ["bat", "batchfile"]], ["batch", ["bat", "batchfile"]], ["php", ["php", "php"]],
  ["c#", ["csharp", "csharp"]], ["csharp", ["csharp", "csharp"]], ["powershell", ["powershell", "powershell"]],
  ["cmake", ["cmake", "cmake"]], ["kotlin", ["kotlin", "kotlin"]], ["ruby", ["ruby", "ruby"]],
  ["swift", ["swift", "swift"]], ["go", ["go", "go"]], ["plpgsql", ["sql", "plpgsql"]],
  ["rust", ["rust", "rust"]], ["objective-c", ["objective-c", "objective-c"]],
  ["objective_c", ["objective-c", "objective-c"]], ["lua", ["lua", "lua"]], ["dart", ["dart", "dart"]],
  ["procfile", ["shellscript", "procfile"]], ["assembly", ["asm", "assembly"]], ["nasm", ["asm", "assembly"]],
  ["hcl", ["hcl", "hcl"]], ["nix", ["nix", "nix"]], ["perl", ["perl", "perl"]], ["r", ["r", "r"]],
  ["hack", ["hack", "hack"]], ["mako", ["jinja", "mako"]], ["hlsl", ["hlsl", "hlsl"]],
  ["glsl", ["glsl", "glsl"]], ["shaderlab", ["shaderlab", "shaderlab"]], ["smarty", ["liquid", "smarty"]],
  ["xslt", ["xsl", "xslt"]], ["vim script", ["viml", "vim-script"]], ["viml", ["viml", "vim-script"]],
  ["matlab", ["matlab", "matlab"]], ["cython", ["python", "cython"]], ["cuda", ["cpp", "cuda"]],
  ["objective-c++", ["objective-cpp", "objective-cpp"]], ["objective_cpp", ["objective-cpp", "objective-cpp"]],
  ["fortran", ["fortran-free-form", "fortran"]], ["solidity", ["solidity", "solidity"]],
  ["meson", ["python", "meson"]], ["m4", ["shellscript", "m4"]], ["tsql", ["sql", "tsql"]],
  ["emacs lisp", ["emacs-lisp", "emacs-lisp"]], ["gherkin", ["gherkin", "gherkin"]],
  ["emacs", ["emacs-lisp", "emacs-lisp"]], ["plsql", ["sql", "plpgsql"]],
  ["starlark", ["python", "starlark"]], ["awk", ["awk", "awk"]], ["html", ["html", "html"]],
  ["css", ["css", "css"]], ["markdown", ["markdown", "markdown"]], ["vue", ["vue", "vue"]],
  ["svelte", ["svelte", "svelte"]],
  ["astro", ["astro", "astro"]],
  ["diff", ["diff", "diff"]], ["patch", ["diff", "diff"]],
  ["yaml", ["yaml", "yaml"]], ["zig", ["zig", "zig"]], ["haskell", ["haskell", "haskell"]],
  ["julia", ["julia", "julia"]], ["visual basic .net", ["vb", "vb"]], ["nim", ["nim", "nim"]],
  ["d", ["d", "d"]], ["scala", ["scala", "scala"]], ["puppet", ["puppet", "puppet"]],
  ["graphql", ["graphql", "graphql"]], ["nginx", ["nginx", "nginx"]], ["bicep", ["bicep", "bicep"]],
  ["systemverilog", ["system-verilog", "systemverilog"]], ["prisma", ["prisma", "prisma"]],
  ["coffeescript", ["coffeescript", "coffeescript"]], ["apex", ["apex", "apex"]],
  ["toml", ["toml", "toml"]],
  ["json", ["json", "json"]], ["xml", ["xml", "xml"]], ["groovy", ["groovy", "groovy"]],
  ["protocol buffer", ["proto", "protobuf"]], ["less", ["less", "css"]], ["wgsl", ["wgsl", "wgsl"]],
  ["gdscript", ["gdscript", "gdscript"]], ["html+razor", ["razor", "razor"]],
]);

export function languageFamily(language) {
  return languageFamilies.get(language) ?? language;
}

export function languageForPath(path) {
  const normalized = path.replaceAll("\\", "/");
  const sample = normalized.match(/(?:^|\/)samples\/([^/]+)(?:\/|$)/i)?.[1]
    ?? normalized.match(/(?:^|\/)spec\/visual\/samples\/([^/]+)$/i)?.[1]
    ?? normalized.match(/(?:^|\/)tests\/(?:examplefiles|snippets)\/([^/]+)(?:\/|$)/i)?.[1]
    ?? normalized.match(/(?:^|\/)(?:languages|markup|detect|testdata)\/([^/]+)(?:\/|$)/i)?.[1]
    ?? normalized.match(/(?:^|\/)lexers\/testdata\/([^/.]+)\.actual$/i)?.[1];
  if (sample) {
    const descriptor = sampleLanguages.get(sample.toLowerCase());
    if (descriptor) return { language: descriptor[0], family: descriptor[1] };
  }
  const basename = normalized.slice(normalized.lastIndexOf("/") + 1).toLowerCase();
  const named = namedLanguages.get(basename) ?? conventionalBuildFile(basename);
  if (named) return { language: named[0], family: named[1] };
  const extension = extnameFromPath(basename);
  // .d is also DTrace and Make dependencies, .pp also Pascal, .m also MATLAB
  // and Mathematica, .pl also Prolog, .f also Forth, .gd also GAP: inside a
  // sample directory for some other language, such an extension is not evidence.
  if (sample && sharedExtensions.has(extension)) return null;
  const grammar = languagesByExtension.get(extension);
  if (!grammar) return null;
  let family = languageFamily(grammar);
  if (grammar === "python" && /(?:\.pyx|\.pxd)$/i.test(basename)) family = "cython";
  else if (grammar === "python" && /(?:\.bzl|\.star)$/i.test(basename)) family = "starlark";
  else if (grammar === "cpp" && /(?:\.cu|\.cuh)$/i.test(basename)) family = "cuda";
  else if (grammar === "shellscript" && /\.m4$/i.test(basename)) family = "m4";
  return { language: grammar, family };
}

export const teacherLanguages = Object.freeze([...new Set([
  ...languagesByExtension.values(), ...[...namedLanguages.values()].map(([language]) => language),
  ...[...sampleLanguages.values()].map(([language]) => language),
])]);

// Dockerfile.alpine, base.dockerfile, Containerfile, GNUmakefile, Makefile.common, rules.mk.
// A suffix that names another format (Makefile.am, Dockerfile.template, Makefile.py) is not one.
const sharedExtensions = new Set([".d", ".pp", ".m", ".pl", ".pm", ".f", ".for", ".gd", ".cls"]);
const buildFileSuffixes = /^(?:am|in|md|txt|json|ya?ml|toml|lock|template|tmpl|tpl|j2|erb|dockerignore|orig|bak)$/;
function conventionalBuildFile(basename) {
  const extension = extnameFromPath(basename);
  if (extension === ".mk" || basename === "gnumakefile") return ["make", "makefile"];
  if (extension === ".dockerfile" || basename === "containerfile") return ["docker", "dockerfile"];
  const match = basename.match(/^(makefile|dockerfile|containerfile)[._-](.+)$/);
  if (!match) return null;
  const suffix = match[2].slice(match[2].lastIndexOf(".") + 1);
  if (buildFileSuffixes.test(suffix) || languagesByExtension.has(`.${suffix}`)) return null;
  return match[1] === "makefile" ? ["make", "makefile"] : ["docker", "dockerfile"];
}

function extnameFromPath(basename) {
  const index = basename.lastIndexOf(".");
  return index < 0 ? "" : basename.slice(index);
}

export async function readCorpusManifest() {
  return JSON.parse(await readFile(manifestUrl, "utf8"));
}

export async function readLanguagePopularity() {
  return JSON.parse(await readFile(popularityUrl, "utf8"));
}

export function validateCorpusManifest(manifest, popularity) {
  const splits = ["train", "verification", "mining"];
  const allowed = new Set(manifest.policy.allowedLicenses);
  const strataTotal = Object.values(manifest.policy.strata).reduce((sum, value) => sum + value, 0);
  if (Math.abs(strataTotal - 1) > 1e-9) throw new Error("corpus strata must sum to 1");
  const gitSeen = new Map();
  const npmSeen = new Map();
  const popularityFamilies = new Set(popularity.languages.map((entry) => entry.family ?? languageFamily(entry.shiki[0])));
  const samplingFamilies = new Set([...popularityFamilies, ...Object.keys(manifest.policy.trainingSamplingWeights ?? {})]);
  for (const [split, minimums] of Object.entries(manifest.policy.minimumLanguageTokens ?? {})) {
    if (!splits.includes(split) || !minimums || Object.entries(minimums).some(([language, minimum]) =>
      !teacherLanguages.includes(language) || !Number.isSafeInteger(minimum) || minimum < 1)) {
      throw new Error("minimumLanguageTokens must contain known languages and positive integer floors");
    }
  }

  for (const split of splits) {
    for (const entry of manifest[split].git) {
      validateLicense(entry, allowed);
      if (!/^[\da-f]{40}$/.test(entry.commit)) throw new Error(`${entry.repo} must pin a full commit SHA`);
      // A source may be limited to the families it was added for, so a large
      // repository does not also reshape the mix of every other language.
      if (entry.families !== undefined && (!Array.isArray(entry.families) || !entry.families.length ||
          entry.families.some((family) => !samplingFamilies.has(family)))) {
        throw new Error(`${entry.repo} families must be a non-empty list of weighted language families`);
      }
      if (gitSeen.has(entry.repo)) throw new Error(`${entry.repo} occurs in both ${gitSeen.get(entry.repo)} and ${split}`);
      gitSeen.set(entry.repo, split);
    }
    for (const entry of manifest[split].npm) {
      validateLicense(entry, allowed);
      if (!/^\d+\.\d+\.\d+(?:[-+].*)?$/.test(entry.version)) throw new Error(`${entry.name} must pin an exact version`);
      if (npmSeen.has(entry.name)) throw new Error(`${entry.name} occurs in both ${npmSeen.get(entry.name)} and ${split}`);
      npmSeen.set(entry.name, split);
    }
  }

  for (const language of manifest.policy.requiredLanguagesPerSplit) {
    if (!popularityFamilies.has(language)) throw new Error(`${language} has no popularity weight`);
  }
  const percent = popularity.languages.reduce((sum, entry) => sum + entry.percent, 0);
  if (Math.abs(percent - 100) > 0.01) throw new Error("language popularity percentages must sum to 100");
}

function validateLicense(entry, allowed) {
  if (!allowed.has(entry.license)) throw new Error(`${entry.repo ?? entry.name} uses disallowed license ${entry.license}`);
}

export function gitDirectoryName(entry) {
  return `${entry.repo.replaceAll("/", "__")}@${entry.commit}`;
}

export function packageDirectoryName(entry) {
  return `${entry.name.replaceAll("/", "__")}@${entry.version}`;
}

export function popularityWeights(popularity, supplemental = {}) {
  const weights = new Map();
  for (const entry of popularity.languages) {
    const families = [entry.family ?? languageFamily(entry.shiki[0])];
    for (const family of families) weights.set(family, (weights.get(family) ?? 0) + entry.percent / families.length);
  }
  for (const [family, weight] of Object.entries(supplemental)) weights.set(family, weight);
  return weights;
}

export function corpusConfigurationDigest(manifest, popularity, split) {
  const base = baseCorpusConfigurationDigest(manifest, popularity, split);
  if (split !== "verification") return base;
  return createHash("sha256").update(JSON.stringify({
    base,
    websiteExamples: readWebsiteExamples().map(({ id, fileName, url, inlineSource, shiki, family }) =>
      ({ id, fileName, url, inlineSource, shiki, family })),
  })).digest("hex");
}

export function baseCorpusConfigurationDigest(manifest, popularity, split) {
  const policy = structuredClone(manifest.policy);
  const languageMinimums = policy.minimumLanguageTokens?.[split];
  if (languageMinimums) policy.minimumLanguageTokens = { [split]: languageMinimums };
  else delete policy.minimumLanguageTokens;
  return createHash("sha256").update(JSON.stringify({
    labelSchemaVersion: 7,
    taxonomyVersion,
    tokenizerVersion: TREE_TOKENIZER_VERSION,
    selectionSchemaVersion: 2,
    policy,
    sources: manifest[split],
    languages: popularity.languages,
  })).digest("hex");
}

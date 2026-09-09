import { createReadStream } from "node:fs";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createGunzip } from "node:zlib";
import { resolve } from "node:path";
import readline from "node:readline";

import hljs from "highlight.js";
import Prism from "prismjs";
import loadPrismLanguages from "prismjs/components/index.js";
import { highlight as sugarHigh } from "sugar-high";
import { all as starryGrammars, createStarryNight } from "@wooorm/starry-night";
import { toHtml } from "hast-util-to-html";

import { classNames } from "../../training/src/classes.js";
import { loadFloatCheckpoint } from "../../training/src/checkpoint.js";
import { dequantizeTensors } from "../../training/src/quantization.js";
import { promotedRunId } from "../../training/src/promoted-run.generated.js";
import { alignTreeLabels } from "../../training/src/tree-label-alignment.js";
import { createTreeRecord, treeProbabilities } from "../../training/src/tree-model.js";
import { labelsFromHighlightedHtml } from "./html-labels.js";

const root = new URL("../../../", import.meta.url);
const popularityPath = new URL("packages/training/data/language-popularity.json", root);
const defaultVerificationPath = new URL("packages/training/data/generated/shards/verification.jsonl.gz", root);
const defaultOutputPath = new URL("apps/website/app/correctness.generated.ts", root);
const TOP = 25;

const languages = {
  javascript: { hljs: "javascript", prism: "javascript", sugar: "javascript" },
  python: { hljs: "python", prism: "python", sugar: "python" },
  typescript: { hljs: "typescript", prism: "typescript", sugar: "typescript" },
  shell: { hljs: "bash", prism: "bash", sugar: "shell" },
  dockerfile: { hljs: "dockerfile", prism: "docker", sugar: "dockerfile" },
  java: { hljs: "java", prism: "java", sugar: "java" },
  cpp: { hljs: "cpp", prism: "cpp", sugar: "cpp" },
  c: { hljs: "c", prism: "c", sugar: "c" },
  makefile: { hljs: "makefile", prism: "makefile" },
  batchfile: { hljs: "dos", prism: "batch" },
  php: { hljs: "php", prism: "php", sugar: "php" },
  csharp: { hljs: "csharp", prism: "csharp", sugar: "csharp" },
  powershell: { hljs: "powershell", prism: "powershell", sugar: "powershell" },
  cmake: { hljs: "cmake", prism: "cmake" },
  kotlin: { hljs: "kotlin", prism: "kotlin", sugar: "kotlin" },
  ruby: { hljs: "ruby", prism: "ruby", sugar: "ruby" },
  swift: { hljs: "swift", prism: "swift", sugar: "swift" },
  go: { hljs: "go", prism: "go", sugar: "go" },
  plpgsql: { hljs: "pgsql", prism: "sql", sugar: "sql" },
  rust: { hljs: "rust", prism: "rust", sugar: "rust" },
  "objective-c": { hljs: "objectivec", prism: "objectivec" },
  lua: { hljs: "lua", prism: "lua", sugar: "lua" },
  dart: { hljs: "dart", prism: "dart" },
  procfile: { hljs: "bash", prism: "bash", sugar: "shell" },
  assembly: { hljs: "x86asm", prism: "nasm" },
};
const starryScopes = {
  javascript: "source.js", python: "source.python", typescript: "source.ts",
  shell: "source.shell", dockerfile: "source.dockerfile", java: "source.java",
  cpp: "source.c++", c: "source.c", makefile: "source.makefile",
  batchfile: "source.batchfile", php: "text.html.php", csharp: "source.cs",
  powershell: "source.powershell", cmake: "source.cmake", kotlin: "source.kotlin",
  ruby: "source.ruby", swift: "source.swift", go: "source.go", plpgsql: "source.sql",
  rust: "source.rust", "objective-c": "source.objc", lua: "source.lua",
  dart: "source.dart", procfile: "source.procfile", assembly: "source.assembly",
};

const popularity = JSON.parse(await readFile(popularityPath, "utf8"));
const topLanguages = popularity.languages.filter(({ supplemental }) => !supplemental).slice(0, TOP);
const topFamilies = new Set(topLanguages.map(({ family }) => family));
if (topLanguages.length !== TOP || topLanguages.some(({ family }) => !languages[family])) {
  throw new Error("top-25 language mappings are incomplete");
}
const starryNight = await createStarryNight(selectStarryGrammars());

loadPrismLanguages([...new Set(Object.values(languages).flatMap(({ prism }) => prism ? [prism] : [])), "jsx", "tsx"]);

const runArgument = argument("--run") ?? promotedRunId;
const outputPath = argument("--output") ?? defaultOutputPath;
const checkpoint = await loadFloatCheckpoint(runArgument);
const verificationPath = argument("--verification") ?? checkpoint.metadata.config?.verificationShard ??
  defaultVerificationPath;
const verificationSha256 = createHash("sha256").update(await readFile(verificationPath)).digest("hex");
if (verificationSha256 !== checkpoint.metadata.corpus.verification.sha256) {
  throw new Error("correctness benchmark verification corpus does not match the checkpoint");
}
const entries = await readdir(checkpoint.path);
const bits = checkpoint.metadata.quantization.bits;
const weightName = entries.find((name) => new RegExp(`^weights-int${bits}-.+\\.bin$`).test(name));
if (!weightName) throw new Error(`stored int${bits} weights are missing`);
const model = dequantizeTensors(await readFile(resolve(checkpoint.path, weightName)),
  checkpoint.metadata.quantization);
const shape = {
  hiddenSize: checkpoint.metadata.hiddenSize,
  classifierSize: checkpoint.metadata.architecture.classifierDimensions,
  hashBuckets: checkpoint.metadata.architecture.lexemeHashBuckets,
};
const scores = Object.fromEntries(["gpu-lexer", "highlight.js", "prism.js", "sugar-high", "starry-night"]
  .map((engine) => [engine, Object.fromEntries(topLanguages.map(({ family }) => [family, { correct: 0, total: 0 }]))]));
const input = createReadStream(verificationPath).pipe(createGunzip());
const lines = readline.createInterface({ input, crlfDelay: Infinity });
let files = 0;

for await (const line of lines) {
  if (!line) continue;
  const item = JSON.parse(line);
  if (!topFamilies.has(item.family)) continue;
  files++;
  const { record } = createTreeRecord(item, shape.hashBuckets, {
    featureVersion: checkpoint.metadata.featureVersion,
    retainSource: true,
  });
  const probabilities = treeProbabilities(model, record, shape);
  scorePredictions(scores["gpu-lexer"][item.family], record,
    probabilities.map((values) => classNames[argmax(values)]));
  for (const engine of ["highlight.js", "prism.js", "sugar-high", "starry-night"]) {
    const language = languageFor(engine, item);
    const score = scores[engine][item.family];
    if (!language) {
      score.total += supervisedParts(record);
      continue;
    }
    const html = highlight(engine, item.source, language);
    const labels = labelsFromHighlightedHtml(item.source, html, engine, { family: item.family });
    const aligned = alignTreeLabels(labels, record.ranges, { sourceLength: item.source.length });
    scorePredictions(score, record, aligned.map(({ class: name }) => name));
  }
}

const totalPushers = topLanguages.reduce((sum, { pushers }) => sum + pushers, 0);
const rows = [
  { label: "Shiki", value: 100, coverage: 100 },
  ...Object.entries(scores).map(([label, families]) => {
    let value = 0, coverage = 0;
    for (const language of topLanguages) {
      const weight = language.pushers / totalPushers;
      const score = families[language.family];
      value += weight * score.correct / Math.max(1, score.total);
      if (label === "gpu-lexer" || label === "starry-night" ||
          languages[language.family][engineKey(label)]) coverage += weight;
    }
    return {
      label: { "gpu-lexer": "gpu-lexer", "highlight.js": "Highlight.js",
        "prism.js": "Prism.js", "sugar-high": "Sugar High", "starry-night": "Starry Night" }[label],
      value: value * 100,
      coverage: coverage * 100,
      featured: label === "gpu-lexer",
    };
  }),
].sort((left, right) => right.value - left.value);

const generated = {
  runId: checkpoint.metadata.runId,
  normalizationVersion: 3,
  generatedAt: new Date().toISOString(),
  period: popularity.source.period,
  source: popularity.source.page,
  verificationSha256,
  languages: topLanguages.map(({ github }) => github),
  files,
  rows,
};
await writeFile(outputPath,
  `// Generated by \`pnpm benchmark:correctness\`. Do not edit by hand.\n` +
  `export const correctnessComparison = Object.freeze(${JSON.stringify(generated, null, 2)});\n`);

console.table(rows.map(({ label, value, coverage }) => ({
  library: label, correctness: `${value.toFixed(2)}%`, coverage: `${coverage.toFixed(2)}%`,
})));
console.log(`evaluated ${files} held-out files; website data: ${outputPath.pathname ?? outputPath}`);

function argument(name) {
  const index = process.argv.indexOf(name);
  return index < 0 ? null : process.argv[index + 1];
}

function languageFor(engine, item) {
  if (engine === "starry-night") {
    return item.language === "tsx" ? "source.tsx" : starryScopes[item.family];
  }
  if (engine === "prism.js" && item.language === "tsx") return "tsx";
  if (engine === "prism.js" && item.language === "jsx") return "jsx";
  return languages[item.family][engineKey(engine)];
}

function engineKey(engine) {
  return engine === "highlight.js" ? "hljs" : engine === "prism.js" ? "prism" :
    engine === "sugar-high" ? "sugar" : engine === "gpu-lexer" ? "gpu" : "shiki";
}

function highlight(engine, source, language) {
  if (engine === "highlight.js") return hljs.highlight(source, { language, ignoreIllegals: true }).value;
  if (engine === "prism.js") return Prism.highlight(source, Prism.languages[language], language);
  if (engine === "starry-night") return toHtml(starryNight.highlight(source, language));
  return sugarHigh(source, { lang: language });
}

function selectStarryGrammars() {
  const byScope = new Map(starryGrammars.map((grammar) => [grammar.scopeName, grammar]));
  const selected = new Map();
  const add = (scope) => {
    if (selected.has(scope)) return;
    const grammar = byScope.get(scope);
    if (!grammar) throw new Error(`Starry Night grammar is missing ${scope}`);
    selected.set(scope, grammar);
    for (const dependency of grammar.dependencies ?? []) add(dependency);
  };
  for (const scope of [...Object.values(starryScopes), "source.tsx"]) add(scope);
  return [...selected.values()];
}

function scorePredictions(score, record, predicted) {
  for (let index = 0; index < record.targets.length; index++) {
    if (!record.supervisionWeights[index]) continue;
    score.total++;
    if (predicted[index] === classNames[record.targets[index]]) score.correct++;
  }
}

function supervisedParts(record) {
  let total = 0;
  for (const weight of record.supervisionWeights) if (weight) total++;
  return total;
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index++) if (values[index] > values[best]) best = index;
  return best;
}

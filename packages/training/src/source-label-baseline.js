import { readFile, stat } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { createLabeler } from "./label.js";
import { alignTreeLabels, corpusSourceLabels } from "./tree-label-alignment.js";
import { auditSourceLabels } from "./source-label-audit.js";
import { detectSyntaxConstructs } from "./syntax-constructs.js";
import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";

export const representativeManifest = fileURLToPath(new URL("../test/fixtures/source-labels/manifest.json", import.meta.url));

/** Representation-only baseline. No weights, training, shard generation or model inference. */
export function auditLabelRecord(item) {
  const { sourceLabels, ...provenance } = corpusSourceLabels(item);
  const prepared = prepareTreeSource(item.source);
  try {
    const ranges = prepared[1];
    const direct = alignTreeLabels(sourceLabels, ranges, { sourceLength: item.source.length });
    const plain = item.source.length ? [{ from: 0, to: item.source.length, class: "plain" }] : [];
    const score = (spans) => auditSourceLabels(item.source, sourceLabels, spans);
    return {
      ...provenance,
      teacherSpans: sourceLabels.length, parts: direct.length,
      constructs: detectSyntaxConstructs(item.source, item.language, item.family),
      directTreeProjection: score(direct),
      plainBaseline: score(plain),
    };
  } finally { releaseTreePrepared(prepared); }
}

export async function auditRepresentativeFiles({ manifest = representativeManifest, maxFiles = 32, maxBytes = 65536 } = {}) {
  for (const [name, value] of Object.entries({ maxFiles, maxBytes })) {
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  }
  manifest = resolve(manifest);
  const entries = JSON.parse(await readFile(manifest, "utf8"));
  if (!Array.isArray(entries)) throw new Error("audit manifest must be an array of {path, language, family?}");
  const records = [], skipped = [], malformed = [];
  let labeler;
  try {
    // Load all supported teachers together so Markdown fences can resolve embedded grammars.
    const { teacherLanguages } = await import("./corpus.js");
    labeler = await createLabeler({ langs: teacherLanguages });
    for (const [index, entry] of entries.entries()) {
      const identity = { index, path: entry?.path ?? null, language: entry?.language ?? null };
      if (index >= maxFiles) { skipped.push({ ...identity, reason: "max-files" }); continue; }
      if (!entry || typeof entry.path !== "string" || typeof entry.language !== "string") {
        malformed.push({ ...identity, reason: "expected path and language strings" }); continue;
      }
      const path = resolve(dirname(manifest), entry.path);
      let source;
      try {
        const info = await stat(path);
        if (!info.isFile() || info.size > maxBytes) {
          skipped.push({ ...identity, reason: info.isFile() ? "max-bytes" : "not-file", bytes: info.size }); continue;
        }
        const bytes = await readFile(path);
        if (bytes.length > maxBytes) { skipped.push({ ...identity, reason: "max-bytes" }); continue; }
        source = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
      } catch (error) { skipped.push({ ...identity, reason: error.message }); continue; }
      try {
        const sourceLabels = labeler.labelSource(source, entry.language);
        const audit = auditLabelRecord({ source, sourceLabels, sourceLabelsVersion: 1, language: entry.language, family: entry.family });
        records.push({
          ...identity, absolutePath: path, family: entry.family ?? entry.language,
          sha256: createHash("sha256").update(source).digest("hex"), sourceLength: source.length,
          ...audit,
        });
      } catch (error) { malformed.push({ ...identity, reason: error.message }); }
    }
  } finally { labeler?.dispose(); }
  const incomplete = records.filter((record) => record.directTreeProjection.teacherCoverage.unlabeled > 0);
  return {
    diagnostic: "representative-source-label-baseline", readOnly: true, isModelScore: false, isFullCorpusScore: false,
    teacher: { name: "shiki", version: createRequire(import.meta.url)("shiki/package.json").version, theme: "github-dark-default", sourceLabelsVersion: 1 },
    sample: { manifest, selection: "explicit-local-files", maxFiles, maxBytes, requested: entries.length, audited: records.length },
    groups: { direct: records },
    skipped, malformed, incompleteLabels: incomplete.map(({ path, directTreeProjection }) => ({ path, ...directTreeProjection.teacherCoverage })),
    ok: records.length > 0 && skipped.length === 0 && malformed.length === 0 && incomplete.length === 0,
    notes: [
      "The direct tree projection and plain baseline use the same original Shiki UTF-16 target grid; these are alignment baselines, not learned-model accuracy or capacity ceilings.",
      "Plain and majority baselines, character confusion, exact styled runs and boundaries exclude ASCII whitespace and unlabeled target units; teacherCoverage reports omitted non-whitespace explicitly.",
      "Macro F1 includes all eight styled classes, including absent classes. Missing predictions count as plain and separately reduce predictionCoverage.",
      "Construct detection is heuristic metadata, not proof of teacher embedded-language correctness; representative fixtures assert selected semantic spans separately.",
    ],
  };
}

function argumentsToOptions(args) {
  const options = {};
  for (let index = 0; index < args.length; index++) {
    if (args[index] === "--") continue;
    if (args[index] === "--help") return null;
    const names = { "--manifest": "manifest", "--max-files": "maxFiles", "--max-bytes": "maxBytes" };
    const name = names[args[index]];
    if (!name || !args[index + 1] || args[index + 1].startsWith("--")) throw new Error(`invalid audit argument ${args[index]}`);
    const value = args[++index];
    options[name] = name === "manifest" ? value : Number(value);
  }
  return options;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const options = argumentsToOptions(process.argv.slice(2));
    if (!options) console.log("node src/source-label-baseline.js [--manifest files.json] [--max-files 32] [--max-bytes 65536]");
    else {
      const report = await auditRepresentativeFiles(options);
      console.log(JSON.stringify(report, null, 2));
      if (!report.ok) process.exitCode = 1;
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}

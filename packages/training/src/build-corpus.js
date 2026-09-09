import { createReadStream, createWriteStream } from "node:fs";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { relative, sep } from "node:path";
import { pipeline } from "node:stream/promises";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
import { createGunzip, createGzip } from "node:zlib";
import readline from "node:readline";

import { createLabeler, sourceParts } from "./label.js";
import { auditRepresentativeFiles } from "./source-label-baseline.js";
import { clipSourceLabels } from "./tree-label-alignment.js";
import { minifySource } from "./minify-source.js";
import { classifyGitPath, isVendoredPath, selectCorpus } from "./corpus-selection.js";
import { detectSyntaxConstructs } from "./syntax-constructs.js";
import {
  corpusRoot, gitDirectoryName, languageForPath, packageDirectoryName, teacherLanguages,
  baseCorpusConfigurationDigest, corpusConfigurationDigest, popularityWeights,
  readCorpusManifest, readLanguagePopularity,
  validateCorpusManifest,
} from "./corpus.js";
import {
  readWebsiteExamples, WEBSITE_EXAMPLE_SOURCE_LIMIT, websiteExampleFileName,
  websiteExampleMatchesCandidate,
} from "./website-examples.js";

const [manifest, popularity] = await Promise.all([readCorpusManifest(), readLanguagePopularity()]);
const websiteExamples = readWebsiteExamples();
const execute = promisify(execFile);
validateCorpusManifest(manifest, popularity);
console.log("audit representative direct Shiki labels before corpus rebuild");
const labelAudit = await auditRepresentativeFiles();
if (!labelAudit.ok) throw new Error(`representative label audit failed: ${JSON.stringify({ skipped: labelAudit.skipped, malformed: labelAudit.malformed, incomplete: labelAudit.incompleteLabels })}`);
const output = new URL("shards/", corpusRoot);
await mkdir(output, { recursive: true });
const labeler = await createLabeler({
  langs: [...new Set([...teacherLanguages, ...websiteExamples.map(({ shiki }) => shiki)])],
});
const weights = popularityWeights(popularity, manifest.policy.supplementalSamplingWeights);
const requestedSplit = parseSplit(process.argv.slice(2));
const previousSummary = requestedSplit ? await readPreviousSummary() : null;
const summary = {
  ...previousSummary,
  schemaVersion: 6,
  generatedAt: new Date().toISOString(),
  labeler: { name: "shiki", version: "4.4.3", taxonomyVersion: 4 },
  labelAudit: { teacher: labelAudit.teacher, files: labelAudit.groups.direct.map(({ path, sha256 }) => ({ path, sha256 })) },
  popularity: popularity.source,
  policy: manifest.policy,
  splits: previousSummary?.splits ?? {},
};

try {
  for (const split of requestedSplit ? [requestedSplit] : ["train", "verification", "mining"]) {
    console.log(`scan ${split} corpus`);
    const reusableBase = split === "verification"
      ? await reusableVerificationBase(previousSummary)
      : null;
    const candidates = reusableBase
      ? await collectWebsiteExamples()
      : await collectCandidates(split, manifest[split]);
    if (split === "mining") candidates.push(...await createMinifiedCandidates(candidates, split));
    if (split === "train") {
      candidates.push(...await createMinifiedCandidates(
        candidates.filter(({ origin }) => origin === "npm"), split,
      ));
    }
    console.log(`label ${split}: ${candidates.length} candidate files`);
    const labeled = [];
    const rejected = [];
    let index = 0;
    for (const candidate of candidates) {
      const rawSource = candidate.generatedSource ?? await readFile(candidate.url, "utf8");
      const source = candidate.websiteExample
        ? rawSource.slice(0, WEBSITE_EXAMPLE_SOURCE_LIMIT)
        : rawSource;
      try {
        const sourceLabels = labeler.labelSource(source, candidate.language);
        const parts = sourceParts(source, sourceLabels);
        if (parts.length) {
          const constructs = detectSyntaxConstructs(source, candidate.language, candidate.family);
          labeled.push({ ...candidate, source, parts, sourceLabels, constructs,
            priority: syntaxPriority(candidate, constructs,
              split === "train" ? manifest.policy.constructSamplingWeights : undefined) });
        }
      } catch (error) {
        if (candidate.websiteExample) {
          throw new Error(`website verification example ${candidate.websiteExample} could not be labeled: ${error.message}`, { cause: error });
        }
        rejected.push({ sourceName: candidate.sourceName, path: candidate.path, language: candidate.language, error: error.message });
        console.warn(`skip ${split}/${candidate.sourceName}:${candidate.path}: ${error.message}`);
      }
      index += 1;
      if (index === 1 || index % 100 === 0 || index === candidates.length) {
        console.log(`label ${split}: ${index}/${candidates.length}`);
      }
    }

    const required = labeled.filter(({ websiteExample }) => websiteExample);
    const regular = labeled.filter(({ websiteExample }) => !websiteExample);
    const unique = deduplicateLabeled(regular);
    console.log(`deduplicate ${split}: ${regular.length - unique.length} exact/structural source-form copies removed`);
    unique.sort((left, right) => right.priority - left.priority || candidateOrder(left, right));
    const baseSelected = reusableBase?.records ?? (split === "mining"
      ? selectMiningCorpus(
          unique,
          manifest.policy.targetTokens.mining,
          manifest.policy.miningMinifiedFraction ?? 0,
          manifest.mining.minimumFamilyTokens ?? 0,
        )
      : selectCorpus(unique, manifest.policy, split, split === "train"
        ? popularityWeights(popularity, manifest.policy.trainingSamplingWeights ?? manifest.policy.supplementalSamplingWeights)
        : weights, { minifiedFraction: manifest[split].minifiedCompiledFraction ?? 0 }));
    const baseSummary = reusableBase?.summary ?? summarize(baseSelected);
    validateSelected(split, baseSummary, manifest.policy, manifest[split]);
    const selected = [...baseSelected, ...required];
    const splitSummary = {
      configurationSha256: corpusConfigurationDigest(manifest, popularity, split),
      ...(split === "verification" ? {
        baseConfigurationSha256: baseCorpusConfigurationDigest(manifest, popularity, split),
      } : {}),
      ...summarize(selected), rejected,
      ...(split === "verification" ? {
        base: baseSummary,
        websiteExamples: {
          files: required.length,
          tokens: required.reduce((sum, record) => sum + record.parts.length, 0),
          ids: required.map(({ websiteExample }) => websiteExample),
        },
      } : {}),
    };
    await writeShard(split, selected);
    summary.splits[split] = splitSummary;
    console.log(`write ${split}: ${splitSummary.files} files, ${splitSummary.tokens} tokens, ${formatMix(splitSummary.strata)}`);
  }
} finally {
  labeler.dispose();
}

async function reusableVerificationBase(summary) {
  const previous = summary?.splits?.verification;
  const expected = baseCorpusConfigurationDigest(manifest, popularity, "verification");
  const actual = previous?.baseConfigurationSha256 ?? previous?.configurationSha256;
  if (actual !== expected) return null;
  const path = new URL("verification.jsonl.gz", output);
  const records = [];
  const lines = readline.createInterface({
    input: createReadStream(path).pipe(createGunzip()), crlfDelay: Infinity,
  });
  for await (const line of lines) {
    if (!line) continue;
    const record = JSON.parse(line);
    if (record.origin !== "website") records.push(record);
  }
  const summaryFromShard = summarize(records);
  if (summaryFromShard.tokens !== manifest.policy.targetTokens.verification) return null;
  console.log(`reuse immutable verification base: ${summaryFromShard.files} files / ${summaryFromShard.tokens} tokens`);
  return { records, summary: summaryFromShard };
}

await writeFile(new URL("summary.json", output), `${JSON.stringify(summary, null, 2)}\n`);
await writeFile(new URL("../data/corpus-summary.json", import.meta.url), `${JSON.stringify(summary, null, 2)}\n`);

async function readPreviousSummary() {
  try { return JSON.parse(await readFile(new URL("summary.json", output), "utf8")); }
  catch (error) { if (error.code === "ENOENT") return null; throw error; }
}

function parseSplit(arguments_) {
  const index = arguments_.indexOf("--split");
  if (index < 0) return null;
  const split = arguments_[index + 1];
  if (!["train", "verification", "mining"].includes(split)) throw new Error("--split must be train, verification, or mining");
  return split;
}

async function collectCandidates(split, sources) {
  const candidates = [];
  for (const entry of sources.git) {
    const root = new URL(`${gitDirectoryName(entry)}/`, new URL("repositories/", corpusRoot));
    await assertProvenance(root, { repo: entry.repo, commit: entry.commit });
    const found = await findCandidates(root, manifest.policy.maxFileBytes, "git", split, entry);
    const maximum = split === "mining"
      ? (manifest.policy.maxFilesPerMiningRepository ?? manifest.policy.maxFilesPerRepository)
      : manifest.policy.maxFilesPerRepository;
    const limited = limitRepository(found, maximum, split);
    candidates.push(...limited);
    console.log(`scan ${split}/${entry.repo}: ${found.length} eligible, ${limited.length} candidates`);
  }
  for (const entry of sources.npm) {
    const root = new URL(`${packageDirectoryName(entry)}/package/`, new URL("packages/", corpusRoot));
    const packageJson = JSON.parse(await readFile(new URL("package.json", root), "utf8"));
    if (packageJson.name !== entry.name || packageJson.version !== entry.version) {
      throw new Error(`download does not match ${entry.name}@${entry.version}`);
    }
    const found = await findCandidates(root, manifest.policy.maxFileBytes, "npm", split, entry);
    found.sort(candidateOrder);
    const maximum = split === "train"
      ? (manifest.policy.maxFilesPerTrainingNpmPackage ?? manifest.policy.maxFilesPerNpmPackage)
      : manifest.policy.maxFilesPerNpmPackage;
    const limited = found.slice(0, maximum);
    candidates.push(...limited);
    console.log(`scan ${split}/${entry.name}: ${found.length} eligible, ${limited.length} candidates`);
  }
  const filtered = candidates.filter((candidate) => !websiteExamples.some(
    (example) => websiteExampleMatchesCandidate(example, candidate),
  ));
  if (filtered.length !== candidates.length) {
    console.log(`reserve ${candidates.length - filtered.length} ${split} paths for website verification`);
  }
  if (split === "verification") filtered.push(...await collectWebsiteExamples());
  return filtered.sort(candidateOrder);
}

async function collectWebsiteExamples() {
  const root = new URL("website-examples/", corpusRoot);
  const candidates = [];
  for (const entry of websiteExamples) {
    const name = websiteExampleFileName(entry);
    const provenance = JSON.parse(await readFile(new URL(`${name}.json`, root), "utf8"));
    if (provenance.id !== entry.id || provenance.url !== (entry.url ?? null) ||
        provenance.shiki !== entry.shiki || provenance.family !== entry.family) {
      throw new Error(`website example provenance mismatch: ${entry.id}`);
    }
    candidates.push({
      url: new URL(name, root), origin: "website", stratum: "website", split: "verification",
      language: entry.shiki, family: entry.family, path: entry.fileName,
      sourceName: `website:${entry.id}`, revision: provenance.sha256,
      license: "verification-only", websiteExample: entry.id,
    });
  }
  return candidates;
}

async function createMinifiedCandidates(candidates, split) {
  const augmented = [];
  for (const candidate of candidates) {
    if (!["javascript", "jsx", "typescript", "tsx", "css", "scss", "html"].includes(candidate.language)) continue;
    const source = await readFile(candidate.url, "utf8");
    try {
      const minified = await minifySource(source, candidate.language, candidate.path);
      if (!minified || minified === source || minified.length < 16) continue;
      augmented.push({
        ...candidate,
        generatedSource: minified,
        minified: true,
        origin: split === "train" ? candidate.origin : "generated",
        stratum: split === "train" ? "compiled" : "minified",
        path: `${candidate.path}.minified`,
      });
    } catch (error) {
      console.warn(`skip minification ${candidate.sourceName}:${candidate.path}: ${error.message}`);
    }
  }
  console.log(`augment ${split}: ${augmented.length} minified JS/CSS/HTML candidates`);
  return augmented;
}

async function assertProvenance(root, expected) {
  let actual;
  try { actual = JSON.parse(await readFile(new URL("provenance.json", root), "utf8")); }
  catch (error) { throw new Error(`missing repository corpus; run corpus:fetch first`, { cause: error }); }
  for (const [key, value] of Object.entries(expected)) {
    if (actual[key] !== value) throw new Error(`repository provenance mismatch for ${value}`);
  }
}

async function findCandidates(root, maxFileBytes, origin, split, entry) {
  const { stdout } = await execute("find", [root.pathname, "-type", "f", "-size", "+0c", "-size", `-${maxFileBytes + 1}c`, "-print0"], {
    encoding: "buffer",
    maxBuffer: 32 * 1024 * 1024,
  });
  const paths = stdout.toString("utf8").split("\0").filter(Boolean);
  const candidates = [];
  for (const pathname of paths) {
    const path = relative(root.pathname, pathname).split(sep).join("/");
    if (origin === "git" && isExcludedDirectory(path)) continue;
    const descriptor = languageForPath(path);
    if (!descriptor || isGeneratedFile(path)) continue;
    const { language, family } = descriptor;
    const url = pathToFileURL(pathname);
    const stratum = origin === "npm" ? "compiled" : classifyGitPath(path, language);
    if (!stratum) continue;
    candidates.push({
      url, origin, stratum, split, language, family, path,
      sourceName: entry.repo ?? entry.name, revision: entry.commit ?? entry.version, license: entry.license,
    });
  }
  return candidates;
}

function isExcludedDirectory(path) {
  return isVendoredPath(path);
}

function isGeneratedFile(path) {
  return /(?:\.min|\.bundle|\.generated|\.d)\.[cm]?[jt]sx?$/.test(path.toLowerCase()) || /(?:^|\/)package-lock\.json$/.test(path);
}

function limitRepository(candidates, maximum, split) {
  const groups = new Map();
  for (const candidate of candidates) {
    const key = `${candidate.stratum}:${candidate.family}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(candidate);
  }
  for (const group of groups.values()) group.sort((left, right) =>
    (split === "mining" ? Number(isMinifiable(right)) - Number(isMinifiable(left)) : 0) || candidateOrder(left, right));
  const selected = [];
  while (selected.length < maximum) {
    let changed = false;
    for (const group of groups.values()) {
      const next = group.shift();
      if (!next) continue;
      selected.push(next);
      changed = true;
      if (selected.length === maximum) break;
    }
    if (!changed) break;
  }
  return selected;
}

function isMinifiable(candidate) {
  return ["javascript", "jsx", "typescript", "tsx", "css", "scss", "html"].includes(candidate.language);
}

function candidateOrder(left, right) {
  return left.sourceName.localeCompare(right.sourceName) || left.path.localeCompare(right.path);
}

function syntaxPriority(candidate, constructs, weights = {}) {
  return constructs.reduce((score, construct) => score + (weights[construct] ?? 1), 0) +
    Number(candidate.family === "diff");
}

function deduplicateLabeled(records) {
  const exact = new Set();
  const structural = new Set();
  return records.filter((record) => {
    const form = record.minified ? "minified" : "source";
    const exactKey = createHash("sha256")
      .update(record.family).update("\0")
      .update(form).update("\0")
      .update(record.source.replaceAll("\r\n", "\n").trim())
      .digest("base64");
    if (exact.has(exactKey)) return false;
    exact.add(exactKey);
    if (record.source.length >= 128 && record.parts.length >= 16) {
      const partSource = record.parts
        .filter((part) => part.class !== "comment")
        .map((part) => part.value)
        .join("\0");
      const structuralKey = createHash("sha256")
        .update(record.family).update("\0")
        .update(form).update("\0")
        .update(partSource)
        .digest("base64");
      if (structural.has(structuralKey)) return false;
      structural.add(structuralKey);
    }
    return true;
  });
}

async function writeShard(split, records) {
  const temporary = new URL(`${split}.jsonl`, output);
  const writer = createWriteStream(temporary);
  const written = new Promise((resolve, reject) => writer.once("finish", resolve).once("error", reject));
  for (const record of records) {
    const item = {
      split, origin: record.origin, stratum: record.stratum, sourceName: record.sourceName,
      revision: record.revision, license: record.license, path: record.path, language: record.language,
      family: record.family,
      constructs: detectSyntaxConstructs(record.source, record.language, record.family),
      source: record.source,
      // Additive per-item version: partial builds may coexist with older shards.
      sourceLabelsVersion: 1,
      sourceLabels: clipSourceLabels(record.sourceLabels, record.source.length),
    };
    if (!writer.write(`${JSON.stringify(item)}\n`)) await new Promise((resolve) => writer.once("drain", resolve));
  }
  writer.end();
  await written;
  await pipeline(createReadStream(temporary), createGzip({ level: 9 }), createWriteStream(new URL(`${split}.jsonl.gz`, output)));
  await unlink(temporary);
}

function summarize(records) {
  const result = { files: records.length, bytes: 0, tokens: 0, origins: {}, strata: {}, forms: {}, languages: {}, languageFamilies: {}, constructs: {}, classes: {}, sources: {}, supervision: { confidenceTotal: 0, downweightedTokens: 0 } };
  for (const record of records) {
    result.bytes += record.source.length;
    result.tokens += record.parts.length;
    increment(result.origins, record.origin, record.parts.length);
    increment(result.strata, record.stratum, record.parts.length);
    increment(result.forms, record.minified ? "minified" : "original", record.parts.length);
    increment(result.languages, record.language, record.parts.length);
    increment(result.languageFamilies, record.family, record.parts.length);
    increment(result.sources, record.sourceName, record.parts.length);
    for (const construct of detectSyntaxConstructs(record.source, record.language, record.family)) {
      increment(result.constructs, construct, record.parts.length);
    }
    for (const part of record.parts) {
      increment(result.classes, part.class, 1);
      const confidence = (part.confidence ?? 255) / 255;
      result.supervision.confidenceTotal += confidence;
      result.supervision.downweightedTokens += Number(confidence < 1);
    }
  }
  result.supervision.meanConfidence = result.supervision.confidenceTotal / Math.max(1, result.tokens);
  delete result.supervision.confidenceTotal;
  return result;
}

function selectMiningCorpus(records, targetTokens, minifiedFraction, minimumFamilyTokens) {
  const minifiedTarget = Math.round(targetTokens * minifiedFraction);
  const minified = selectMiningPool(records.filter(({ stratum }) => stratum === "minified"), minifiedTarget);
  const regular = selectMiningPool(
    records.filter(({ stratum }) => stratum !== "minified"),
    targetTokens - minifiedTarget,
    minimumFamilyTokens,
  );
  return interleaveRecords(minified, regular);
}

function selectMiningPool(records, targetTokens, minimumFamilyTokens = 0) {
  const selected = [];
  let remaining = targetTokens;
  const used = new Set();
  const families = new Map();
  for (const record of records) {
    if (!families.has(record.family)) families.set(record.family, []);
    families.get(record.family).push(record);
  }
  for (const family of families.values()) {
    let familyRemaining = Math.min(minimumFamilyTokens, remaining);
    for (const record of family) {
      if (familyRemaining <= 0) break;
      const count = takeMiningRecord(record, selected, used, familyRemaining);
      familyRemaining -= count;
      remaining -= count;
    }
  }
  const groups = new Map();
  for (const record of records) {
    if (used.has(record)) continue;
    const key = `${record.sourceName}:${record.family}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }
  while (remaining > 0) {
    let changed = false;
    for (const group of groups.values()) {
      const record = group.shift();
      if (!record) continue;
      const count = takeMiningRecord(record, selected, used, remaining);
      remaining -= count;
      changed = true;
      if (remaining === 0) break;
    }
    if (!changed) throw new Error(`not enough mining tokens; need ${remaining} more`);
  }
  return selected;
}

function takeMiningRecord(record, selected, used, maximum) {
  const count = Math.min(maximum, record.parts.length);
  if (!count) return 0;
  used.add(record);
  if (count === record.parts.length) selected.push(record);
  else {
    const parts = record.parts.slice(0, count);
    selected.push({ ...record, source: record.source.slice(0, parts.at(-1).to), parts });
  }
  return count;
}

function interleaveRecords(left, right) {
  const records = [];
  for (let index = 0; index < Math.max(left.length, right.length); index++) {
    if (left[index]) records.push(left[index]);
    if (right[index]) records.push(right[index]);
  }
  return records;
}

function increment(object, key, amount) { object[key] = (object[key] ?? 0) + amount; }

function validateSelected(split, summary, policy, splitPolicy) {
  if (summary.tokens !== policy.targetTokens[split]) throw new Error(`${split} has ${summary.tokens} tokens; expected ${policy.targetTokens[split]}`);
  if (split === "mining") return;
  for (const [stratum, ratio] of Object.entries(policy.strata)) {
    const expected = Math.round(summary.tokens * ratio);
    if (Math.abs((summary.strata[stratum] ?? 0) - expected) > 1) throw new Error(`${split}/${stratum} missed its token quota`);
  }
  if (splitPolicy.minifiedCompiledFraction) {
    const expected = Math.round(summary.strata.compiled * splitPolicy.minifiedCompiledFraction);
    if (summary.forms.minified !== expected) {
      throw new Error(`${split} has ${summary.forms.minified ?? 0} minified tokens; expected ${expected}`);
    }
  }
  const missing = policy.requiredLanguagesPerSplit.filter((language) => !summary.languageFamilies[language]);
  if (missing.length) throw new Error(`${split} corpus is missing required languages: ${missing.join(", ")}`);
  for (const [language, minimum] of Object.entries(policy.minimumLanguageTokens?.[split] ?? {})) {
    if ((summary.languages[language] ?? 0) < minimum) {
      throw new Error(`${split} corpus has ${summary.languages[language] ?? 0} ${language} tokens; expected at least ${minimum}`);
    }
  }
  const constructFloor = policy.constructCoverage?.minimumTokens?.[split] ?? 0;
  const missingConstructs = (policy.constructCoverage?.required ?? []).filter(
    (construct) => (summary.constructs[construct] ?? 0) < constructFloor,
  );
  if (missingConstructs.length) {
    throw new Error(`${split} corpus missed its ${constructFloor}-token construct floor: ${missingConstructs.join(", ")}`);
  }
}

function formatMix(strata) {
  const total = Object.values(strata).reduce((sum, count) => sum + count, 0);
  return Object.entries(strata).map(([name, count]) => `${name}=${(count / total * 100).toFixed(0)}%`).join(" ");
}

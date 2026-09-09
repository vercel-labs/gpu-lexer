import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";
import { spawnSync } from "node:child_process";
import { pinBaseline, fixedBaselineMetrics } from "../src/tree-training-policy.js";

import {
  assertSameProvenance, compareLanguageMetrics, compareStrictLanguages,
  createLanguageObjective, languageMetrics,
} from "../src/language-objective.js";
import { createTreeModel, evaluateTree, treeAuxiliaryNames, treeTensorNames } from "../src/tree-model.js";
import { accuracyTreePromotionDecision, evaluateTreePromotion, validateTreeComparisonProvenance } from "../src/promote.js";
import { classNames } from "../src/classes.js";
import { quantizeTensors, dequantizeTensors } from "../src/quantization.js";
import { treeInputSize } from "../../core/src/tree-features.js";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const objective = (overrides = {}) => ({
  schemaVersion: 1, popularitySha256: "a".repeat(64), supplementalWeights: {},
  familyWeights: { javascript: 0.75, python: 0.25 }, protectedFamilies: ["javascript", "python"],
  minSupport: 100, minPlainSupport: 100, maxErrorIncrease: 0.01, maxFalseColorIncrease: 0.01,
  strictLanguages: ["javascript", "python"], maxStrictErrorIncrease: 0.002,
  maxStrictFalseColorIncrease: 0.002,
  z: 1.96, minWeightedImprovement: 0, ...overrides,
});
const provenance = { shardSha256: "b".repeat(64), labelSource: "shiki-spans-v1", featureVersion: 2, tokenizerVersion: 2, maxTokens: null };
const count = (errors, support = 10000, falseColors = 0, plainSupport = 1000) => ({ support, errors, falseColors, plainSupport });
const metrics = (counts, config) => ({ ...languageMetrics(counts, config),
  perLanguage: Object.fromEntries(Object.entries(counts).map(([language, value]) => [language, {
    ...value, error: value.errors / value.support,
    falseColorRate: value.plainSupport ? value.falseColors / value.plainSupport : null,
  }])), provenance: { ...provenance } });

test("Python and JS share family weights and fixed-baseline guard decisions", () => {
  const config = objective();
  const baseline = metrics({ javascript: count(2000), python: count(1000) }, config);
  const cases = [metrics({ javascript: count(1000), python: count(1000) }, config),
    metrics({ javascript: count(1000), python: count(2000) }, config),
    metrics({ javascript: count(1000), python: count(900, 10000, 100) }, config)];
  const python = new URL("../../../.venv/bin/python", import.meta.url).pathname;
  const result = spawnSync(python, ["-c", `import json,sys
from tree_model import normalized_family_weights, protected_families_eligible
v=json.load(sys.stdin)
print(json.dumps({'weights':normalized_family_weights(v['config']), 'guards':[protected_families_eligible(m,v['baseline'],v['config']) for m in v['cases']]}))`], {
    cwd: new URL("../torch/", import.meta.url), input: JSON.stringify({ config, baseline, cases }), encoding: "utf8", timeout: 10000,
  });
  assert.equal(result.status, 0, result.stderr);
  const actual = JSON.parse(result.stdout);
  assert.deepEqual(actual.weights, config.familyWeights);
  assert.deepEqual(actual.guards, cases.map((m) => compareLanguageMetrics(m, baseline, config).guards.every((g) => g.passed)));
});

test("snapshot percentages and explicit supplemental weights normalize globally", async () => {
  const config = createLanguageObjective();
  const snapshot = await readFile(new URL("../data/language-popularity.json", import.meta.url));
  const data = JSON.parse(snapshot);
  const denominator = data.languages.reduce((sum, row) => sum + (row.supplemental ? 1 : row.percent), 0);
  assert.equal(config.popularitySha256, sha(snapshot));
  assert.equal(config.familyWeights.html, 1 / denominator);
  assert.equal(config.familyWeights.javascript, 19.8731 / denominator);
  assert.deepEqual(config.strictLanguages, ["css", "html", "javascript", "python", "tsx", "typescript"]);
  assert.equal(config.maxStrictErrorIncrease, 0.002);
  assert.equal(config.maxStrictFalseColorIncrease, 0.002);
  assert.ok(Math.abs(Object.values(config.familyWeights).reduce((a, b) => a + b) - 1) < 1e-12);
  assert.throws(() => createLanguageObjective({ supplementalWeights: {} }), /missing explicit supplemental/);
  assert.throws(() => createLanguageObjective({ minSupport: 0 }), /minSupport/);
});

test("family weighting differs from token micro accuracy; missing families never renormalize", () => {
  const config = objective();
  const result = languageMetrics({ javascript: count(10, 100, 2, 50), python: count(900, 1000, 3, 100) }, config);
  assert.equal(result.weightedError, 0.75 * 0.1 + 0.25 * 0.9);
  assert.equal(result.perFamily.javascript.falseColorRate, 2 / 50);
  const missing = languageMetrics({ javascript: count(10) }, config);
  assert.equal(missing.weightedError, null);
  assert.equal(missing.objectiveComplete, false);
  assert.deepEqual(missing.missingFamilies, ["python"]);
  assert.equal(missing.perFamily.python.error, null);
  const unknown = languageMetrics({ javascript: count(10), python: count(10), other: count(1) }, config);
  assert.equal(unknown.weightedError, null);
  assert.deepEqual(unknown.unknownFamilies, ["other"]);
  assert.throws(() => languageMetrics({}, objective({ familyWeights: { javascript: 3 } })), /normalized/);
});

test("tree evaluation excludes whitespace and masked labels, not low confidence or loss weight", () => {
  const config = { hiddenSize: 2, classifierSize: 2, languageObjective: objective({ familyWeights: { javascript: 1 }, protectedFamilies: [] }) };
  const model = createTreeModel({ ...config, hashBuckets: 0, random: () => 0.5 });
  model.outputBias[4] = 2;
  const records = [{ family: "javascript", language: "javascript",
    features: [0, 0, 0, 1, 2].map((kind) => Uint16Array.of(kind)),
    targets: Uint8Array.of(0, 4, 4, 0, 0), supervisionWeights: Uint8Array.of(1, 255, 0, 255, 255), lossWeights: [9, 1, 1, 1, 1] }];
  const result = evaluateTree(model, records, config);
  assert.equal(result.perLanguage.javascript.accuracy,
    1 - result.perLanguage.javascript.errors / result.perLanguage.javascript.support);
  assert.equal(result.accuracy, 0.5);
  assert.equal(result.weightedError, 0.5);
  assert.deepEqual(result.perFamily.javascript, { support: 2, errors: 1, plainSupport: 1, falseColors: 1, error: 0.5, falseColorRate: 1 });
});

test("fixed baseline selection rejects strong-language regressions despite lower weighted error", () => {
  const config = objective();
  const baseline = metrics({ javascript: count(2000), python: count(1000) }, config);
  const improved = metrics({ javascript: count(1500), python: count(1000) }, config);
  assert.equal(compareLanguageMetrics(improved, baseline, config).accepted, true);
  assert.equal(compareLanguageMetrics(baseline, baseline, config).accepted, false);
  const regressed = metrics({ javascript: count(1000), python: count(2000) }, config);
  const result = compareLanguageMetrics(regressed, baseline, config);
  assert.ok(result.improvement > 0);
  assert.equal(result.accepted, false);
  assert.ok(result.failures.includes("python error: regression"));
  assert.ok(result.warnings.includes("python error: regression"));
  const falseColor = metrics({ javascript: count(1000), python: count(900, 10000, 100) }, config);
  assert.ok(compareLanguageMetrics(falseColor, baseline, config).warnings.includes("python falseColorRate: regression"));
  assert.ok(compareLanguageMetrics(falseColor, baseline, config).failures.includes("python falseColorRate: regression"));
});

test("accuracy promotion remains blocked by a strict language regression", () => {
  const weighted = { guards: [], warnings: [], strictGuards: [
    { language: "typescript", metric: "error", passed: false, reason: "regression" },
  ] };
  const decision = accuracyTreePromotionDecision({ accuracy: 0.88 }, { accuracy: 0.87 }, weighted);
  assert.equal(decision.accepted, false);
  assert.deepEqual(decision.failures, ["typescript error: regression"]);
});

test("mature language guards allow one point while strict guards stay tighter", () => {
  const config = objective({ matureLanguages: ["jsx"], maxMatureErrorIncrease: 0.01,
    maxMatureFalseColorIncrease: 0.01 });
  const baseline = metrics({ javascript: count(1000), python: count(1000) }, config);
  const candidate = metrics({ javascript: count(1025), python: count(1000) }, config);
  baseline.perLanguage.jsx = count(1000);
  candidate.perLanguage.jsx = count(1090);
  const guards = compareLanguageMetrics(candidate, baseline, config).strictGuards;
  assert.equal(guards.find((guard) => guard.language === "javascript" && guard.metric === "error").passed, false);
  assert.equal(guards.find((guard) => guard.language === "jsx" && guard.metric === "error").passed, true);
  candidate.perLanguage.jsx.errors = 1101;
  assert.equal(compareStrictLanguages(candidate, baseline, config)
    .find((guard) => guard.language === "jsx" && guard.metric === "error").passed, false);
});

test("sample-size aware tolerance reports low support without blocking improvement", () => {
  const config = objective({ familyWeights: { javascript: 0.1, python: 0.9 }, strictLanguages: ["python"] });
  const baseline = metrics({ javascript: count(10, 100, 0, 100), python: count(2000) }, config);
  const candidate = metrics({ javascript: count(12, 100, 0, 100), python: count(1000) }, config);
  assert.equal(compareLanguageMetrics(candidate, baseline, config).accepted, true);
  const sparse = metrics({ javascript: count(1, 99, 0, 99), python: count(2000) }, config);
  const sparseCandidate = metrics({ javascript: count(1, 99, 0, 99), python: count(1000) }, config);
  assert.ok(compareLanguageMetrics(sparseCandidate, sparse, config).warnings.includes("javascript error: insufficient support"));
  const noPlain = metrics({ javascript: count(1, 100, 0, 0), python: count(2000) }, config);
  assert.equal(noPlain.perFamily.javascript.falseColorRate, null);
  assert.ok(compareLanguageMetrics(noPlain, noPlain, config).warnings.includes("javascript falseColorRate: insufficient support"));
});

test("comparison rejects missing, inconsistent and mismatched metric provenance", () => {
  const config = objective();
  const baseline = metrics({ javascript: count(2000), python: count(1000) }, config);
  assert.throws(() => compareLanguageMetrics({ ...baseline, provenance: undefined }, baseline, config), /provenance/);
  assert.throws(() => assertSameProvenance(provenance, { ...provenance, maxTokens: 100 }), /mismatch/);
  assert.throws(() => compareLanguageMetrics({ ...baseline, languageObjective: objective({ z: 0 }) }, baseline, config), /objective provenance/);
  assert.throws(() => compareLanguageMetrics({ ...baseline, weightedError: 0 }, baseline, config), /inconsistent/);
  assert.throws(() => compareLanguageMetrics(metrics({ javascript: count(2000) }, config), baseline, config), /incomplete/);
  assert.throws(() => compareLanguageMetrics(metrics({ javascript: count(2000, 9000), python: count(1000) }, config), baseline, config), /support mismatch/);
  const tampered = structuredClone(baseline);
  tampered.perFamily.python.falseColorRate = 1;
  assert.throws(() => compareLanguageMetrics(tampered, baseline, config), /inconsistent rates/);
});

async function promotionFixture(t) {
  const directory = await mkdtemp(join(tmpdir(), "language-promotion-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const shard = join(directory, "verification.jsonl.gz");
  const bytes = gzipSync(JSON.stringify({ source: "a+b", family: "javascript", language: "javascript", sourceName: "fixture", path: "a.js",
    sourceLabelsVersion: 1, sourceLabels: [{ from: 0, to: 3, class: "plain", confidence: 1 }] }) + "\n");
  await writeFile(shard, bytes);
  const model = createTreeModel({ hiddenSize: 32, classifierSize: 2, hashBuckets: 0,
    featureVersion: 2, scaleBuckets: 8, random: () => 0.5 });
  model.outputBias[4] = 2;
  const packed = quantizeTensors(model, treeTensorNames, 6);
  const description = {
    runId: "baseline", model: "hierarchical-tree", formatVersion: 8, featureVersion: 2, tokenizerVersion: 2,
    labelSource: "shiki-spans-v1", inputSize: treeInputSize(0, 2), hiddenSize: 32, classNames, auxiliaryNames: treeAuxiliaryNames,
    corpus: { verification: { sha256: sha(bytes) } },
    architecture: { tree: "scale-aware-butterfly-binary", direction: "bidirectional", blockParts: 32, scaleBuckets: 8,
      localNeighborParts: 1, classifierAuxiliaryStates: true, lexemeHashBuckets: 0, classifierDimensions: 2, auxiliaryStates: treeAuxiliaryNames },
    quantization: packed.metadata, runtimeWeightBytes: packed.data.length, runtimeParameterCount: packed.parameterCount,
  };
  await writeFile(join(directory, "model-baseline.json"), JSON.stringify(description));
  await writeFile(join(directory, "weights-int6-baseline.bin"), packed.data);
  const candidate = { ...description, runId: "candidate", config: {
    fixedBaseline: { run: directory, runId: "baseline", weightsSha256: sha(packed.data) },
    languageObjective: objective({ familyWeights: { javascript: 1 }, protectedFamilies: ["javascript"],
      strictLanguages: ["javascript"], minSupport: 1, minPlainSupport: 1 }),
  } };
  model.outputBias[4] = 0;
  model.outputBias[0] = 2;
  const candidatePacked = quantizeTensors(model, treeTensorNames, 6);
  return { directory, shard, candidate, baseline: description, model: dequantizeTensors(candidatePacked.data, candidatePacked.metadata) };
}

test("promotion reevaluates pinned packed baseline and candidate on the same direct-label records without writes", async (t) => {
  const { directory, shard, candidate, baseline, model } = await promotionFixture(t);
  const before = await readFile(join(directory, "model-baseline.json"));
  const pinned = await pinBaseline(directory);
  assert.equal(pinned.weightsSha256, candidate.config.fixedBaseline.weightsSha256);
  const baselineForTraining = await fixedBaselineMetrics({ ...candidate.config, fixedBaseline: pinned,
    hashBuckets: 128, verificationShard: shard }, candidate, []);
  assert.equal(baselineForTraining.weightedError, 1);
  const comparison = await evaluateTreePromotion(candidate, model, shard);
  assert.equal(comparison.decision.accepted, true);
  assert.equal(comparison.candidate.weightedError, 0);
  assert.equal(comparison.baseline.weightedError, 1);
  assert.deepEqual(comparison.candidate.provenance, comparison.baseline.provenance);
  assert.deepEqual(await readFile(join(directory, "model-baseline.json")), before);
  assert.throws(() => validateTreeComparisonProvenance(candidate, { ...baseline, labelSource: undefined }, candidate.corpus.verification.sha256), /provenance/);
  assert.throws(() => validateTreeComparisonProvenance(candidate, { ...baseline, tokenizerVersion: 1 }, candidate.corpus.verification.sha256), /provenance/);
  candidate.config.fixedBaseline.weightsSha256 = "0".repeat(64);
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /digest mismatch/);
});

test("any accuracy-selected tree run promotes on untouched verification accuracy", async (t) => {
  const { shard, candidate, model } = await promotionFixture(t);
  candidate.config.selectionMetric = "accuracy";
  const comparison = await evaluateTreePromotion(candidate, model, shard);
  assert.equal(comparison.decision.accepted, true);
  assert.equal(comparison.decision.criterion, "fixed-baseline-untouched-verification-accuracy");
  assert.ok(comparison.decision.candidateAccuracy > comparison.decision.baselineAccuracy);

  model.outputBias[0] = 0;
  model.outputBias[4] = 2;
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /verification accuracy did not improve/);
});

test("explicit comparison pin permits fresh evaluation on a repaired shard without rewriting baseline history", async (t) => {
  const { directory, shard, candidate, baseline, model } = await promotionFixture(t);
  baseline.corpus = { verification: { sha256: "a".repeat(64) } };
  await writeFile(join(directory, "model-baseline.json"), JSON.stringify(baseline));
  const before = await readFile(join(directory, "model-baseline.json"));
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /provenance/);
  candidate.config.fixedBaseline.comparisonShardSha256 = candidate.corpus.verification.sha256;
  const baselineForTraining = await fixedBaselineMetrics({ ...candidate.config, hashBuckets: 128, verificationShard: shard }, candidate, []);
  assert.equal(baselineForTraining.weightedError, 1);
  const comparison = await evaluateTreePromotion(candidate, model, shard);
  assert.equal(comparison.decision.accepted, true);
  assert.deepEqual(comparison.candidate.provenance, comparison.baseline.provenance);
  assert.deepEqual(await readFile(join(directory, "model-baseline.json")), before);
  candidate.config.fixedBaseline.comparisonShardSha256 = "b".repeat(64);
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /provenance|mismatch/);
});

test("promotion compares feature-v3 candidates with the feature-v2 active baseline", async (t) => {
  const { shard, candidate, model } = await promotionFixture(t);
  candidate.featureVersion = 3;
  candidate.architecture = { ...candidate.architecture, lexemeHashBuckets: 256, scaleBuckets: 12,
    secondaryLexemeHashBuckets: 128, neighborSymbolHashBuckets: 32 };
  candidate.inputSize = treeInputSize(256, 3);
  const larger = createTreeModel({ hiddenSize: 32, classifierSize: 2, hashBuckets: 256,
    featureVersion: 3, scaleBuckets: 12, random: () => .5 });
  larger.outputBias.set(model.outputBias);
  const comparison = await evaluateTreePromotion(candidate, larger, shard);
  assert.equal(comparison.decision.accepted, true);
  assert.equal(comparison.candidate.perFamily.javascript.support, comparison.baseline.perFamily.javascript.support);
});

test("promotion refuses legacy tokenizer-only shard labels even with matching digest", async (t) => {
  const { directory, shard, candidate, baseline, model } = await promotionFixture(t);
  const bytes = gzipSync(JSON.stringify({ source: "x", family: "javascript", language: "javascript",
    tokens: [{ from: 0, to: 1, class: "plain", confidence: 255 }] }) + "\n");
  await writeFile(shard, bytes);
  candidate.corpus = baseline.corpus = { verification: { sha256: sha(bytes) } };
  await writeFile(join(directory, "model-baseline.json"), JSON.stringify(baseline));
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /sourceLabels/);
});

test("promotion rejects ties, incomplete coverage, mismatched shard, and unpinned baselines", async (t) => {
  const { shard, candidate, model } = await promotionFixture(t);
  model.outputBias[0] = 0;
  model.outputBias[4] = 2;
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /weighted error did not improve/);
  const forced = await evaluateTreePromotion(candidate, model, shard, { force: true });
  assert.equal(forced.automaticDecision.accepted, false);
  assert.equal(forced.decision.accepted, true);
  assert.equal(forced.decision.forced, true);
  assert.equal(forced.decision.criterion, "explicit-user-override");
  candidate.config.languageObjective = objective();
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /incomplete/);
  candidate.corpus = { verification: { sha256: "0".repeat(64) } };
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /provenance/);
  delete candidate.config.fixedBaseline;
  await assert.rejects(evaluateTreePromotion(candidate, model, shard), /pinned/);
});

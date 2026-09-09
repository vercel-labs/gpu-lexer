import { createHash } from "node:crypto";
import { basename, resolve } from "node:path";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import { classNames } from "./classes.js";
import { loadFloatCheckpoint } from "./checkpoint.js";
import { languageFamily, readLanguagePopularity } from "./corpus.js";
import { addFailureToBank, resolveFailureBankPath } from "./failure-bank.js";
import { createLabeler } from "./label.js";
import { promote } from "./promote.js";
import { dequantizeTensors, quantizeTensors } from "./quantization.js";
import { trainTree } from "./train-tree.js";
import {
  TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION, TREE_MODEL,
  createTreeRecord, treeProbabilities, treeTensorNamesFor,
} from "./tree-model.js";

const DEFAULT_EPOCHS = 8;
const DEFAULT_HEAD_EPOCHS = 3;
const DEFAULT_CALIBRATION_EPOCHS = 2;
const DEFAULT_FAILURE_WEIGHT = 3;
const DEFAULT_CONSISTENCY_WEIGHT = 0.2;
const DEFAULT_FOCUSED_REPLAY_STEPS = 1;
const DEFAULT_FOCUSED_REPLAY_LEARNING_RATE = 0.001;
const DEFAULT_FOCUSED_REPLAY_FINAL_LEARNING_RATE = 0.0005;

export async function fineTune(options) {
  validateOptions(options);
  const sourcePath = resolve(options.file);
  const source = await readFile(sourcePath, "utf8");
  if (!source.length) throw new Error("fine-tune source file is empty");
  const language = normalizeLanguage(options.lang);
  const { promotedRunId, promotedRunPath } = await import("./promoted-run.generated.js");
  const initial = await loadFloatCheckpoint(promotedRunPath);
  const shape = treeShape(initial.metadata);
  const family = await familyForLanguage(language);
  if (!(family in initial.metadata.config.languageObjective.familyWeights)) {
    throw new Error(`Shiki language ${language} maps to untrained family ${family}; add corpus coverage before fine-tuning it`);
  }

  const labeler = await createLabeler({ langs: [language] });
  let sourceLabels;
  try { sourceLabels = labeler.labelSource(source, language); }
  finally { labeler.dispose(); }
  const { record } = createTreeRecord({
    source, sourceLabels, sourceLabelsVersion: 1, language, family,
    sourceName: "fine-tune", path: basename(sourcePath), origin: "user-failure",
  }, shape.hashBuckets, { retainSource: true });
  const initialDeployed = deployedModel(initial.model, shape.weightBits);
  const before = mismatchReport(initialDeployed, record, shape, source);
  if (!before.supervised) throw new Error("Shiki produced no supervised parts for this snippet");
  if (!before.mismatches.length) {
    console.log(`active model ${promotedRunId} already matches Shiki on all ${before.supervised} supervised parts`);
    return { trained: false, promoted: false, before };
  }
  for (const mismatch of before.mismatches.slice(0, 20)) {
    console.log(`  ${mismatch.from}:${mismatch.to} ${JSON.stringify(mismatch.text)}: ` +
      `${mismatch.predicted} -> ${mismatch.expected}`);
  }
  if (before.mismatches.length > 20) console.log(`  … ${before.mismatches.length - 20} more disagreements`);

  const requiredRecord = focusedFailureRecord(record, before.mismatches);
  const failureWeight = options.failureWeight ?? DEFAULT_FAILURE_WEIGHT;
  requiredRecord.replayWeight = failureWeight;
  const currentFailure = focusedFailureRecord(record, before.mismatches, failureWeight);
  currentFailure.replayWeight = failureWeight;
  const sourceSha256 = createHash("sha256").update(source).digest("hex");
  const bankPath = resolveFailureBankPath(options.bank);
  const bank = await addFailureToBank({
    language, family, source, sourceLabels, sourceFile: basename(sourcePath),
  }, bankPath);
  const hardRecords = [currentFailure];
  let bankFailures = before.mismatches.length;
  for (const entry of bank.entries) {
    if (!(entry.family in initial.metadata.config.languageObjective.familyWeights)) continue;
    if (entry.language === language && entry.sourceSha256 === sourceSha256) continue;
    const { record: bankRecord } = createTreeRecord({
      source: entry.source, sourceLabels: entry.sourceLabels, sourceLabelsVersion: 1,
      language: entry.language, family: entry.family, sourceName: "failure-bank",
      path: entry.sourceFile, origin: "user-failure-bank",
    }, shape.hashBuckets, { retainSource: true });
    const report = mismatchReport(initialDeployed, bankRecord, shape, entry.source);
    if (!report.mismatches.length) continue;
    const focused = focusedFailureRecord(bankRecord, report.mismatches, failureWeight);
    focused.replayWeight = entry.sourceSha256 === sourceSha256 ? failureWeight : 1;
    hardRecords.push(focused);
    bankFailures += report.mismatches.length;
  }

  const epochs = options.epochs ?? DEFAULT_EPOCHS;
  const headTuneEpochs = options.headTuneEpochs ?? Math.min(DEFAULT_HEAD_EPOCHS, epochs);
  const calibrationEpochs = options.calibrationEpochs ??
    Math.min(DEFAULT_CALIBRATION_EPOCHS, epochs - headTuneEpochs);
  const fullTuneEpochs = epochs - headTuneEpochs - calibrationEpochs;
  const consistencyWeight = options.consistencyWeight ?? DEFAULT_CONSISTENCY_WEIGHT;
  const focusedReplaySteps = options.focusedReplaySteps ?? DEFAULT_FOCUSED_REPLAY_STEPS;
  const focusedReplayLearningRate = options.focusedReplayLearningRate ?? DEFAULT_FOCUSED_REPLAY_LEARNING_RATE;
  const focusedReplayFinalLearningRate = options.focusedReplayFinalLearningRate ??
    DEFAULT_FOCUSED_REPLAY_FINAL_LEARNING_RATE;
  console.log(`[${new Date().toISOString()}] fine-tuning ${promotedRunId} with ` +
    `${hardRecords.length}/${bank.entries.length} active failure-bank snippets and ${bankFailures} failing parts; ` +
    `${headTuneEpochs} classifier-only + ${fullTuneEpochs} full-model + ` +
    `${calibrationEpochs} natural calibration epochs; ` +
    `${focusedReplaySteps} balanced failure-only update${focusedReplaySteps === 1 ? "" : "s"} per learning epoch at ` +
    `${focusedReplayLearningRate} -> ${focusedReplayFinalLearningRate}, followed by natural rehearsal; ` +
    `${(consistencyWeight * 100).toFixed(0)}% active-model consistency`);

  const result = await trainTree({
    initialRun: promotedRunPath, baselineRun: promotedRunPath,
    teacherRun: promotedRunPath, distillationWeight: consistencyWeight,
    distillationWarmupEpochs: 0, distillationRampEpochs: 1,
    selectionMetric: "accuracy", classWeightPower: 0,
    epochs, fineTuneEpochs: epochs, headTuneEpochs, calibrationEpochs, stagedFineTune: true,
    repeatHardReplay: true, naturalReplay: true, focusedReplay: true, focusedReplaySteps,
    focusedReplayLearningRate, focusedReplayFinalLearningRate,
    qatEpochs: epochs, emaStartEpoch: epochs + 1,
    fineTuneLearningRate: options.headLearningRate ?? 0.000005,
    fineTuneFinalLearningRate: options.headFinalLearningRate ?? 0.000001,
    scanFineTuneLearningRate: options.learningRate ?? 0.000001,
    scanFineTuneFinalLearningRate: options.finalLearningRate ?? 0.0000002,
    calibrationLearningRate: options.calibrationLearningRate ?? 0.000001,
    calibrationFinalLearningRate: options.calibrationFinalLearningRate ?? 0.0000002,
    patience: epochs,
    hardRecords, requiredRecords: [requiredRecord],
    device: options.device, python: options.python, output: options.output,
    maxTrainTokens: options.maxTrainTokens, maxVerificationTokens: options.maxVerificationTokens,
    fineTuneMetadata: {
      kind: "single-shiki-failure", language, family, sourceSha256,
      sourceFile: basename(sourcePath), initialRun: promotedRunId,
      supervisedParts: before.supervised, requiredParts: before.mismatches.length,
      headTuneEpochs, fullTuneEpochs, calibrationEpochs,
      failureWeight, consistencyWeight,
      focusedReplaySteps, focusedReplayLearningRate, focusedReplayFinalLearningRate,
      failureBankEntries: bank.entries.length, activeFailureBankEntries: hardRecords.length,
      activeFailureBankParts: bankFailures,
    },
  });

  const after = mismatchReport(deployedModel(result.model, shape.weightBits), requiredRecord, shape, source);
  const diagnostics = {
    createdAt: new Date().toISOString(), runId: result.metadata.runId, initialRun: promotedRunId,
    language, family, sourceFile: basename(sourcePath), sourceSha256,
    gates: {
      snippetPassed: after.mismatches.length === 0,
      verificationImproved: result.metadata.verification.accuracy >
        result.metadata.config.fixedBaselineMetrics.accuracy,
    },
    before, after,
    verification: {
      before: result.metadata.config.fixedBaselineMetrics,
      after: result.metadata.verification,
    },
    selection: result.selection,
  };
  const diagnosticsPath = resolve(result.path, `fine-tune-${result.metadata.runId}.json`);
  await writeFile(diagnosticsPath, `${JSON.stringify(diagnostics, null, 2)}\n`, { flag: "wx" });

  if (result.selection.status !== "eligible") {
    console.log(`[${new Date().toISOString()}] saved diagnostic run ${result.metadata.runId}; not promoted`);
    console.log(`targeted snippet: ${after.correct}/${after.supervised}; ` +
      `verification accuracy: ${(result.metadata.verification.accuracy * 100).toFixed(2)}%`);
    return { trained: true, promoted: false, result, diagnostics, diagnosticsPath };
  }
  await promote(result.path);
  return { trained: true, promoted: true, result, diagnostics, diagnosticsPath };
}

export function mismatchReport(model, record, shape, source = record.source ?? "") {
  const probabilities = treeProbabilities(model, record, shape);
  const mismatches = [];
  let correct = 0, supervised = 0;
  for (let part = 0; part < record.targets.length; part++) {
    if (!record.supervisionWeights[part]) continue;
    supervised += 1;
    const predicted = argmax(probabilities[part]);
    const expected = record.targets[part];
    if (predicted === expected) { correct += 1; continue; }
    const from = record.ranges?.[part * 2] ?? null;
    const to = record.ranges?.[part * 2 + 1] ?? null;
    mismatches.push({
      part, from, to, text: from == null ? null : source.slice(from, to),
      expected: classNames[expected], predicted: classNames[predicted],
    });
  }
  return { passed: supervised > 0 && correct === supervised, correct, supervised, accuracy: correct / Math.max(1, supervised), mismatches };
}

export function focusedFailureRecord(record, mismatches, failureWeight = 1) {
  const focused = cloneRecord(record);
  const originalSupervision = focused.supervisionWeights.slice();
  focused.supervisionWeights.fill(0);
  focused.lossWeights.fill(1);
  for (const { part } of mismatches) {
    if (!Number.isSafeInteger(part) || part < 0 || part >= focused.targets.length) {
      throw new RangeError(`invalid failing part ${part}`);
    }
    focused.supervisionWeights[part] = originalSupervision[part];
    focused.lossWeights[part] = failureWeight;
  }
  return focused;
}

export function parseFineTuneArguments(arguments_) {
  const result = {};
  const aliases = {
    lang: "lang", file: "file", epochs: "epochs", "head-epochs": "headTuneEpochs",
    "calibration-epochs": "calibrationEpochs",
    "failure-weight": "failureWeight",
    "failure-steps": "focusedReplaySteps",
    "failure-lr": "focusedReplayLearningRate", "failure-final-lr": "focusedReplayFinalLearningRate",
    bank: "bank", "consistency-weight": "consistencyWeight",
    "head-lr": "headLearningRate", "head-final-lr": "headFinalLearningRate",
    lr: "learningRate", "final-lr": "finalLearningRate", device: "device", python: "python",
    "calibration-lr": "calibrationLearningRate", "calibration-final-lr": "calibrationFinalLearningRate",
    output: "output", "max-train-tokens": "maxTrainTokens", "max-verification-tokens": "maxVerificationTokens",
  };
  const integers = new Set(["epochs", "headTuneEpochs", "calibrationEpochs", "focusedReplaySteps", "failureWeight", "maxTrainTokens", "maxVerificationTokens"]);
  const numbers = new Set(["consistencyWeight", "headLearningRate", "headFinalLearningRate", "learningRate", "finalLearningRate", "calibrationLearningRate", "calibrationFinalLearningRate", "focusedReplayLearningRate", "focusedReplayFinalLearningRate"]);
  for (let index = 0; index < arguments_.length; index++) {
    if (arguments_[index] === "--") continue;
    const [raw, inline] = arguments_[index].replace(/^--/, "").split("=", 2);
    const name = aliases[raw];
    if (!name) throw new Error(`unknown option --${raw}`);
    const value = inline ?? arguments_[++index];
    if (value == null) throw new Error(`missing value for --${raw}`);
    result[name] = integers.has(name) ? Number.parseInt(value, 10) : numbers.has(name) ? Number(value) : value;
  }
  return result;
}

function validateOptions(options = {}) {
  if (!options.lang || !options.file) throw new Error("usage: pnpm fine-tune -- --lang <shiki-language> --file <source-file>");
  const epochs = options.epochs ?? DEFAULT_EPOCHS;
  const head = options.headTuneEpochs ?? Math.min(DEFAULT_HEAD_EPOCHS, epochs);
  const calibration = options.calibrationEpochs ?? Math.min(DEFAULT_CALIBRATION_EPOCHS, epochs - head);
  const weight = options.failureWeight ?? DEFAULT_FAILURE_WEIGHT;
  const consistency = options.consistencyWeight ?? DEFAULT_CONSISTENCY_WEIGHT;
  const focusedReplaySteps = options.focusedReplaySteps ?? DEFAULT_FOCUSED_REPLAY_STEPS;
  const focusedReplayLearningRate = options.focusedReplayLearningRate ?? DEFAULT_FOCUSED_REPLAY_LEARNING_RATE;
  const focusedReplayFinalLearningRate = options.focusedReplayFinalLearningRate ??
    DEFAULT_FOCUSED_REPLAY_FINAL_LEARNING_RATE;
  if (!Number.isSafeInteger(epochs) || epochs < 1 || !Number.isSafeInteger(head) || head < 0 ||
      !Number.isSafeInteger(calibration) || calibration < 0 || head + calibration > epochs) {
    throw new Error("require nonnegative head/calibration epochs whose sum does not exceed epochs");
  }
  if (!Number.isSafeInteger(weight) || weight < 2 || weight > 3) throw new Error("failure-weight must be 2 or 3");
  if (!Number.isSafeInteger(focusedReplaySteps) || focusedReplaySteps < 1 || focusedReplaySteps > 4) {
    throw new Error("failure-steps must be between 1 and 4");
  }
  if (!Number.isFinite(focusedReplayLearningRate) || !Number.isFinite(focusedReplayFinalLearningRate) ||
      focusedReplayLearningRate <= 0 || focusedReplayFinalLearningRate <= 0 ||
      focusedReplayFinalLearningRate > focusedReplayLearningRate) {
    throw new Error("failure learning rates must be positive and final must not exceed initial");
  }
  if (!Number.isFinite(consistency) || consistency < 0 || consistency > 0.5) {
    throw new Error("consistency-weight must be between 0 and 0.5");
  }
  for (const [name, start, end] of [
    ["head", options.headLearningRate ?? 0.000005, options.headFinalLearningRate ?? 0.000001],
    ["full-model", options.learningRate ?? 0.000001, options.finalLearningRate ?? 0.0000002],
    ["calibration", options.calibrationLearningRate ?? 0.000001,
      options.calibrationFinalLearningRate ?? 0.0000002],
  ]) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || start <= 0 || end <= 0 || end > start) {
      throw new Error(`${name} learning rates must be positive and final must not exceed initial`);
    }
  }
}

function treeShape(metadata) {
  if (metadata.model !== TREE_MODEL ||
      ![TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION].includes(metadata.formatVersion)) {
    throw new Error(`active checkpoint ${metadata.runId} is not a supported tree model`);
  }
  return {
    hiddenSize: metadata.hiddenSize,
    classifierSize: metadata.architecture.classifierDimensions,
    hashBuckets: metadata.architecture.lexemeHashBuckets,
    weightBits: metadata.quantization?.bits ?? metadata.config?.weightBits ?? 6,
  };
}

function deployedModel(model, weightBits) {
  const packed = quantizeTensors(model, treeTensorNamesFor(model), weightBits);
  return dequantizeTensors(packed.data, packed.metadata);
}

function cloneRecord(record) {
  return {
    ...record, features: record.features.map((features) => features.slice()),
    targets: record.targets.slice(), auxiliary: record.auxiliary.slice(),
    supervisionWeights: record.supervisionWeights.slice(), lossWeights: record.lossWeights.slice(),
    ranges: record.ranges?.slice(), sourceLabels: record.sourceLabels?.map((label) => ({ ...label })),
  };
}

async function familyForLanguage(language) {
  const popularity = await readLanguagePopularity();
  return popularity.languages.find((entry) => entry.shiki.includes(language))?.family ?? languageFamily(language);
}

function normalizeLanguage(language) {
  return ({ js: "javascript", ts: "typescript", py: "python", rb: "ruby", sh: "shellscript",
    bash: "shellscript", zsh: "shellscript", yml: "yaml", md: "markdown" })[language.toLowerCase()] ?? language;
}

function argmax(values) {
  let result = 0;
  for (let index = 1; index < values.length; index++) if (values[index] > values[result]) result = index;
  return result;
}

const invoked = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (import.meta.url === invoked) try { await fineTune(parseFineTuneArguments(process.argv.slice(2))); }
catch (error) { console.error(error instanceof Error ? error.stack : error); process.exitCode = 1; }

import { createHash } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { classNames } from "./classes.js";
import { loadDeployedCheckpoint, loadFloatCheckpoint } from "./checkpoint.js";
import { dequantizeTensors, quantizeTensors } from "./quantization.js";
import { resolveTorchRuntime, runTorchTraining } from "./torch-runner.js";
import {
  TREE_FEATURE_VERSION, TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION, TREE_MODEL, TREE_TOKENIZER_VERSION,
  createTreeModel, loadTreeShard, migrateTreeFeatureV2Model,
  treeAuxiliaryNames, treeTensorLayout, treeTensorNamesFor,
} from "./tree-model.js";
import {
  TREE_HASH_BUCKETS, TREE_LEGACY_FEATURE_VERSION, TREE_LEGACY_SCALE_BUCKETS,
  TREE_SCALE_BUCKETS, TREE_SECONDARY_HASH_BUCKETS, TREE_SYMBOL_PAIR_BUCKETS, treeInputSize,
} from "../../core/src/tree-features.js";
import { languageFamily } from "./corpus.js";
import { balanceLanguageRecords, balancedAuxiliaryPositiveWeights, upsampleMixedRecords } from "./training-data.js";
import { treeTrainingPolicy, assertObjectiveCoverage, fixedBaselineMetrics } from "./tree-training-policy.js";
import {
  TREE_AUTO_REPLAY_REPEATS, TREE_REPLAY_FAMILIES, mineTreeReplay, treeReplayFraction,
} from "./tree-replay.js";
import { websiteExampleSourceKeys } from "./website-examples.js";

const runsRoot = fileURLToPath(new URL("../runs/", import.meta.url));
const TREE_PARAMETER_CEILING = 75_000;
const TREE_PACKED_BYTE_BUDGET = 37_500;

export async function trainTree(options = {}) {
  const teacherMode = options.teacherMode === true;
  if (options.fresh && options.initialRun) throw new Error("--fresh cannot be combined with --initial-run");
  let promotedRunId = null;
  let active = null;
  let migrationSource = null;
  if (!teacherMode) {
    const promoted = await import("./promoted-run.generated.js");
    promotedRunId = promoted.promotedRunId;
    active = await loadFloatCheckpoint(promoted.promotedRunPath);
  }
  let initial = options.initialRun ? await loadFloatCheckpoint(options.initialRun) : null;
  if (initial && options.initialRun) {
    const mismatch = treeWarmStartMismatch(initial.metadata, options);
    if (mismatch) throw new Error(`cannot warm-start ${initial.metadata.runId}: ${mismatch}`);
  }
  if (!initial && !options.fresh && active) {
    const mismatch = treeWarmStartMismatch(active.metadata, options);
    if (mismatch && active.metadata.featureVersion === 2) migrationSource = active;
    else if (mismatch) log(`cannot warm-start ${active.metadata.runId}: ${mismatch}; starting fresh`);
    else initial = active;
  }
  if (initial && !teacherMode) initial = await loadDeployedCheckpoint(initial.path);
  if (initial && teacherMode) throw new Error("tree teacher training cannot continue a runtime checkpoint");
  // Ordinary compatible continuations polish the deployed checkpoint. Targeted
  // failure fine-tuning supplies fineTuneMetadata and retains its own schedule.
  const targetedFineTune = Boolean(options.fineTuneMetadata);
  const polishMode = Boolean(initial && !teacherMode && !targetedFineTune);
  const schedule = treeTrainingSchedule(options, polishMode, targetedFineTune);
  let baselineRun = options.baselineRun;
  if (!teacherMode && baselineRun === undefined && active?.metadata.model === TREE_MODEL) baselineRun = active.path;
  const policy = await treeTrainingPolicy({ ...options, ...schedule,
    classWeightPower: polishMode ? 0 : options.classWeightPower, baselineRun });
  const runtime = await resolveTorchRuntime({ backend: "torch", device: options.device ?? "auto", python: options.python });
  // An incompatible promoted model remains the fixed comparison baseline, but
  // must not silently pull the new experiment back to its old feature shape.
  const shapeSource = initial;
  const weightBits = options.weightBits ?? shapeSource?.metadata.quantization?.bits ?? shapeSource?.metadata.config?.weightBits ?? 6;
  if (![4, 5, 6].includes(weightBits)) throw new RangeError("tree weights must use 4, 5, or 6 bits");
  if (teacherMode && options.teacherRun) throw new Error("--teacher and --teacher-run cannot be combined");
  const teacher = options.teacherRun ? await loadDeployedCheckpoint(options.teacherRun) : polishMode ? initial : null;
  const autoMining = polishMode && !options.hardRecords && !options.fineTuneMetadata;
  const treeContext = options.treeContext ?? shapeSource?.metadata.config?.treeContext ??
    (shapeSource ? shapeSource.metadata.formatVersion === TREE_HYBRID_FORMAT_VERSION ? "hybrid" : "tree" : "hybrid");
  if (!["tree", "hybrid"].includes(treeContext)) throw new RangeError("--context must be tree or hybrid");
  const defaultHashBuckets = 256;
  const defaultClassifierSize = weightBits === 6 ? 72 : weightBits === 5 ? 84 : 176;
  const hashBuckets = options.hashBuckets ?? shapeSource?.metadata.architecture?.lexemeHashBuckets ?? defaultHashBuckets;
  const config = {
    ...policy,
    model: TREE_MODEL, polishMode, epochs: schedule.epochs, fineTuneEpochs: schedule.fineTuneEpochs,
    agreementEpochs: schedule.agreementEpochs,
    headTuneEpochs: options.headTuneEpochs ?? 0, stagedFineTune: options.stagedFineTune ?? false,
    treeContext, localRadius: treeContext === "hybrid" ? 2 : 1,
    labelSource: "shiki-spans-v1", precision: teacherMode ? "float32" : `int${weightBits}`,
    hiddenSize: options.hiddenSize ?? shapeSource?.metadata.hiddenSize ?? (teacherMode ? 128 : treeContext === "hybrid" ? 32 : 64),
    classifierSize: options.classifierSize ?? shapeSource?.metadata.architecture?.classifierDimensions ??
      (teacherMode ? 256 : defaultClassifierSize),
    weightBits,
    hashBuckets,
    batchTokens: options.batchTokens ?? 0,
    learningRate: options.learningRate ?? (polishMode ? 0.000001 : 0.002),
    finalLearningRate: options.finalLearningRate ?? (polishMode ? 0.0000002 : 0.00005),
    fineTuneLearningRate: options.fineTuneLearningRate ?? 0.00005,
    fineTuneFinalLearningRate: options.fineTuneFinalLearningRate ?? 0.00001,
    agreementLearningRate: options.agreementLearningRate ?? (polishMode ? 0.000001 : 0.00005),
    agreementFinalLearningRate: options.agreementFinalLearningRate ?? (polishMode ? 0.0000002 : 0.00001),
    calibrationLearningRate: options.calibrationLearningRate ??
      (polishMode || targetedFineTune ? 0.000001 : 0.00005),
    calibrationFinalLearningRate: options.calibrationFinalLearningRate ??
      (polishMode || targetedFineTune ? 0.0000002 : 0.00001),
    scanFineTuneLearningRate: options.scanFineTuneLearningRate ?? 0.00001,
    scanFineTuneFinalLearningRate: options.scanFineTuneFinalLearningRate ?? 0.000002,
    gradientClip: 1, patience: options.patience ?? (polishMode ? schedule.epochs : 8), minDelta: 0,
    seed: options.seed ?? 1337, progressEvery: options.progressEvery ?? 10,
    replayFraction: treeReplayFraction(options.replayFraction, { autoMining }),
    repeatHardReplay: options.repeatHardReplay ?? autoMining,
    maxHardReplayRepeats: options.maxHardReplayRepeats ?? (autoMining ? TREE_AUTO_REPLAY_REPEATS : null),
    ...focusedReplayConfig(options),
    boundaryLossMultiplier: options.boundaryLossMultiplier ?? 1.25,
    mixedF1Tolerance: 0.005, qatEpochs: teacherMode ? 0 : options.qatEpochs ?? (polishMode ? schedule.epochs : 6),
    emaDecay: 0.999, emaStartEpoch: options.emaStartEpoch ?? (polishMode ? schedule.epochs + 1 : 20),
    distillationWeight: teacher ? options.distillationWeight ?? (polishMode ? 0.2 : 0.25) : 0,
    distillationTemperature: teacher ? options.distillationTemperature ?? 2 : 1,
    distillationWarmupEpochs: teacher ? options.distillationWarmupEpochs ?? (polishMode ? 0 : 2) : 0,
    distillationRampEpochs: teacher ? options.distillationRampEpochs ?? (polishMode ? 1 : 4) : 1,
    featureVersion: TREE_FEATURE_VERSION, scaleBuckets: TREE_SCALE_BUCKETS,
    inputSize: treeInputSize(hashBuckets, TREE_FEATURE_VERSION),
    maxTrainTokens: options.maxTrainTokens ?? null,
    maxVerificationTokens: options.maxVerificationTokens ?? null,
    teacherMode, teacherRun: teacher?.metadata.runId ?? null,
    initialRun: initial?.metadata.runId ?? null,
    excludedFamilies: ["lisp"],
    trainShard: options.trainShard ?? fileURLToPath(new URL("../data/generated/shards/train.jsonl.gz", import.meta.url)),
    verificationShard: options.verificationShard ?? fileURLToPath(new URL("../data/generated/shards/verification.jsonl.gz", import.meta.url)),
    miningShard: options.miningShard ?? fileURLToPath(new URL("../data/generated/shards/mining.jsonl.gz", import.meta.url)),
    output: options.output ?? runsRoot,
  };
  config.initialIsBaseline = Boolean(initial && config.fixedBaseline?.runId === initial.metadata.runId);
  if (!Number.isFinite(config.boundaryLossMultiplier) || config.boundaryLossMultiplier < 1) {
    throw new RangeError("boundary loss multiplier must be at least 1");
  }
  if (config.maxHardReplayRepeats !== null &&
      (!Number.isSafeInteger(config.maxHardReplayRepeats) || config.maxHardReplayRepeats < 1)) {
    throw new RangeError("max replay repeats must be a positive integer");
  }
  let migrated = null;
  if (migrationSource) {
    try {
      validateTreeMigrationSource(migrationSource, config);
      migrated = migrateTreeFeatureV2Model(migrationSource.model, config);
      config.initialRun = migrationSource.metadata.runId;
    } catch (error) {
      log(`cannot migrate ${migrationSource.metadata.runId}: ${error.message}; starting fresh`);
      migrationSource = null;
    }
  }
  if (!teacherMode) log(initial
    ? `warm-starting from promoted tree checkpoint ${initial.metadata.runId}`
    : migrated ? `migrating promoted feature-v2 checkpoint ${migrationSource.metadata.runId} to feature v3`
      : "starting from random tree weights");
  log("loading simple-part tree corpus");
  const excludedWebsiteSources = websiteExampleSourceKeys();
  const [training, verification, trainDigest, verificationDigest] = await Promise.all([
    loadTreeShard(config.trainShard, options.maxTrainTokens ?? Infinity, config.hashBuckets,
      { requireSourceLabels: true, excludeFamilies: config.excludedFamilies,
        excludeSources: excludedWebsiteSources, featureVersion: config.featureVersion }),
    loadTreeShard(config.verificationShard, options.maxVerificationTokens ?? Infinity, config.hashBuckets,
      { requireSourceLabels: true, excludeFamilies: config.excludedFamilies,
        featureVersion: config.featureVersion }),
    digest(config.trainShard), digest(config.verificationShard),
  ]);
  assertObjectiveCoverage(training.records, config.languageObjective, "training");
  for (const warning of assertObjectiveCoverage(verification.records, config.languageObjective, "verification")) {
    log(`verification coverage warning: ${warning}; per-family promotion checks are advisory`);
  }
  if (teacher) validateTreeTeacher(teacher, config);
  if (config.fixedBaseline) config.fixedBaseline.comparisonShardSha256 = verificationDigest;
  config.fixedBaselineMetrics = await fixedBaselineMetrics(config, {
    config,
    model: TREE_MODEL, formatVersion: treeContext === "hybrid" ? TREE_HYBRID_FORMAT_VERSION : TREE_FORMAT_VERSION,
    labelSource: config.labelSource, featureVersion: TREE_FEATURE_VERSION, tokenizerVersion: TREE_TOKENIZER_VERSION,
    inputSize: config.inputSize, classNames, architecture: { lexemeHashBuckets: config.hashBuckets },
    corpus: { verification: { sha256: verificationDigest } },
  }, verification.records);
  config.languageObjective = enrollMatureLanguageGuards(
    config.languageObjective,
    config.fixedBaselineMetrics,
    active?.metadata.config?.languageObjective?.matureLanguages,
  );
  if (config.fixedBaselineMetrics) {
    config.fixedBaselineMetrics.languageObjective = config.languageObjective;
  }
  config.trainingLanguageMultipliers = agreementLanguageMultipliers(
    config.fixedBaselineMetrics,
    [...config.languageObjective.strictLanguages, ...config.languageObjective.matureLanguages],
    0.9,
    targetedFineTune ? [options.fineTuneMetadata.language] : null,
  );
  const boostedLanguages = Object.entries(config.trainingLanguageMultipliers)
    .filter(([, multiplier]) => multiplier > 1)
    .sort((left, right) => right[1] - left[1]);
  if (boostedLanguages.length) {
    log(`weak-language curriculum=${boostedLanguages.map(([language, multiplier]) =>
      `${language}:${multiplier.toFixed(2)}x`).join(", ")}`);
  }
  if (migrated) {
    if (config.fixedBaseline?.runId !== migrationSource.metadata.runId || !config.fixedBaselineMetrics) {
      throw new Error("feature-v2 migration requires the active checkpoint as its fixed verification baseline");
    }
    config.migrationExpectedAccuracy = config.fixedBaselineMetrics.accuracy;
  }
  const fine = options.naturalReplay || polishMode ? training.records : upsampleMixedRecords(
    balanceLanguageRecords(training.records, options.minimumFamilyTokens ?? 10_000), 2,
  );
  const pretrainWeights = classWeights(training.classCounts);
  const fineCounts = countClasses(fine);
  const model = initial?.model ?? migrated ?? createTreeModel({ ...config, random: createRandom(config.seed) });
  if (initial) validateTreeInitial(initial, config);
  const tensorNames = treeTensorNamesFor(model);
  let hardRecords = options.hardRecords ?? null;
  let mining = null;
  if (autoMining && config.replayFraction > 0) {
    log(`loading independent mining split for ${TREE_REPLAY_FAMILIES.join(", ")}`);
    const [miningData, miningDigest] = await Promise.all([
      loadTreeShard(config.miningShard, Infinity, config.hashBuckets, {
        requireSourceLabels: true, excludeFamilies: config.excludedFamilies,
        includeFamilies: TREE_REPLAY_FAMILIES, featureVersion: config.featureVersion,
      }),
      digest(config.miningShard),
    ]);
    const packed = quantizeTensors(model, tensorNames, weightBits);
    const deployed = dequantizeTensors(packed.data, packed.metadata);
    const replay = mineTreeReplay(miningData.records, deployed, config);
    hardRecords = replay.records;
    mining = {
      split: "mining", selection: "deployed-error-contribution",
      sha256: miningDigest, checkpoint: initial.metadata.runId,
      tokens: miningData.tokenCount, files: miningData.fileCount,
      sources: miningData.sourceNames, ...replay.stats,
    };
    config.replayMining = mining;
    log(`mined ${replay.stats.selectedWindows.toLocaleString("en-US")} error-ranked windows / ` +
      `${replay.stats.selectedParts.toLocaleString("en-US")} parts; replay ceiling ` +
      `${percent(config.replayFraction)} with at most ${config.maxHardReplayRepeats} passes per window`);
  }
  const parameterCount = tensorNames.reduce((sum, name) => sum + model[name].length, 0);
  const packedBytes = Math.ceil(parameterCount * weightBits / 8);
  if (!teacherMode && (parameterCount > TREE_PARAMETER_CEILING || packedBytes > TREE_PACKED_BYTE_BUDGET)) {
    throw new RangeError(`tree shape needs ${parameterCount.toLocaleString("en-US")} parameters / ` +
      `${packedBytes.toLocaleString("en-US")} packed bytes; limits are ` +
      `${TREE_PARAMETER_CEILING.toLocaleString("en-US")} / ${TREE_PACKED_BYTE_BUDGET.toLocaleString("en-US")}`);
  }
  log(`architecture=hierarchical-tree (${treeContext}; hidden=${config.hiddenSize}; ` +
    `classifier=${config.classifierSize}; word-hash=${config.hashBuckets}; ${config.precision}; ` +
    `${parameterCount.toLocaleString("en-US")} parameters / ` +
    `${teacherMode ? parameterCount * 4 : packedBytes} ${teacherMode ? "float" : "packed"} bytes)`);
  const supervisedTrain = countSupervised(training.records);
  const supervisedVerification = countSupervised(verification.records);
  log(`loaded train=${training.tokenCount.toLocaleString("en-US")} parts ` +
    `(${supervisedTrain.toLocaleString("en-US")} scored) / verification=${verification.tokenCount.toLocaleString("en-US")} parts ` +
    `(${supervisedVerification.toLocaleString("en-US")} scored); whitespace is context-only`);
  if (teacher) log(`distilling ${teacher.metadata.runId} at ${(config.distillationWeight * 100).toFixed(0)}% ` +
    `soft-target weight / temperature ${config.distillationTemperature}`);
  if (polishMode) log(`continuation schedule=${schedule.epochs - schedule.agreementEpochs - schedule.calibrationEpochs} ` +
    `replay-polish + ${schedule.agreementEpochs} agreement + ${schedule.calibrationEpochs} classifier-calibration epochs; ` +
    `agreement learning rate ${config.agreementLearningRate} -> ${config.agreementFinalLearningRate}; ` +
    `active checkpoint is epoch 0`);
  const result = await runTorchTraining({
    runtime, config, model, pretrainRecords: training.records, fineTuneRecords: fine,
    hardRecords, requiredRecords: options.requiredRecords ?? null,
    verificationRecords: verification.records,
    pretrainClassWeights: pretrainWeights, fineTuneClassWeights: classWeights(fineCounts),
    pretrainAuxiliaryPositiveWeights: balancedAuxiliaryPositiveWeights(training.records, undefined, treeAuxiliaryNames.length),
    fineTuneAuxiliaryPositiveWeights: balancedAuxiliaryPositiveWeights(fine, undefined, treeAuxiliaryNames.length),
    classNames, auxiliaryNames: treeAuxiliaryNames, sameVerification: Boolean(initial || migrated),
    teacherModel: teacher?.model,
    teacherConfig: teacher ? {
      model: TREE_MODEL, inputSize: teacher.metadata.inputSize, hiddenSize: teacher.metadata.hiddenSize,
      classifierSize: teacher.metadata.architecture.classifierDimensions,
      weightBits: teacher.metadata.config?.weightBits ?? teacher.metadata.quantization?.bits ?? 6,
      treeContext: teacher.metadata.config?.treeContext ?? "tree",
      teacherFloat: teacher.metadata.config?.teacherMode === true,
    } : null,
    log,
  });
  const selected = result.best.model;
  const quantized = teacherMode ? null : quantizeTensors(selected, tensorNames, weightBits);
  const run = await createRunDirectory(config.output);
  const floatWeights = concatenate(selected, tensorNames);
  const metadata = {
    formatVersion: treeContext === "hybrid" ? TREE_HYBRID_FORMAT_VERSION : TREE_FORMAT_VERSION,
    precision: config.precision, labelSource: config.labelSource,
    model: TREE_MODEL, featureVersion: TREE_FEATURE_VERSION,
    tokenizerVersion: TREE_TOKENIZER_VERSION, createdAt: run.createdAt, runId: run.id,
    selectedEpoch: result.best.epoch, selectedSource: result.best.source ?? "raw",
    config, inputSize: config.inputSize, hiddenSize: config.hiddenSize,
    architecture: {
      direction: "bidirectional", tree: "scale-aware-butterfly-binary", blockParts: 32,
      classifierDimensions: config.classifierSize, lexemeHashBuckets: config.hashBuckets,
      secondaryLexemeHashBuckets: TREE_SECONDARY_HASH_BUCKETS,
      neighborSymbolHashBuckets: TREE_SYMBOL_PAIR_BUCKETS,
      scaleBuckets: config.scaleBuckets, localNeighborParts: config.localRadius,
      context: treeContext === "hybrid" ? "local-affine-tree" : "tree", localRadius: config.localRadius,
      auxiliaryStates: treeAuxiliaryNames, classifierAuxiliaryStates: true,
    },
    acceptance: teacherMode
      ? { accepted: false, criterion: "offline-distillation-teacher" }
      : { accepted: false, criterion: config.selectionMetric === "accuracy"
        ? "pending-fixed-baseline-untouched-verification-accuracy"
        : "pending-fixed-baseline-language-weighted-error" },
    classNames, auxiliaryNames: treeAuxiliaryNames, parameterCount,
    runtimeParameterCount: quantized?.parameterCount ?? null,
    runtimeWeightBytes: quantized?.data.byteLength ?? null,
    tensorLayout: treeTensorLayout(selected, config), quantization: quantized?.metadata ?? null,
    corpus: {
      train: { sha256: trainDigest, tokens: training.tokenCount, scoredTokens: supervisedTrain, files: training.fileCount, streams: training.records.length, sources: training.sourceNames },
      verification: { sha256: verificationDigest, tokens: verification.tokenCount, scoredTokens: supervisedVerification, files: verification.fileCount, streams: verification.records.length, sources: verification.sourceNames },
      ...(mining ? { mining } : {}),
    },
    history: result.history, verification: result.best.metrics, selection: result.selection,
    ...(options.fineTuneMetadata ? { fineTune: options.fineTuneMetadata } : {}),
    ...(teacher ? { distillation: {
      teacherRunId: teacher.metadata.runId, weight: config.distillationWeight,
      temperature: config.distillationTemperature,
    } } : {}),
  };
  await Promise.all([
    writeFile(resolve(run.path, `model-${run.id}.json`), `${JSON.stringify(metadata, null, 2)}\n`, { flag: "wx" }),
    writeFile(resolve(run.path, `weights-f32-${run.id}.bin`), floatWeights, { flag: "wx" }),
    ...(quantized ? [writeFile(resolve(run.path, `weights-int${weightBits}-${run.id}.bin`), quantized.data, { flag: "wx" })] : []),
    writeFile(resolve(run.path, `history-${run.id}.jsonl`), result.history.map(JSON.stringify).join("\n") + "\n", { flag: "wx" }),
  ]);
  if (result.recoveryDirectory) await rm(result.recoveryDirectory, { recursive: true, force: true });
  log(`saved ${teacherMode ? "offline tree teacher" : "tree experiment"} ${run.id} to ${run.path}`);
  if (result.selection?.status === "diagnostic-only") {
    log(`diagnostic only; checkpoint selection guards failed: ${result.selection.failures.join("; ")}`);
  }
  log(`${teacherMode ? `float32=${floatWeights.byteLength}` : `int${weightBits}=${quantized.data.byteLength}`} bytes / ` +
    `weighted error=${percent(metadata.verification.weightedError)} / accuracy=${percent(metadata.verification.accuracy)} / ` +
    `styled macro F1=${percent(metadata.verification.macroF1)} / false color=${percent(metadata.verification.falseColorRate)}`);
  const majorLanguages = majorLanguageErrorRows(metadata.verification, config.languageObjective);
  log(`major-language held-out error (${majorLanguages.length} language variants):`);
  for (const row of majorLanguages) {
    log(`  ${row.language}${row.family === row.language ? "" : ` (${row.family})`}: ` +
      `error=${percent(row.errorRate)} hard-guard=${row.hardGuard ? "yes" : "no"}`);
  }
  return { path: run.path, metadata, selection: result.selection, model: selected };
}

export function treeTrainingSchedule(options = {}, polishMode = false, targetedFineTune = false) {
  const epochs = options.epochs ?? (polishMode ? 10 : 32);
  const calibrationEpochs = options.calibrationEpochs ?? 2;
  const fineTuneEpochs = options.fineTuneEpochs ?? (polishMode ? calibrationEpochs : 4);
  const agreementEpochs = options.agreementEpochs ?? (targetedFineTune ? 0 : 6);
  return { epochs, fineTuneEpochs, agreementEpochs, calibrationEpochs };
}

export function countClasses(records) {
  const counts = new Uint32Array(classNames.length);
  for (const record of records) for (let i = 0; i < record.targets.length; i++) {
    if (record.supervisionWeights?.[i] === 0) continue;
    counts[record.targets[i]] += 1;
  }
  return counts;
}

/** Give weak, unguarded languages a gentle curriculum without allowing one
 * scarce language to dominate a converged model. A language at 80% agreement
 * receives 2x loss and one at 70% or below reaches the 3x cap. Guarded
 * languages retain their natural 1x weight.
 */
export function agreementLanguageMultipliers(
  metrics, guardedLanguages = [], targetAccuracy = 0.9, targetLanguages = null,
) {
  if (metrics == null) return {};
  if (!Number.isFinite(targetAccuracy) || targetAccuracy <= 0 || targetAccuracy >= 1) {
    throw new RangeError("target agreement must be between zero and one");
  }
  const guarded = new Set(guardedLanguages);
  const targets = targetLanguages == null ? null : new Set(targetLanguages);
  return Object.fromEntries(Object.entries(metrics.perLanguage ?? {}).sort(([left], [right]) =>
    left.localeCompare(right)).map(([language, value]) => {
    const support = value.support ?? 0;
    const accuracy = support ? 1 - value.errors / support : value.accuracy;
    if (!Number.isFinite(accuracy) || accuracy < 0 || accuracy > 1) {
      throw new Error(`invalid baseline agreement for ${language}`);
    }
    const multiplier = guarded.has(language) || (targets && !targets.has(language)) || accuracy >= targetAccuracy
      ? 1
      : 1 + Math.min(2, (targetAccuracy - accuracy) / (1 - targetAccuracy));
    return [language, multiplier];
  }));
}

export function focusedReplayConfig(options = {}) {
  const focusedReplay = options.focusedReplay === true;
  const focusedReplaySteps = options.focusedReplaySteps ?? 1;
  const focusedReplayLearningRate = options.focusedReplayLearningRate ?? 0.001;
  const focusedReplayFinalLearningRate = options.focusedReplayFinalLearningRate ?? 0.0005;
  if (!Number.isSafeInteger(focusedReplaySteps) || focusedReplaySteps < 1 || focusedReplaySteps > 4) {
    throw new RangeError("focused replay steps must be between 1 and 4");
  }
  if (!Number.isFinite(focusedReplayLearningRate) || !Number.isFinite(focusedReplayFinalLearningRate) ||
      focusedReplayLearningRate <= 0 || focusedReplayFinalLearningRate <= 0 ||
      focusedReplayFinalLearningRate > focusedReplayLearningRate) {
    throw new RangeError("focused replay learning rates must be positive and final must not exceed initial");
  }
  return { focusedReplay, focusedReplaySteps, focusedReplayLearningRate, focusedReplayFinalLearningRate };
}

export function enrollMatureLanguageGuards(objective, verification, previous = []) {
  const strict = new Set(objective.strictLanguages);
  const mature = new Set(previous ?? objective.matureLanguages ?? []);
  for (const row of majorLanguageErrorRows(verification, objective)) {
    const metrics = verification.perLanguage[row.language];
    if (!strict.has(row.language) && row.errorRate < 0.1 &&
        row.support >= objective.minSupport &&
        (metrics.plainSupport ?? 0) >= objective.minPlainSupport) mature.add(row.language);
  }
  for (const language of strict) mature.delete(language);
  return {
    ...objective,
    matureLanguages: [...mature].sort(),
    maxMatureErrorIncrease: objective.maxMatureErrorIncrease ?? 0.01,
    maxMatureFalseColorIncrease: objective.maxMatureFalseColorIncrease ?? 0.01,
  };
}

export function majorLanguageErrorRows(verification, objective) {
  const majorFamilies = new Set(objective?.protectedFamilies ?? []);
  const hardGuards = new Set([
    ...(objective?.strictLanguages ?? []),
    ...(objective?.matureLanguages ?? []),
  ]);
  return Object.entries(verification?.perLanguage ?? {}).map(([language, metrics]) => {
    const family = languageFamily(language);
    const support = metrics.support ?? 0;
    const errorRate = support ? metrics.errors / support
      : Number.isFinite(metrics.accuracy) ? 1 - metrics.accuracy : null;
    return { language, family, support, errorRate, hardGuard: hardGuards.has(language) };
  }).filter((row) => majorFamilies.has(row.family) && row.support > 0 && Number.isFinite(row.errorRate))
    .sort((left, right) =>
      (objective.familyWeights[right.family] ?? 0) - (objective.familyWeights[left.family] ?? 0) ||
      left.family.localeCompare(right.family) || left.language.localeCompare(right.language));
}

function countSupervised(records) {
  let count = 0;
  for (const record of records) for (const weight of record.supervisionWeights) count += Number(weight > 0);
  return count;
}

function classWeights(counts) {
  const total = counts.reduce((sum, count) => sum + count, 0);
  const values = Float32Array.from(counts, (count) => Math.min(4, Math.sqrt(total / (counts.length * Math.max(1, count)))));
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Float32Array.from(values, (value) => value / mean);
}

export function validateTreeTeacher(teacher, config) {
  const metadata = teacher.metadata;
  if (metadata.model !== TREE_MODEL || ![TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION].includes(metadata.formatVersion) ||
      metadata.labelSource !== config.labelSource ||
      metadata.featureVersion !== TREE_FEATURE_VERSION || metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION ||
      metadata.inputSize !== config.inputSize ||
      metadata.architecture?.lexemeHashBuckets !== config.hashBuckets ||
      metadata.architecture?.scaleBuckets !== config.scaleBuckets ||
      metadata.architecture?.tree !== "scale-aware-butterfly-binary" ||
      JSON.stringify(metadata.classNames) !== JSON.stringify(classNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(treeAuxiliaryNames) ||
      JSON.stringify(Object.keys(teacher.model)) !== JSON.stringify(treeTensorNamesFor(teacher.model))) {
    throw new Error("tree distillation teacher is incompatible with the current model features");
  }
}

function validateTreeInitial(initial, config) {
  const metadata = initial.metadata;
  const expectedFormat = config.treeContext === "hybrid" ? TREE_HYBRID_FORMAT_VERSION : TREE_FORMAT_VERSION;
  if (metadata.model !== TREE_MODEL || metadata.formatVersion !== expectedFormat ||
      metadata.labelSource !== config.labelSource || metadata.featureVersion !== TREE_FEATURE_VERSION ||
      metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION || metadata.inputSize !== config.inputSize ||
      metadata.hiddenSize !== config.hiddenSize ||
      metadata.architecture?.classifierDimensions !== config.classifierSize ||
      metadata.architecture?.lexemeHashBuckets !== config.hashBuckets ||
      metadata.architecture?.scaleBuckets !== config.scaleBuckets ||
      JSON.stringify(metadata.classNames) !== JSON.stringify(classNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(treeAuxiliaryNames)) {
    throw new Error("initial tree checkpoint is incompatible with the current model or untouched verification corpus");
  }
  const expected = createTreeModel({ ...config, random: () => 0.5 });
  for (const name of treeTensorNamesFor(expected)) {
    if (initial.model[name]?.length !== expected[name].length) {
      throw new Error(`initial tree tensor ${name} is incompatible with the requested shape`);
    }
  }
}

function validateTreeMigrationSource(source, config) {
  const metadata = source.metadata;
  const expectedFormat = config.treeContext === "hybrid" ? TREE_HYBRID_FORMAT_VERSION : TREE_FORMAT_VERSION;
  if (metadata.model !== TREE_MODEL || metadata.formatVersion !== expectedFormat ||
      metadata.labelSource !== config.labelSource || metadata.featureVersion !== TREE_LEGACY_FEATURE_VERSION ||
      metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION ||
      metadata.inputSize !== treeInputSize(128, TREE_LEGACY_FEATURE_VERSION) ||
      metadata.hiddenSize !== config.hiddenSize ||
      metadata.architecture?.classifierDimensions !== config.classifierSize ||
      metadata.architecture?.lexemeHashBuckets !== 128 ||
      metadata.architecture?.scaleBuckets !== TREE_LEGACY_SCALE_BUCKETS ||
      config.featureVersion !== TREE_FEATURE_VERSION || config.hashBuckets !== TREE_HASH_BUCKETS ||
      config.scaleBuckets !== TREE_SCALE_BUCKETS ||
      JSON.stringify(metadata.classNames) !== JSON.stringify(classNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(treeAuxiliaryNames)) {
    throw new Error("only the matching 128-hash/8-scale feature-v2 tree can migrate losslessly to feature v3");
  }
}

export function treeWarmStartMismatch(metadata, options = {}) {
  if (metadata.model !== TREE_MODEL || ![TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION].includes(metadata.formatVersion)) {
    return "the promoted model is not a supported tree architecture";
  }
  if (metadata.labelSource !== "shiki-spans-v1" || metadata.featureVersion !== TREE_FEATURE_VERSION ||
      metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION ||
      JSON.stringify(metadata.classNames) !== JSON.stringify(classNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(treeAuxiliaryNames)) {
    return "the promoted model uses incompatible labels or features";
  }
  const currentContext = metadata.config?.treeContext ??
    (metadata.formatVersion === TREE_HYBRID_FORMAT_VERSION ? "hybrid" : "tree");
  const requested = [
    ["context", options.treeContext, currentContext],
    ["hidden", options.hiddenSize, metadata.hiddenSize],
    ["classifier", options.classifierSize, metadata.architecture?.classifierDimensions],
    ["hash", options.hashBuckets, metadata.architecture?.lexemeHashBuckets],
  ];
  const changed = requested.filter(([, value, current]) => value !== undefined && value !== current)
    .map(([name, value, current]) => `${name} ${current} -> ${value}`);
  return changed.length ? `requested architecture changes shape (${changed.join(", ")})` : null;
}

function concatenate(model, names) {
  const values = new Float32Array(names.reduce((sum, name) => sum + model[name].length, 0));
  let offset = 0;
  for (const name of names) { values.set(model[name], offset); offset += model[name].length; }
  return new Uint8Array(values.buffer);
}

async function createRunDirectory(root) {
  const createdAt = new Date().toISOString();
  const id = createdAt.replaceAll(/[-:]/g, "");
  const path = resolve(root, id);
  await mkdir(path, { recursive: false });
  return { id, path, createdAt };
}

async function digest(path) {
  const { readFile } = await import("node:fs/promises");
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function createRandom(seed) {
  let state = seed >>> 0 || 1;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
}

function parse(arguments_) {
  const result = {};
  const integer = new Set(["agreementEpochs", "calibrationEpochs", "epochs", "fineTuneEpochs", "hiddenSize", "classifierSize", "hashBuckets", "batchTokens", "patience", "seed", "progressEvery", "qatEpochs", "emaStartEpoch", "maxTrainTokens", "maxVerificationTokens", "minimumFamilyTokens", "weightBits", "distillationWarmupEpochs", "distillationRampEpochs", "maxHardReplayRepeats"]);
  const aliases = { "language-objective": "languageObjectivePath", "baseline-run": "baselineRun", "initial-run": "initialRun", "class-weight-power": "classWeightPower", "agreement-epochs": "agreementEpochs", "calibration-epochs": "calibrationEpochs", context: "treeContext", "train-shard": "trainShard", "verification-shard": "verificationShard", "mining-shard": "miningShard", epochs: "epochs", "fine-tune-epochs": "fineTuneEpochs", hidden: "hiddenSize", classifier: "classifierSize", hash: "hashBuckets", bits: "weightBits", batch: "batchTokens", patience: "patience", seed: "seed", "progress-every": "progressEvery", "qat-epochs": "qatEpochs", "ema-start-epoch": "emaStartEpoch", "max-train-tokens": "maxTrainTokens", "max-verification-tokens": "maxVerificationTokens", "min-family-tokens": "minimumFamilyTokens", "boundary-loss": "boundaryLossMultiplier", "replay-fraction": "replayFraction", "max-replay-repeats": "maxHardReplayRepeats", teacher: "teacherMode", fresh: "fresh", "teacher-run": "teacherRun", "distillation-weight": "distillationWeight", "distillation-temperature": "distillationTemperature", "distillation-warmup": "distillationWarmupEpochs", "distillation-ramp": "distillationRampEpochs", device: "device", python: "python", output: "output", lr: "learningRate", "final-lr": "finalLearningRate", "agreement-lr": "agreementLearningRate", "agreement-final-lr": "agreementFinalLearningRate", "calibration-lr": "calibrationLearningRate", "calibration-final-lr": "calibrationFinalLearningRate", "failure-lr": "focusedReplayLearningRate", "failure-final-lr": "focusedReplayFinalLearningRate" };
  for (let index = 0; index < arguments_.length; index++) {
    if (arguments_[index] === "--") continue;
    const [raw, inline] = arguments_[index].replace(/^--/, "").split("=", 2);
    const name = aliases[raw];
    if (!name) throw new Error(`unknown option --${raw}`);
    if (name === "teacherMode" || name === "fresh") { result[name] = true; continue; }
    const value = inline ?? arguments_[++index];
    result[name] = integer.has(name) ? Number.parseInt(value, 10)
      : ["classWeightPower", "learningRate", "finalLearningRate", "agreementLearningRate", "agreementFinalLearningRate", "calibrationLearningRate", "calibrationFinalLearningRate", "focusedReplayLearningRate", "focusedReplayFinalLearningRate", "replayFraction", "boundaryLossMultiplier", "distillationWeight", "distillationTemperature"].includes(name) ? Number(value) : value;
  }
  return result;
}

function log(message) { console.log(`[${new Date().toISOString()}] ${message}`); }
function percent(value) { return `${(value * 100).toFixed(2)}%`; }

export function shouldAutoPromoteTreeRun(result) {
  const baseline = result.metadata.config.fixedBaselineMetrics;
  return result.metadata.config.teacherMode !== true && result.selection.status === "eligible" &&
    baseline != null && result.metadata.verification.accuracy > baseline.accuracy;
}

const invoked = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (import.meta.url === invoked) try {
  const result = await trainTree(parse(process.argv.slice(2)));
  if (shouldAutoPromoteTreeRun(result)) {
    const { promote } = await import("./promote.js");
    await promote(result.path);
  }
} catch (error) { console.error(error instanceof Error ? error.stack : error); process.exitCode = 1; }

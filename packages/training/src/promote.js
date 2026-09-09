import { readFile, readdir, rename, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { classNames } from "./classes.js";
import { assertSameProvenance, compareLanguageMetrics, validateLanguageObjective } from "./language-objective.js";
import { loadFloatCheckpoint, writeActiveCheckpoint } from "./checkpoint.js";
import {
  dequantizeTensors, encodeWeightSymbols, quantizeTensors, WEIGHT_SYMBOL_ENCODING,
} from "./quantization.js";
import {
  TREE_FEATURE_VERSION, TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION, TREE_MODEL, TREE_TOKENIZER_VERSION,
  evaluateTree, loadTreeShard, treeTensorNamesFor, treeAuxiliaryNames,
} from "./tree-model.js";
import {
  TREE_SECONDARY_HASH_BUCKETS, TREE_SYMBOL_PAIR_BUCKETS,
  isTreeFeatureVersion, treeInputSize, treeScaleBuckets,
} from "../../core/src/tree-features.js";
import { isHybridTree, runtimeTensorLayout } from "../../core/src/model-layout.js";
import { generatedShaderContents } from "../../core/scripts/generate-shader.js";
import { promotedModelContents, runtimeModelContents } from "../../core/scripts/generate-runtime-model.js";

const runsRoot = fileURLToPath(new URL("../runs/", import.meta.url));
const generatedModel = fileURLToPath(new URL("../../core/src/model.generated.js", import.meta.url));
const generatedShader = fileURLToPath(new URL("../../core/src/shader.min.generated.js", import.meta.url));
const generatedRuntimeModel = fileURLToPath(new URL("../../core/src/model.runtime.generated.js", import.meta.url));
const generatedWebsiteStats = fileURLToPath(new URL("../../../apps/website/app/model-stats.generated.ts", import.meta.url));
const generatedWebsiteCorrectness = fileURLToPath(new URL("../../../apps/website/app/correctness.generated.ts", import.meta.url));
const generatedRunPointer = fileURLToPath(new URL("./promoted-run.generated.js", import.meta.url));
const correctnessBenchmark = fileURLToPath(new URL("../../benchmark/src/correctness.js", import.meta.url));
const runFile = promisify(execFile);
const MAX_TREE_PARAMETERS = 75_000;
const MAX_TREE_PACKED_BYTES = 37_500;

export async function promote(runArgument, { force = false } = {}) {
  if (!runArgument) {
    throw new Error("run ID or path is required; usage: pnpm model:promote <run-id-or-path> [--force]");
  }
  const runPath = resolveRunPath(runArgument);
  const entries = await import("node:fs/promises").then(({ readdir }) => readdir(runPath));
  const metadataName = one(entries.filter((name) => /^model-.*\.json$/.test(name)), "model metadata");
  const metadata = JSON.parse(await readFile(resolve(runPath, metadataName), "utf8"));
  const teacherRejection = teacherPromotionRejection(metadata);
  if (teacherRejection) throw new Error(teacherRejection);
  if (!isTreeModel(metadata)) {
    throw new Error(`unsupported model ${metadata.model} format ${metadata.formatVersion}; only the tree runtime can be promoted`);
  }
  const checkpoint = await loadFloatCheckpoint(runPath);
  const runtimeTensors = treeTensorNamesFor(checkpoint.model);
  const weightBits = metadata.config?.weightBits ?? metadata.quantization?.bits ?? 6;
  const quantized = quantizeTensors(checkpoint.model, runtimeTensors, weightBits);
  const weights = quantized.data;
  const verificationShard = metadata.config?.verificationShard ??
    fileURLToPath(new URL("../data/generated/shards/verification.jsonl.gz", import.meta.url));
  const deployed = dequantizeTensors(weights, quantized.metadata);
  const comparison = await evaluateTreePromotion(metadata, deployed, verificationShard, { force });
  const verification = comparison.candidate;
  const runtimeMetadata = {
    ...metadata,
    runtimeParameterCount: quantized.parameterCount,
    runtimeWeightBytes: weights.byteLength,
    quantization: quantized.metadata,
    verification,
    promotionComparison: comparison, acceptance: comparison.decision,
  };
  validateModel(runtimeMetadata, weights);

  const promotedAt = new Date().toISOString();
  const model = {
    formatVersion: metadata.formatVersion,
    model: metadata.model,
    featureVersion: metadata.featureVersion,
    inputSize: metadata.inputSize,
    hiddenSize: metadata.hiddenSize,
    architecture: metadata.architecture ?? { direction: "forward", lookaheadTokens: 0 },
    outputSize: metadata.classNames.length,
    quantization: quantized.metadata,
    weightEncoding: WEIGHT_SYMBOL_ENCODING,
    weights: encodeWeightSymbols(weights, quantized.parameterCount, weightBits),
  };
  const contents = promotedModelContents(model);
  const shaderContents = await generatedShaderContents(model);
  const runtimeContents = runtimeModelContents(model);
  const runtimeParameterCount = isHybridTree(model)
    ? runtimeTensorLayout(model).reduce((total, tensor) => total + tensor.length, 0)
    : quantized.parameterCount;
  const websiteStats = websiteStatsContents({
    ...metadata,
    verification,
    runtimeParameterCount,
    runtimeWeightBytes: Math.ceil(runtimeParameterCount * weightBits / 8),
  });
  const runPointer = `// Generated by \`pnpm model:promote\`. Do not edit by hand.\n` +
    `export const promotedRunId = ${JSON.stringify(metadata.runId)};\n` +
    `export const promotedRunPath = new URL("../active/", import.meta.url).pathname;\n`;
  const nonce = `${process.pid}.${Date.now()}`;
  const temporaryModel = `${generatedModel}.${nonce}.tmp`;
  const temporaryShader = `${generatedShader}.${nonce}.tmp`;
  const temporaryRuntimeModel = `${generatedRuntimeModel}.${nonce}.tmp`;
  const temporaryWebsiteStats = `${generatedWebsiteStats}.${nonce}.tmp`;
  const temporaryWebsiteCorrectness = `${generatedWebsiteCorrectness}.${nonce}.tmp`;
  const temporaryRunPointer = `${generatedRunPointer}.${nonce}.tmp`;
  await Promise.all([
    writeFile(resolve(runPath, `promotion-${nonce}.json`),
      JSON.stringify({ promotedAt, runId: metadata.runId, config: metadata.config, ...comparison }, null, 2) + "\n", { flag: "wx" }),
    writeFile(temporaryModel, contents, { flag: "wx" }),
    writeFile(temporaryShader, shaderContents, { flag: "wx" }),
    writeFile(temporaryRuntimeModel, runtimeContents, { flag: "wx" }),
    writeFile(temporaryWebsiteStats, websiteStats, { flag: "wx" }),
    writeFile(temporaryRunPointer, runPointer, { flag: "wx" }),
  ]);
  const correctness = await runFile(process.execPath, [correctnessBenchmark,
    "--run", runPath, "--verification", verificationShard,
    "--output", temporaryWebsiteCorrectness], { maxBuffer: 1024 * 1024 });
  await rename(temporaryWebsiteStats, generatedWebsiteStats);
  await rename(temporaryWebsiteCorrectness, generatedWebsiteCorrectness);
  await rename(temporaryShader, generatedShader);
  await rename(temporaryRuntimeModel, generatedRuntimeModel);
  await rename(temporaryModel, generatedModel);
  await writeActiveCheckpoint(runtimeMetadata, checkpoint.model, weights);
  await rename(temporaryRunPointer, generatedRunPointer);
  console.log(`[${promotedAt}] promoted ${metadata.runId}`);
  console.log(`source: ${runPath}`);
  console.log(`target: ${generatedModel}`);
  console.log(`website stats: ${generatedWebsiteStats}`);
  console.log(correctness.stdout.trim());
  console.log(`training pointer: ${generatedRunPointer}`);
  console.log(`int${weightBits} weights: ${weights.length} packed bytes / ${quantized.parameterCount} parameters / verification accuracy: ${percent(verification.accuracy)} / styled macro F1: ${percent(verification.macroF1)}`);
  return model;
}

// Read-only evaluation; promotion does not trust stored candidate/baseline scores.
export async function evaluateTreePromotion(metadata, deployed, verificationShard, { force = false } = {}) {
  const objective = validateLanguageObjective(metadata.config?.languageObjective);
  const fixed = metadata.config?.fixedBaseline;
  if (!fixed?.run || !fixed.runId || !/^[a-f0-9]{64}$/.test(fixed.weightsSha256 ?? "")) {
    throw new Error("tree promotion requires pinned config.fixedBaseline { run, runId, weightsSha256 }");
  }
  const baselinePath = resolveRunPath(fixed.run);
  const entries = await readdir(baselinePath);
  const baselineMetadata = JSON.parse(await readFile(resolve(baselinePath,
    one(entries.filter((name) => /^model-.*\.json$/.test(name)), "baseline metadata")), "utf8"));
  if (baselineMetadata.runId !== fixed.runId || baselineMetadata.runId === metadata.runId) {
    throw new Error("fixed baseline identity mismatch or candidate is its own baseline");
  }
  const baselineBytes = await readFile(resolve(baselinePath,
    one(entries.filter((name) => /^weights-int[456]-.*\.bin$/.test(name)), "stored packed baseline")));
  if (digest(baselineBytes) !== fixed.weightsSha256) throw new Error("fixed baseline weights digest mismatch");
  const shardSha256 = digest(await readFile(verificationShard));
  validateTreeComparisonProvenance(metadata, baselineMetadata, shardSha256);
  const baselineTeacherRejection = teacherPromotionRejection(baselineMetadata);
  if (baselineTeacherRejection) throw new Error(`invalid fixed baseline: ${baselineTeacherRejection}`);
  const quantization = baselineMetadata.quantization;
  if (!quantization || ![4, 5, 6].includes(quantization.bits) ||
      baselineBytes.length !== quantization.packedByteLength) throw new Error("invalid packed baseline artifact");
  // Layout validation is independent of an old run's acceptance decision/metrics.
  validateTreeModel({ ...baselineMetadata, verification: { accuracy: 0, macroF1: 0 } }, baselineBytes);
  const baselineModel = dequantizeTensors(baselineBytes, quantization);
  const maxTokens = metadata.config?.maxVerificationTokens ?? Infinity;
  if (!(maxTokens === Infinity || Number.isSafeInteger(maxTokens) && maxTokens > 0)) throw new Error("invalid verification token limit");
  const excludeFamilies = metadata.config?.excludedFamilies ?? [];
  if (!Array.isArray(excludeFamilies) || excludeFamilies.some((family) => typeof family !== "string")) {
    throw new Error("invalid excluded training families");
  }
  const { records } = await loadTreeShard(verificationShard, maxTokens,
    metadata.architecture.lexemeHashBuckets, { requireSourceLabels: true, excludeFamilies,
      featureVersion: metadata.featureVersion });
  const sameFeatures = baselineMetadata.featureVersion === metadata.featureVersion &&
    baselineMetadata.architecture.lexemeHashBuckets === metadata.architecture.lexemeHashBuckets;
  const baselineRecords = sameFeatures
    ? records : (await loadTreeShard(verificationShard, maxTokens,
      baselineMetadata.architecture.lexemeHashBuckets, { requireSourceLabels: true, excludeFamilies,
        featureVersion: baselineMetadata.featureVersion })).records;
  // Detect a shard changed between hashing and loading; neither side may use legacy labels.
  if (digest(await readFile(verificationShard)) !== shardSha256) throw new Error("verification shard changed during evaluation");
  const provenance = { shardSha256, labelSource: metadata.labelSource,
    featureVersion: metadata.featureVersion, tokenizerVersion: metadata.tokenizerVersion,
    maxTokens: maxTokens === Infinity ? null : maxTokens };
  const evaluateModel = (model, description, inputRecords = records) => evaluateTree(model, inputRecords, {
    hiddenSize: description.hiddenSize, classifierSize: description.architecture.classifierDimensions,
    languageObjective: objective, evaluationProvenance: provenance,
  });
  const candidate = evaluateModel(deployed, metadata);
  const baseline = evaluateModel(baselineModel, baselineMetadata, baselineRecords);
  const weightedDecision = compareLanguageMetrics(candidate, baseline, objective);
  const accuracySelection = metadata.config?.selectionMetric === "accuracy";
  const accuracyImprovement = candidate.accuracy - baseline.accuracy;
  const automaticDecision = accuracySelection
    ? accuracyTreePromotionDecision(candidate, baseline, weightedDecision)
    : weightedDecision;
  if (!automaticDecision.accepted && !force) {
    throw new Error(`tree promotion rejected: ${automaticDecision.failures.join("; ")}`);
  }
  const decision = automaticDecision.accepted ? automaticDecision : {
    ...automaticDecision,
    accepted: true,
    forced: true,
    criterion: "explicit-user-override",
    overriddenCriterion: automaticDecision.criterion,
    overriddenFailures: automaticDecision.failures,
    failures: [],
  };
  if (decision.warnings.length) {
    console.warn(`tree promotion warnings: ${decision.warnings.join("; ")}`);
  }
  if (decision.forced) {
    console.warn(`forcing promotion despite: ${automaticDecision.failures.join("; ")}`);
  }
  return { fixedBaseline: fixed, languageObjective: objective, candidate, baseline,
    automaticDecision, decision };
}

export function accuracyTreePromotionDecision(candidate, baseline, weightedDecision) {
  const improvement = candidate.accuracy - baseline.accuracy;
  const strictFailures = weightedDecision.strictGuards
    .filter((guard) => !guard.passed)
    .map(({ language, metric, reason }) => `${language} ${metric}: ${reason}`);
  const failures = [...(improvement > 0 ? [] : ["verification accuracy did not improve"]), ...strictFailures];
  return {
    accepted: failures.length === 0,
    criterion: "fixed-baseline-untouched-verification-accuracy",
    improvement,
    candidateAccuracy: candidate.accuracy,
    baselineAccuracy: baseline.accuracy,
    guards: weightedDecision.guards,
    strictGuards: weightedDecision.strictGuards,
    failures,
    warnings: weightedDecision.warnings,
  };
}

export function validateTreeComparisonProvenance(candidate, baseline, shardSha256) {
  for (const metadata of [candidate, baseline]) {
    if (!isTreeModel(metadata) || metadata.labelSource !== "shiki-spans-v1" ||
        !isTreeFeatureVersion(metadata.featureVersion) || metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION ||
        (metadata === candidate
          ? metadata.corpus?.verification?.sha256 !== shardSha256
          : metadata.corpus?.verification?.sha256 !== shardSha256 &&
            candidate.config?.fixedBaseline?.comparisonShardSha256 !== shardSha256) ||
        !Number.isInteger(metadata.architecture?.lexemeHashBuckets) ||
        metadata.architecture.lexemeHashBuckets < 0 ||
        metadata.inputSize !== treeInputSize(metadata.architecture.lexemeHashBuckets, metadata.featureVersion) ||
        JSON.stringify(metadata.classNames) !== JSON.stringify(classNames)) {
      throw new Error("tree comparison requires compatible direct-label corpus, tokenizer and class provenance");
    }
  }
  if (candidate.config?.fixedBaseline?.comparisonShardSha256 &&
      candidate.config.fixedBaseline.comparisonShardSha256 !== shardSha256) {
    throw new Error("pinned comparison shard mismatch");
  }
  // Both packed models are freshly evaluated on this explicitly pinned shard;
  // the baseline is independently re-encoded, while the score record uses the
  // candidate experiment's feature contract as its comparison identifier.
  const provenance = (metadata) => ({ shardSha256,
    labelSource: metadata.labelSource, featureVersion: candidate.featureVersion,
    tokenizerVersion: metadata.tokenizerVersion, maxTokens: null });
  assertSameProvenance(provenance(candidate), provenance(baseline));
}

function digest(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

export function websiteStatsContents(model) {
  const languageAccuracies = Object.entries(model.verification.perLanguage ?? {})
    .filter(([, language]) => language.support > 0 && Number.isFinite(language.accuracy))
    .map(([name, language]) => ({ name, accuracy: language.accuracy }));
  const bands = [
    ["95–100%", 0.95, Infinity],
    ["90–<95%", 0.9, 0.95],
    ["80–<90%", 0.8, 0.9],
    ["70–<80%", 0.7, 0.8],
    ["60–<70%", 0.6, 0.7],
    ["50–<60%", 0.5, 0.6],
    ["<50%", -Infinity, 0.5],
  ];
  const corpusTokens = model.corpus?.train?.tokens ?? 0;
  const trainingTokensPerEpoch = Math.max(corpusTokens, ...(model.history ?? []).map((epoch) =>
    epoch.trainingTokens ?? corpusTokens + (epoch.replayTokens ?? 0)));
  const stats = {
    runId: model.runId,
    accuracy: model.verification.accuracy,
    shikiDisagreementRate: 1 - model.verification.accuracy,
    trainingTokensPerEpoch,
    modelParameters: model.runtimeParameterCount ?? model.parameterCount ?? null,
    modelWeightBytes: model.runtimeWeightBytes ?? null,
    modelWeightBits: model.quantization?.bits ?? model.config?.weightBits ?? null,
    languageAccuracyDistribution: bands.map(([range, minimum, maximum]) => ({
      range,
      languages: languageAccuracies
        .filter(({ accuracy }) => accuracy >= minimum && accuracy < maximum)
        .map(({ name }) => name)
        .sort(),
    })),
  };
  return `// Generated by \`pnpm model:promote\`. Do not edit by hand.\n` +
    `export const modelStats = Object.freeze(${JSON.stringify(stats, null, 2)});\n`;
}

function resolveRunPath(argument) {
  return argument.includes("/") || argument.startsWith(".")
    ? resolve(argument)
    : resolve(runsRoot, argument);
}

export function validateModel(metadata, weights) {
  const rejection = promotionRejection(metadata);
  if (rejection) throw new Error(rejection);
  if (!isTreeModel(metadata)) throw new Error(`unsupported model ${metadata.model} format ${metadata.formatVersion}`);
  validateTreeModel(metadata, weights);
}

function isTreeModel(metadata) {
  return metadata.model === TREE_MODEL &&
    [TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION].includes(metadata.formatVersion);
}

function validateTreeModel(metadata, weights) {
  if (!isTreeFeatureVersion(metadata.featureVersion)) throw new Error("tree feature version is not supported by the runtime");
  const architecture = metadata.architecture ?? {};
  const hashBuckets = architecture.lexemeHashBuckets ?? 128;
  const hybrid = metadata.formatVersion === TREE_HYBRID_FORMAT_VERSION;
  if (architecture.tree !== "scale-aware-butterfly-binary" || architecture.direction !== "bidirectional" ||
      architecture.blockParts !== 32 || architecture.scaleBuckets !== treeScaleBuckets(metadata.featureVersion) ||
      architecture.localNeighborParts !== (hybrid ? 2 : 1) ||
      architecture.classifierAuxiliaryStates !== true) {
    throw new Error("hierarchical-tree architecture does not match the runtime");
  }
  if (hybrid ? architecture.context !== "local-affine-tree" || architecture.localRadius !== 2
    : (architecture.context != null && architecture.context !== "tree") ||
      (architecture.localRadius != null && architecture.localRadius !== 1)) {
    throw new Error("tree context does not match its format version");
  }
  if (metadata.featureVersion >= 3 &&
      (architecture.secondaryLexemeHashBuckets !== TREE_SECONDARY_HASH_BUCKETS ||
       architecture.neighborSymbolHashBuckets !== TREE_SYMBOL_PAIR_BUCKETS)) {
    throw new Error("tree collision-reduction features do not match the runtime");
  }
  if (metadata.inputSize !== treeInputSize(hashBuckets, metadata.featureVersion)) {
    throw new Error("tree input size does not match its feature layout");
  }
  if (![32, 64].includes(metadata.hiddenSize)) {
    throw new Error("tree hidden size must be 32 or 64");
  }
  const classifier = architecture.classifierDimensions;
  if (!Number.isInteger(classifier) || classifier < 1 || classifier > 256) throw new Error("invalid tree classifier size");
  if (JSON.stringify(metadata.classNames) !== JSON.stringify(classNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(treeAuxiliaryNames) ||
      JSON.stringify(metadata.auxiliaryNames) !== JSON.stringify(architecture.auxiliaryStates)) {
    throw new Error("tree label schema does not match the runtime");
  }
  const tensors = metadata.quantization?.tensors;
  const bits = metadata.quantization?.bits;
  if (![4, 5, 6].includes(bits) || metadata.quantization?.type !== `symmetric-int${bits}-per-tensor` ||
      JSON.stringify(tensors?.map(({ name }) => name)) !==
        JSON.stringify(treeTensorNamesFor({ treeContext: hybrid ? "hybrid" : "tree" }))) {
    throw new Error("tree model must use the runtime int4, int5, or int6 tensor layout");
  }
  const auxiliary = metadata.auxiliaryNames.length;
  const expected = {
    featureEmbedding: metadata.inputSize * metadata.hiddenSize,
    neighborScale: 3 * metadata.hiddenSize, leafBias: metadata.hiddenSize,
    mergeOwnLeft: architecture.scaleBuckets * metadata.hiddenSize,
    mergeOwnRight: architecture.scaleBuckets * metadata.hiddenSize,
    mergeCrossLeft: architecture.scaleBuckets * metadata.hiddenSize,
    mergeCrossRight: architecture.scaleBuckets * metadata.hiddenSize,
    mergeBias: architecture.scaleBuckets * metadata.hiddenSize,
    downOwnParent: architecture.scaleBuckets * metadata.hiddenSize,
    downOwnSelf: architecture.scaleBuckets * metadata.hiddenSize,
    downOwnSibling: architecture.scaleBuckets * metadata.hiddenSize,
    downCrossParent: architecture.scaleBuckets * metadata.hiddenSize,
    downCrossSelf: architecture.scaleBuckets * metadata.hiddenSize,
    downCrossSibling: architecture.scaleBuckets * metadata.hiddenSize,
    downSkip: architecture.scaleBuckets * metadata.hiddenSize,
    downLeftBias: architecture.scaleBuckets * metadata.hiddenSize,
    downRightBias: architecture.scaleBuckets * metadata.hiddenSize,
    classifierInput: classifier * (metadata.hiddenSize * 2 + auxiliary), classifierBias: classifier,
    output: metadata.classNames.length * classifier, outputBias: metadata.classNames.length,
    auxiliaryOutput: auxiliary * metadata.hiddenSize * 2, auxiliaryBias: auxiliary,
    ...(hybrid ? {
      localOffsetScale: 5 * metadata.hiddenSize,
      localNonspaceScale: 2 * metadata.hiddenSize,
      stateInput: metadata.hiddenSize ** 2, stateInputBias: metadata.hiddenSize,
      stateGate: metadata.hiddenSize ** 2, stateGateBias: metadata.hiddenSize,
      stateMix: metadata.hiddenSize * 2 * metadata.hiddenSize, stateMixBias: metadata.hiddenSize,
    } : {}),
  };
  let end = 0;
  for (const tensor of tensors) {
    if (tensor.offset !== end || tensor.length !== expected[tensor.name] || !(tensor.scale > 0)) {
      throw new Error(`invalid tree tensor ${tensor.name}`);
    }
    end += tensor.length;
  }
  const expectedBytes = Math.ceil(end * bits / 8);
  if (end > MAX_TREE_PARAMETERS || expectedBytes > MAX_TREE_PACKED_BYTES || metadata.runtimeParameterCount !== end ||
      metadata.quantization.parameterCount !== end || metadata.runtimeWeightBytes !== expectedBytes ||
      metadata.quantization.packedByteLength !== expectedBytes || weights.length !== expectedBytes) {
    throw new Error(`tree int${bits} layout does not match ${end} parameters in ${expectedBytes} bytes`);
  }
  if (!metadata.verification || !Number.isFinite(metadata.verification.accuracy) ||
      !Number.isFinite(metadata.verification.macroF1)) throw new Error("tree model has no verification metrics");
}

function teacherPromotionRejection(metadata) {
  if (metadata.config?.teacherMode === true || metadata.precision === "float32" ||
      metadata.config?.precision === "float32") {
    return "teacher/float32 checkpoints cannot be promoted to the packed runtime";
  }
  return null;
}

export function promotionRejection(metadata) {
  const teacherRejection = teacherPromotionRejection(metadata);
  if (teacherRejection) return teacherRejection;
  const acceptance = metadata.acceptance;
  if (acceptance?.accepted !== false) return null;
  const criterion = acceptance.criterion ?? "unspecified acceptance check";
  const accuracy = Number.isFinite(metadata.verification?.accuracy)
    ? `; verification accuracy ${percent(metadata.verification.accuracy)}` : "";
  const macroF1 = Number.isFinite(metadata.verification?.macroF1)
    ? `; styled macro F1 ${percent(metadata.verification.macroF1)}` : "";
  return `run was rejected by ${criterion}${accuracy}${macroF1}`;
}

function one(entries, label) {
  if (entries.length !== 1) throw new Error(`expected exactly one ${label}, found ${entries.length}`);
  return entries[0];
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (import.meta.url === invokedPath) {
  try {
    const arguments_ = process.argv.slice(2);
    await promote(arguments_.find((argument) => argument !== "--" && !argument.startsWith("--")),
      { force: arguments_.includes("--force") });
  } catch (error) {
    console.error(error instanceof Error ? error.stack : error);
    process.exitCode = 1;
  }
}

import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { createLanguageObjective, validateLanguageObjective, languageMetrics } from "./language-objective.js";
import { resolveRunPath } from "./checkpoint.js";
import { evaluateTree, loadTreeShard } from "./tree-model.js";
import { dequantizeTensors } from "./quantization.js";
import { validateTreeComparisonProvenance } from "./promote.js";

// The complete policy is saved with each run; it never follows shard frequencies.
export async function treeTrainingPolicy(options = {}) {
  const languageObjective = options.languageObjectivePath
    ? validateLanguageObjective(JSON.parse(await readFile(options.languageObjectivePath, "utf8")))
    : createLanguageObjective();
  const classWeightPower = options.classWeightPower ?? 0.5;
  const calibrationEpochs = options.calibrationEpochs ?? 2;
  const agreementEpochs = options.agreementEpochs ?? 0;
  const selectionMetric = options.selectionMetric ?? "accuracy";
  const epochs = options.epochs ?? 32, fineTuneEpochs = options.fineTuneEpochs ?? 4;
  if (!Number.isFinite(classWeightPower) || classWeightPower < 0 || classWeightPower > 1) {
    throw new Error("class-weight-power must be between 0 and 1 (0 = natural, 0.25 = gentle, 0.5 = original)");
  }
  if (!Number.isSafeInteger(epochs) || epochs < 1 || !Number.isSafeInteger(fineTuneEpochs) ||
      fineTuneEpochs < 0 || fineTuneEpochs > epochs || !Number.isSafeInteger(calibrationEpochs) ||
      calibrationEpochs < 0 || calibrationEpochs > fineTuneEpochs ||
      !Number.isSafeInteger(agreementEpochs) || agreementEpochs < 0 || fineTuneEpochs + agreementEpochs > epochs) {
    throw new Error("require 0 <= calibration-epochs <= fine-tune-epochs and fine-tune-epochs + agreement-epochs <= epochs");
  }
  if (!["weightedError", "accuracy"].includes(selectionMetric)) {
    throw new Error("selection metric must be weightedError or accuracy");
  }
  return { languageObjective, classWeightPower, agreementEpochs, calibrationEpochs, selectionMetric,
    fixedBaseline: options.baselineRun ? await pinBaseline(options.baselineRun) : null };
}

export async function pinBaseline(argument) {
  const run = resolveRunPath(argument);
  const entries = await readdir(run);
  const one = (pattern) => {
    const files = entries.filter((name) => pattern.test(name));
    if (files.length !== 1) throw new Error(`expected one baseline artifact matching ${pattern}`);
    return resolve(run, files[0]);
  };
  const metadata = JSON.parse(await readFile(one(/^model-.*\.json$/), "utf8"));
  const bytes = await readFile(one(/^weights-int[456]-.*\.bin$/));
  if (metadata.model !== "hierarchical-tree" || metadata.labelSource !== "shiki-spans-v1" || !metadata.runId) {
    throw new Error("baseline must be a direct-label tree checkpoint; audit old weights separately first");
  }
  return { run, runId: metadata.runId, weightsSha256: createHash("sha256").update(bytes).digest("hex") };
}

export async function fixedBaselineMetrics(config, candidateMetadata, records) {
  if (!config.fixedBaseline) return null;
  const { run, runId, weightsSha256 } = config.fixedBaseline;
  const files = await readdir(run);
  const metadata = JSON.parse(await readFile(resolve(run, files.find((name) => /^model-.*\.json$/.test(name))), "utf8"));
  const bytes = await readFile(resolve(run, files.find((name) => /^weights-int[456]-.*\.bin$/.test(name))));
  if (metadata.runId !== runId || createHash("sha256").update(bytes).digest("hex") !== weightsSha256) {
    throw new Error("fixed baseline changed after pinning");
  }
  validateTreeComparisonProvenance(candidateMetadata, metadata, candidateMetadata.corpus.verification.sha256);
  const sameFeatures = config.featureVersion === metadata.featureVersion &&
    config.hashBuckets === metadata.architecture.lexemeHashBuckets;
  const input = sameFeatures ? records
    : (await loadTreeShard(config.verificationShard, config.maxVerificationTokens ?? Infinity,
      metadata.architecture.lexemeHashBuckets, { requireSourceLabels: true,
        excludeFamilies: config.excludedFamilies, featureVersion: metadata.featureVersion })).records;
  return evaluateTree(dequantizeTensors(bytes, metadata.quantization), input, {
    hiddenSize: metadata.hiddenSize, classifierSize: metadata.architecture.classifierDimensions,
    languageObjective: config.languageObjective,
  });
}

export function assertObjectiveCoverage(records, objective, stage) {
  const counts = {};
  for (const record of records) {
    const count = counts[record.family ?? record.language ?? "unknown"] ??= { support: 0, errors: 0, plainSupport: 0, falseColors: 0 };
    for (let i = 0; i < record.targets.length; i++) {
      if (record.supervisionWeights?.[i] === 0) continue;
      count.support++;
      count.plainSupport += Number(record.targets[i] === 0);
    }
  }
  const metrics = languageMetrics(counts, objective);
  if (!metrics.objectiveComplete) throw new Error(`${stage} language coverage incomplete; missing: ${metrics.missingFamilies.join(", ")}; unknown: ${metrics.unknownFamilies.join(", ")}. Do not renormalize a partial shard.`);
  const warnings = [];
  if (stage === "verification") {
    for (const family of objective.protectedFamilies) {
      const count = metrics.perFamily[family];
      if (count.support < objective.minSupport) warnings.push(`${family} support ${count.support} < ${objective.minSupport}`);
      if (count.plainSupport < objective.minPlainSupport) warnings.push(`${family} plain support ${count.plainSupport} < ${objective.minPlainSupport}`);
    }
  }
  return warnings;
}

export const capacityExperiments = Object.freeze([
  { name: "baseline", hiddenSize: 32, classifierSize: 72, hashBuckets: 128 },
  { name: "classifier-only", hiddenSize: 32, classifierSize: 96, hashBuckets: 128 },
  { name: "hash-only", hiddenSize: 32, classifierSize: 72, hashBuckets: 256 },
  { name: "combined", hiddenSize: 32, classifierSize: 96, hashBuckets: 256 },
]);

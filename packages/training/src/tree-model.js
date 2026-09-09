import { createReadStream } from "node:fs";
import readline from "node:readline";
import { createGunzip } from "node:zlib";

import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { TREE_TOKENIZER_VERSION } from "../../core/src/constants.js";
import {
  TREE_FEATURE_STRIDE, TREE_FEATURE_VERSION, TREE_HASH_BUCKETS,
  TREE_LEGACY_FEATURE_VERSION, TREE_LEGACY_SCALE_BUCKETS, TREE_SCALE_BUCKETS,
  TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, treeFeatureIndices, treeFeatureLayout, treeInputSize,
} from "../../core/src/tree-features.js";
import { auxiliaryNames, classNames } from "./classes.js";
import { languageFamily } from "./corpus.js";
import { alignTreeLabels, corpusSourceLabels } from "./tree-label-alignment.js";
import { languageMetrics } from "./language-objective.js";

export const TREE_MODEL = "hierarchical-tree";
export const TREE_FORMAT_VERSION = 8;
export const TREE_HYBRID_FORMAT_VERSION = 9;
export { TREE_TOKENIZER_VERSION };
export const treeAuxiliaryNames = Object.freeze([
  ...auxiliaryNames,
  "curly-depth-1", "curly-depth-2+",
  "paren-depth-1", "paren-depth-2+",
  "square-depth-1", "square-depth-2+",
]);
export const treeTensorNames = Object.freeze([
  "featureEmbedding", "neighborScale", "leafBias",
  "mergeOwnLeft", "mergeOwnRight", "mergeCrossLeft", "mergeCrossRight", "mergeBias",
  "downOwnParent", "downOwnSelf", "downOwnSibling",
  "downCrossParent", "downCrossSelf", "downCrossSibling", "downSkip",
  "downLeftBias", "downRightBias",
  "classifierInput", "classifierBias", "output", "outputBias",
  "auxiliaryOutput", "auxiliaryBias",
]);

/** Decode the frozen feature-v2 shape only while migrating old checkpoints. */
export function prepareTreeSourceV2(source) {
  const prepared = prepareTreeSource(source);
  const data = prepared[0];
  for (let index = 1; index < data.length; index += TREE_FEATURE_STRIDE) {
    data[index] &= 0x7fff;
  }
  return prepared;
}

export const treeHybridTensorNames = Object.freeze([
  "localOffsetScale", "localNonspaceScale", "stateInput", "stateInputBias",
  "stateGate", "stateGateBias", "stateMix", "stateMixBias",
]);

export function treeTensorNamesFor(modelOrConfig) {
  return modelOrConfig.stateInput || modelOrConfig.treeContext === "hybrid"
    ? [...treeTensorNames, ...treeHybridTensorNames] : treeTensorNames;
}

export function createTreeModel({
  treeContext = "tree",
  hiddenSize = 64,
  classifierSize = 72,
  hashBuckets = TREE_HASH_BUCKETS,
  featureVersion = TREE_FEATURE_VERSION,
  scaleBuckets = TREE_SCALE_BUCKETS,
  random = Math.random,
} = {}) {
  if (!["tree", "hybrid"].includes(treeContext)) throw new RangeError("tree context must be tree or hybrid");
  if (!Number.isInteger(hiddenSize) || hiddenSize < 2 || (hiddenSize & (hiddenSize - 1))) {
    throw new RangeError("tree hidden size must be a power of two of at least 2");
  }
  const inputSize = treeInputSize(hashBuckets, featureVersion);
  const auxiliary = treeAuxiliaryNames.length;
  const output = classNames.length;
  const model = {
    featureEmbedding: new Float32Array(inputSize * hiddenSize),
    neighborScale: new Float32Array(3 * hiddenSize),
    leafBias: new Float32Array(hiddenSize),
    mergeOwnLeft: new Float32Array(scaleBuckets * hiddenSize),
    mergeOwnRight: new Float32Array(scaleBuckets * hiddenSize),
    mergeCrossLeft: new Float32Array(scaleBuckets * hiddenSize),
    mergeCrossRight: new Float32Array(scaleBuckets * hiddenSize),
    mergeBias: new Float32Array(scaleBuckets * hiddenSize),
    downOwnParent: new Float32Array(scaleBuckets * hiddenSize),
    downOwnSelf: new Float32Array(scaleBuckets * hiddenSize),
    downOwnSibling: new Float32Array(scaleBuckets * hiddenSize),
    downCrossParent: new Float32Array(scaleBuckets * hiddenSize),
    downCrossSelf: new Float32Array(scaleBuckets * hiddenSize),
    downCrossSibling: new Float32Array(scaleBuckets * hiddenSize),
    downSkip: new Float32Array(scaleBuckets * hiddenSize),
    downLeftBias: new Float32Array(scaleBuckets * hiddenSize),
    downRightBias: new Float32Array(scaleBuckets * hiddenSize),
    classifierInput: new Float32Array(classifierSize * (hiddenSize * 2 + auxiliary)),
    classifierBias: new Float32Array(classifierSize),
    output: new Float32Array(output * classifierSize),
    outputBias: new Float32Array(output),
    auxiliaryOutput: new Float32Array(auxiliary * hiddenSize * 2),
    auxiliaryBias: new Float32Array(auxiliary),
  };
  initialize(model.featureEmbedding, Math.sqrt(6 / (inputSize + hiddenSize)), random);
  initialize(model.classifierInput, Math.sqrt(6 / (hiddenSize * 2 + auxiliary + classifierSize)), random);
  initialize(model.output, Math.sqrt(6 / (classifierSize + output)), random);
  initialize(model.auxiliaryOutput, Math.sqrt(6 / (hiddenSize * 2 + auxiliary)), random);
  model.neighborScale.fill(1);
  model.mergeOwnLeft.fill(0.5);
  model.mergeOwnRight.fill(0.5);
  model.mergeCrossLeft.fill(0.05);
  model.mergeCrossRight.fill(0.05);
  model.downOwnParent.fill(0.5);
  model.downOwnSelf.fill(0.5);
  model.downOwnSibling.fill(0.25);
  model.downCrossParent.fill(0.05);
  model.downCrossSelf.fill(0.05);
  model.downCrossSibling.fill(0.05);
  if (treeContext === "hybrid") {
    Object.assign(model, {
      localOffsetScale: new Float32Array(5 * hiddenSize).fill(0.2),
      localNonspaceScale: new Float32Array(2 * hiddenSize).fill(0.25),
      stateInput: new Float32Array(hiddenSize * hiddenSize),
      stateInputBias: new Float32Array(hiddenSize),
      stateGate: new Float32Array(hiddenSize * hiddenSize),
      stateGateBias: new Float32Array(hiddenSize).fill(2),
      stateMix: new Float32Array(hiddenSize * hiddenSize * 2),
      stateMixBias: new Float32Array(hiddenSize),
    });
    initialize(model.stateInput, Math.sqrt(3 / hiddenSize), random);
    initialize(model.stateGate, Math.sqrt(3 / hiddenSize), random);
    initialize(model.stateMix, Math.sqrt(2 / hiddenSize), random);
  }
  return model;
}

/** Expand a feature-v2 model into feature v3 without changing its function. */
export function migrateTreeFeatureV2Model(source, {
  treeContext = source.stateInput ? "hybrid" : "tree",
  hiddenSize = source.leafBias?.length,
  classifierSize = source.classifierBias?.length,
} = {}) {
  const sourceShape = createTreeModel({
    treeContext, hiddenSize, classifierSize, hashBuckets: 128,
    featureVersion: TREE_LEGACY_FEATURE_VERSION, scaleBuckets: TREE_LEGACY_SCALE_BUCKETS,
    random: () => 0.5,
  });
  const target = createTreeModel({
    treeContext, hiddenSize, classifierSize, hashBuckets: TREE_HASH_BUCKETS,
    featureVersion: TREE_FEATURE_VERSION, scaleBuckets: TREE_SCALE_BUCKETS,
    random: () => 0.5,
  });
  const names = treeTensorNamesFor(target);
  if (JSON.stringify(Object.keys(source)) !== JSON.stringify(treeTensorNamesFor(sourceShape)) ||
      JSON.stringify(names) !== JSON.stringify(treeTensorNamesFor(source))) {
    throw new Error("feature-v2 migration requires the exact tree tensor schema");
  }
  for (const name of names) {
    if (source[name]?.length !== sourceShape[name].length) {
      throw new Error(`feature-v2 migration tensor ${name} has the wrong shape`);
    }
    if (name === "featureEmbedding") continue;
    if (name.startsWith("merge") || name.startsWith("down")) {
      for (let scale = 0; scale < TREE_SCALE_BUCKETS; scale++) {
        const from = Math.min(scale, TREE_LEGACY_SCALE_BUCKETS - 1) * hiddenSize;
        target[name].set(source[name].subarray(from, from + hiddenSize), scale * hiddenSize);
      }
    } else {
      if (source[name].length !== target[name].length) {
        throw new Error(`feature-v2 migration tensor ${name} cannot preserve its shape`);
      }
      target[name].set(source[name]);
    }
  }

  const before = treeFeatureLayout(128, TREE_LEGACY_FEATURE_VERSION);
  const after = treeFeatureLayout(TREE_HASH_BUCKETS, TREE_FEATURE_VERSION);
  const copyRows = (from, to, count) => target.featureEmbedding.set(
    source.featureEmbedding.subarray(from * hiddenSize, (from + count) * hiddenSize),
    to * hiddenSize,
  );
  copyRows(0, 0, 268);
  for (let bucket = 0; bucket < TREE_HASH_BUCKETS; bucket++) {
    copyRows(before.primaryHash + (bucket & 127), after.primaryHash + bucket, 1);
  }
  copyRows(before.flags, after.flags, 7);
  copyRows(before.previousPair, after.previousPair, 16);
  copyRows(before.nextPair, after.nextPair, 16);
  // createTreeModel's deterministic midpoint initializer leaves the new
  // secondary and generic symbol-pair rows exactly zero.
  return target;
}

export function treeTensorLayout(model, { inputSize, hiddenSize, classifierSize }) {
  const auxiliary = treeAuxiliaryNames.length;
  const scaleBuckets = model.mergeOwnLeft.length / hiddenSize;
  const shapes = {
    featureEmbedding: [inputSize, hiddenSize], neighborScale: [3, hiddenSize], leafBias: [hiddenSize],
    mergeOwnLeft: [scaleBuckets, hiddenSize], mergeOwnRight: [scaleBuckets, hiddenSize],
    mergeCrossLeft: [scaleBuckets, hiddenSize], mergeCrossRight: [scaleBuckets, hiddenSize],
    mergeBias: [scaleBuckets, hiddenSize],
    downOwnParent: [scaleBuckets, hiddenSize], downOwnSelf: [scaleBuckets, hiddenSize],
    downOwnSibling: [scaleBuckets, hiddenSize], downCrossParent: [scaleBuckets, hiddenSize],
    downCrossSelf: [scaleBuckets, hiddenSize], downCrossSibling: [scaleBuckets, hiddenSize],
    downSkip: [scaleBuckets, hiddenSize],
    downLeftBias: [scaleBuckets, hiddenSize], downRightBias: [scaleBuckets, hiddenSize],
    classifierInput: [classifierSize, hiddenSize * 2 + auxiliary], classifierBias: [classifierSize],
    output: [classNames.length, classifierSize], outputBias: [classNames.length],
    auxiliaryOutput: [auxiliary, hiddenSize * 2], auxiliaryBias: [auxiliary],
    localOffsetScale: [5, hiddenSize], localNonspaceScale: [2, hiddenSize],
    stateInput: [hiddenSize, hiddenSize], stateInputBias: [hiddenSize],
    stateGate: [hiddenSize, hiddenSize], stateGateBias: [hiddenSize],
    stateMix: [hiddenSize, hiddenSize * 2], stateMixBias: [hiddenSize],
  };
  let offset = 0;
  return treeTensorNamesFor(model).map((name) => {
    const tensor = { name, shape: shapes[name], offset, length: model[name].length };
    offset += tensor.length;
    return tensor;
  });
}

export async function loadTreeShard(path, maxTokens = Infinity, hashBuckets = TREE_HASH_BUCKETS, {
  requireSourceLabels = true,
  retainSource = false, maxFiles = Infinity, excludeFamilies = null, includeFamilies = null,
  excludeSources = null, featureVersion = TREE_FEATURE_VERSION,
} = {}) {
  if (!requireSourceLabels) throw new Error("tree shards always require direct source labels");
  const excludedFamilies = new Set(excludeFamilies ?? []);
  const includedFamilies = includeFamilies ? new Set(includeFamilies) : null;
  const records = [];
  const classCounts = new Uint32Array(classNames.length);
  let tokenCount = 0;
  let fileCount = 0;
  const sourceNames = new Set();
  const lines = readline.createInterface({
    input: createReadStream(path).pipe(createGunzip()), crlfDelay: Infinity,
  });
  try {
    for await (const line of lines) {
      if (!line) continue;
      if (tokenCount >= maxTokens || fileCount >= maxFiles) break;
      const item = JSON.parse(line);
      if (excludeSources?.has(`${item.origin}\0${item.sourceName}\0${item.path}`)) continue;
      const family = item.family ?? languageFamily(item.language ?? "unknown");
      if (excludedFamilies.has(family) || includedFamilies && !includedFamilies.has(family)) continue;
      const { record, classCounts: itemCounts } = createTreeRecord(item, hashBuckets, {
        maxTokens: maxTokens - tokenCount, retainSource, featureVersion,
      });
      records.push(record);
      for (let index = 0; index < classCounts.length; index++) classCounts[index] += itemCounts[index];
      tokenCount += record.targets.length;
      fileCount += 1;
      sourceNames.add(item.sourceName);
    }
  } catch (error) {
    if (error?.code === "ENOENT") throw new Error(`missing corpus shard ${path}; run corpus:prepare first`, { cause: error });
    throw error;
  }
  return { records, tokenCount, fileCount, classCounts, sourceNames: [...sourceNames].sort() };
}

/** Build the exact sparse tree record used by training from raw source labels.
 * This is also the single-snippet entry point used by constrained fine-tuning.
 */
export function createTreeRecord(item, hashBuckets = TREE_HASH_BUCKETS, {
  maxTokens = Infinity, retainSource = false, featureVersion = TREE_FEATURE_VERSION,
} = {}) {
  const provenance = corpusSourceLabels(item);
  const prepared = featureVersion === TREE_LEGACY_FEATURE_VERSION
    ? prepareTreeSourceV2(item.source)
    : prepareTreeSource(item.source);
  try {
    const [data, preparedRanges, , tokenCount] = prepared;
    const take = Math.min(tokenCount, maxTokens);
    const features = new Array(take);
    const targets = new Uint8Array(take);
    const auxiliary = new Uint16Array(take);
    const supervisionWeights = new Uint8Array(take).fill(255);
    const lossWeights = new Uint8Array(take).fill(1);
    const classCounts = new Uint32Array(classNames.length);
    const ranges = preparedRanges.slice(0, take * 2);
    const directLabels = alignTreeLabels(provenance.sourceLabels, ranges, { sourceLength: item.source.length });
    const depths = { curly: 0, paren: 0, square: 0 };
    for (let index = 0; index < take; index++) {
      const from = preparedRanges[index * 2];
      const to = preparedRanges[index * 2 + 1];
      features[index] = Uint16Array.from(treeFeatureIndices(data, index, hashBuckets, featureVersion));
      const kind = data[index * TREE_FEATURE_STRIDE] & 3;
      auxiliary[index] = nestingBits(depths);
      if (kind === TREE_PART_SPACE || kind === TREE_PART_NEWLINE) {
        supervisionWeights[index] = 0;
        continue;
      }
      const label = directLabels[index];
      targets[index] = classNames.indexOf(label.class);
      auxiliary[index] |= label.auxiliary;
      supervisionWeights[index] = label.confidence;
      if (supervisionWeights[index] > 0) classCounts[targets[index]] += 1;
      updateDepths(depths, item.source, from, to, kind);
    }
    const language = item.language ?? "unknown";
    return { classCounts, record: {
      features, targets, auxiliary, supervisionWeights, lossWeights,
      labelSource: provenance.labelSource,
      ...(retainSource ? { source: item.source, sourceLabels: provenance.sourceLabels, ranges } : {}),
      language, family: item.family ?? languageFamily(language),
      sourceName: item.sourceName, path: item.path, origin: item.origin, stratum: item.stratum,
      replayWeight: item.replayWeight ?? 1,
      mixedLanguage: ["html", "markdown", "vue", "svelte", "astro", "diff"].includes(language),
      embeddedLanguage: ["html", "markdown", "vue", "svelte", "astro"].includes(language),
    } };
  } finally {
    releaseTreePrepared(prepared);
  }
}

export function treeProbabilities(model, record, { hiddenSize, classifierSize }) {
  const count = record.features.length;
  if (!count) return [];
  const raw = Array.from({ length: count }, () => new Float32Array(hiddenSize));
  for (let index = 0; index < count; index++) for (const feature of record.features[index]) {
    const row = feature * hiddenSize;
    for (let h = 0; h < hiddenSize; h++) raw[index][h] += model.featureEmbedding[row + h];
  }
  const leaf = model.stateInput ? hybridLeaves(model, raw, record.features, hiddenSize) : raw.map((current, index) => Float32Array.from(current, (_, h) => Math.tanh(
    (raw[index - 1]?.[h] ?? 0) * model.neighborScale[h] + current[h] * model.neighborScale[hiddenSize + h] +
    (raw[index + 1]?.[h] ?? 0) * model.neighborScale[hiddenSize * 2 + h] + model.leafBias[h],
  )));
  let width = 1;
  while (width < count) width *= 2;
  let level = Array.from({ length: width }, (_, index) => leaf[index] ?? null);
  const levels = [level];
  let depth = 0;
  while (level.length > 1) {
    const next = new Array(level.length / 2);
    for (let index = 0; index < next.length; index++) {
      const left = level[index * 2];
      const right = level[index * 2 + 1];
      if (!left) next[index] = null;
      else if (!right) next[index] = left;
      else {
        const scaleBuckets = model.mergeOwnLeft.length / hiddenSize;
        const scale = Math.min(depth, scaleBuckets - 1) * hiddenSize;
        const shift = 1 << Math.min(depth, Math.max(0, Math.floor(Math.log2(hiddenSize)) - 1));
        next[index] = Float32Array.from(left, (value, h) => {
          const partner = h ^ shift;
          const mixed = Math.tanh(
            value * model.mergeOwnLeft[scale + h] + right[h] * model.mergeOwnRight[scale + h] +
            left[partner] * model.mergeCrossLeft[scale + h] +
            right[partner] * model.mergeCrossRight[scale + h] + model.mergeBias[scale + h],
          );
          const boundary = h < hiddenSize / 2 ? value : right[h];
          return (mixed + boundary) * 0.5;
        });
      }
    }
    levels.push(next);
    level = next;
    depth += 1;
  }
  let down = [level[0]];
  for (let depth = levels.length - 2; depth >= 0; depth--) {
    const children = levels[depth];
    const next = new Array(children.length);
    for (let parent = 0; parent < down.length; parent++) {
      const left = children[parent * 2];
      const right = children[parent * 2 + 1];
      if (!left) continue;
      if (!right) { next[parent * 2] = down[parent]; continue; }
      next[parent * 2] = descend(model, down[parent], left, right, depth, false);
      next[parent * 2 + 1] = descend(model, down[parent], right, left, depth, true);
    }
    down = next;
  }
  return leaf.map((local, index) => {
    if (record.features[index][0] === TREE_PART_SPACE || record.features[index][0] === TREE_PART_NEWLINE) {
      return Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    return classify(model, local, down[index], hiddenSize, classifierSize);
  });
}

export function hybridLeaves(model, raw, features, hidden) {
  const count = raw.length;
  const previous = new Int32Array(count).fill(-1);
  const next = new Int32Array(count).fill(-1);
  let last = -1;
  for (let i = 0; i < count; i++) {
    previous[i] = last;
    if (features[i][0] !== TREE_PART_SPACE && features[i][0] !== TREE_PART_NEWLINE) last = i;
  }
  last = -1;
  for (let i = count - 1; i >= 0; i--) {
    next[i] = last;
    if (features[i][0] !== TREE_PART_SPACE && features[i][0] !== TREE_PART_NEWLINE) last = i;
  }
  const local = raw.map((_, i) => Float32Array.from({ length: hidden }, (_, h) => {
    let value = model.leafBias[h];
    for (let offset = -2; offset <= 2; offset++) {
      value += (raw[i + offset]?.[h] ?? 0) * model.localOffsetScale[(offset + 2) * hidden + h];
    }
    value += (raw[previous[i]]?.[h] ?? 0) * model.localNonspaceScale[h];
    value += (raw[next[i]]?.[h] ?? 0) * model.localNonspaceScale[hidden + h];
    return Math.tanh(value);
  }));
  function linear(input, weights, bias, row) {
    let sum = bias[row];
    for (let h = 0; h < input.length; h++) sum += input[h] * weights[row * input.length + h];
    return sum;
  }
  const gates = local.map((input) => Float32Array.from({ length: hidden }, (_, h) =>
    1 / (1 + Math.exp(-linear(input, model.stateGate, model.stateGateBias, h)))));
  const values = local.map((input) => Float32Array.from({ length: hidden }, (_, h) =>
    Math.tanh(linear(input, model.stateInput, model.stateInputBias, h))));
  function scan(reverse) {
    const result = new Array(count);
    const state = new Float32Array(hidden);
    for (let step = 0; step < count; step++) {
      const i = reverse ? count - 1 - step : step;
      for (let h = 0; h < hidden; h++) state[h] = gates[i][h] * state[h] + (1 - gates[i][h]) * values[i][h];
      result[i] = state.slice();
    }
    return result;
  }
  const forward = scan(false), reverse = scan(true);
  return local.map((input, i) => {
    const context = [...forward[i], ...reverse[i]];
    return Float32Array.from(input, (value, h) => Math.tanh(value + linear(context, model.stateMix, model.stateMixBias, h)));
  });
}

export function evaluateTree(model, records, config) {
  const confusion = Array.from({ length: classNames.length }, () => new Uint32Array(classNames.length));
  let correct = 0;
  let total = 0;
  const mixed = { correct: 0, total: 0 };
  const families = Object.create(null);
  const languages = Object.create(null);
  for (const record of records) {
    const family = record.family ?? languageFamily(record.language ?? "unknown");
    const count = families[family] ??= { support: 0, errors: 0, plainSupport: 0, falseColors: 0 };
    const language = record.language ?? "unknown";
    const languageCount = languages[language] ??= { support: 0, errors: 0, plainSupport: 0, falseColors: 0 };
    const probabilities = treeProbabilities(model, record, config);
    for (let index = 0; index < probabilities.length; index++) {
      const kind = record.features[index][0];
      if (kind === TREE_PART_SPACE || kind === TREE_PART_NEWLINE || record.supervisionWeights?.[index] === 0) continue;
      let predicted = 0;
      for (let value = 1; value < probabilities[index].length; value++) {
        if (probabilities[index][value] > probabilities[index][predicted]) predicted = value;
      }
      const expected = record.targets[index];
      confusion[expected][predicted] += 1;
      count.support += 1;
      count.errors += Number(expected !== predicted);
      count.plainSupport += Number(expected === 0);
      count.falseColors += Number(expected === 0 && predicted !== 0);
      languageCount.support += 1;
      languageCount.errors += Number(expected !== predicted);
      languageCount.plainSupport += Number(expected === 0);
      languageCount.falseColors += Number(expected === 0 && predicted !== 0);
      correct += Number(expected === predicted);
      total += 1;
      if (record.mixedLanguage) { mixed.correct += Number(expected === predicted); mixed.total += 1; }
    }
  }
  let macroF1 = 0;
  for (let target = 1; target < classNames.length; target++) {
    const tp = confusion[target][target];
    let fp = 0, fn = 0;
    for (let other = 0; other < classNames.length; other++) if (other !== target) {
      fp += confusion[other][target]; fn += confusion[target][other];
    }
    macroF1 += 2 * tp / Math.max(1, 2 * tp + fp + fn);
  }
  const perLanguage = Object.fromEntries(Object.entries(languages).map(([language, count]) => [language, {
    ...count, accuracy: count.support ? 1 - count.errors / count.support : null,
    error: count.support ? count.errors / count.support : null,
    falseColorRate: count.plainSupport ? count.falseColors / count.plainSupport : null,
  }]));
  return {
    accuracy: correct / Math.max(1, total), macroF1: macroF1 / (classNames.length - 1),
    falseColorRate: (confusion[0].reduce((sum, n) => sum + n, 0) - confusion[0][0]) /
      Math.max(1, confusion[0].reduce((sum, n) => sum + n, 0)),
    confusion: confusion.map((row) => [...row]),
    perClass: classNames.map((name, index) => {
      const support = confusion[index].reduce((sum, n) => sum + n, 0);
      const predicted = confusion.reduce((sum, row) => sum + row[index], 0);
      const tp = confusion[index][index];
      return { class: name, support, precision: tp / Math.max(1, predicted), recall: tp / Math.max(1, support),
        f1: 2 * tp / Math.max(1, support + predicted) };
    }),
    mixedLanguage: { accuracy: mixed.correct / Math.max(1, mixed.total), macroF1: 0 },
    perLanguage,
    ...languageMetrics(families, config.languageObjective),
    ...(config.evaluationProvenance ? { provenance: config.evaluationProvenance } : {}),
  };
}

function descend(model, parent, self, sibling, depth, right) {
  const hidden = parent.length;
  const scaleBuckets = model.downOwnParent.length / hidden;
  const scale = Math.min(depth, scaleBuckets - 1) * hidden;
  const shift = 1 << Math.min(depth, Math.max(0, Math.floor(Math.log2(hidden)) - 1));
  const bias = right ? model.downRightBias : model.downLeftBias;
  return Float32Array.from(parent, (value, h) => {
    const index = scale + h;
    const partner = h ^ shift;
    const mixed = Math.tanh(
      value * model.downOwnParent[index] + self[h] * model.downOwnSelf[index] +
      sibling[h] * model.downOwnSibling[index] + parent[partner] * model.downCrossParent[index] +
      self[partner] * model.downCrossSelf[index] + sibling[partner] * model.downCrossSibling[index] + bias[index],
    );
    const skip = 1 / (1 + Math.exp(-model.downSkip[index]));
    return value * skip + mixed * (1 - skip);
  });
}

function classify(model, leaf, context, hidden, classifierSize) {
  const joined = [...leaf, ...context];
  const auxiliary = new Float32Array(treeAuxiliaryNames.length);
  for (let a = 0; a < auxiliary.length; a++) {
    let sum = model.auxiliaryBias[a];
    for (let index = 0; index < joined.length; index++) sum += model.auxiliaryOutput[a * joined.length + index] * joined[index];
    auxiliary[a] = 1 / (1 + Math.exp(-sum));
  }
  joined.push(...auxiliary);
  const classifier = new Float32Array(classifierSize);
  for (let c = 0; c < classifierSize; c++) {
    let sum = model.classifierBias[c];
    for (let index = 0; index < joined.length; index++) sum += model.classifierInput[c * joined.length + index] * joined[index];
    classifier[c] = Math.tanh(sum);
  }
  const logits = new Float32Array(classNames.length);
  let maximum = -Infinity;
  for (let output = 0; output < logits.length; output++) {
    let sum = model.outputBias[output];
    for (let c = 0; c < classifierSize; c++) sum += model.output[output * classifierSize + c] * classifier[c];
    logits[output] = sum; maximum = Math.max(maximum, sum);
  }
  let denominator = 0;
  for (let output = 0; output < logits.length; output++) denominator += logits[output] = Math.exp(logits[output] - maximum);
  return Float32Array.from(logits, (value) => value / denominator);
}

function initialize(values, scale, random) {
  for (let index = 0; index < values.length; index++) values[index] = (random() * 2 - 1) * scale;
}

function nestingBits(depths) {
  let bits = 0;
  if (depths.curly > 0) bits |= 1 << 10;
  if (depths.curly > 1) bits |= 1 << 11;
  if (depths.paren > 0) bits |= 1 << 12;
  if (depths.paren > 1) bits |= 1 << 13;
  if (depths.square > 0) bits |= 1 << 14;
  if (depths.square > 1) bits |= 1 << 15;
  return bits;
}

function updateDepths(depths, source, from, to, kind) {
  if (kind !== TREE_PART_SYMBOL || to !== from + 1) return;
  const symbol = source.charCodeAt(from);
  if (symbol === 123) depths.curly += 1;
  else if (symbol === 125) depths.curly = Math.max(0, depths.curly - 1);
  else if (symbol === 40) depths.paren += 1;
  else if (symbol === 41) depths.paren = Math.max(0, depths.paren - 1);
  else if (symbol === 91) depths.square += 1;
  else if (symbol === 93) depths.square = Math.max(0, depths.square - 1);
}

export { TREE_FEATURE_VERSION, TREE_HASH_BUCKETS };

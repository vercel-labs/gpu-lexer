import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { classNames, auxiliaryNames } from "./classes.js";
import { loadFloatCheckpoint } from "./checkpoint.js";
import { dequantizeTensors } from "./quantization.js";
import { auditSourceLabels } from "./source-label-audit.js";
import { alignTreeLabels, clipSourceLabels } from "./tree-label-alignment.js";
import {
  TREE_FEATURE_VERSION, TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION,
  TREE_MODEL, TREE_TOKENIZER_VERSION, loadTreeShard, treeProbabilities,
  treeTensorLayout, treeTensorNamesFor,
} from "./tree-model.js";
import {
  TREE_PART_NEWLINE, TREE_PART_SPACE, isTreeFeatureVersion, treeInputSize,
} from "../../core/src/tree-features.js";

const defaultShard = fileURLToPath(new URL("../data/generated/shards/verification.jsonl.gz", import.meta.url));
export const defaults = Object.freeze({ maxParts: 10_000, maxFiles: 64 });

/** Read-only, deterministic prefix diagnostic. Never trains, exports weights, or
 * infers a deployed artifact by re-quantizing a float checkpoint. */
export async function evaluateTree(options = {}) {
  if (!options.run) throw new Error("a checkpoint is required (--run <run-id-or-path>)");
  const maxParts = positiveInteger(options.maxParts ?? defaults.maxParts, "max-parts");
  const maxFiles = positiveInteger(options.maxFiles ?? defaults.maxFiles, "max-files");
  const checkpoint = await loadFloatCheckpoint(options.run);
  const config = validateCheckpoint(checkpoint);
  const packed = config.teacher ? null : await loadPackedModel(checkpoint);
  const shard = resolve(options.shard ?? defaultShard);
  const data = await loadTreeShard(shard, maxParts, config.hashBuckets, {
    retainSource: true, maxFiles,
    featureVersion: config.featureVersion,
  });
  // Do not silently use a loader which ignored the file bound/diagnostic options.
  if (data.records.length > maxFiles || data.tokenCount > maxParts) {
    throw new Error("loadTreeShard did not respect the diagnostic bounds");
  }
  const float = accumulator(), quantized = packed ? accumulator() : null;
  const changes = { comparedParts: 0, disagreements: 0, floatOnlyCorrect: 0, quantizedOnlyCorrect: 0 };
  const files = [];
  for (const record of data.records) {
    requireSource(record);
    const count = record.features.length;
    const end = count ? record.ranges[count * 2 - 1] : 0;
    const source = record.source.slice(0, end);
    const expected = clipSourceLabels(record.sourceLabels, end);
    const aligned = alignTreeLabels(expected, record.ranges, { sourceLength: end });
    const floating = predict(checkpoint.model, record, config);
    const deployed = packed ? predict(packed.model, record, config) : null;
    scoreRecord(float, record, source, expected, aligned, floating);
    if (deployed) scoreRecord(quantized, record, source, expected, aligned, deployed);
    for (let index = 0; deployed && index < count; index++) {
      if (!scoredPart(record, aligned, index)) continue;
      const target = classNames.indexOf(aligned[index].class);
      changes.comparedParts++;
      changes.disagreements += Number(floating[index] !== deployed[index]);
      changes.floatOnlyCorrect += Number(floating[index] === target && deployed[index] !== target);
      changes.quantizedOnlyCorrect += Number(deployed[index] === target && floating[index] !== target);
    }
    files.push({
      path: record.path ?? null, sourceName: record.sourceName ?? null,
      language: record.language ?? "unknown", parts: count,
      evaluatedSourceEnd: end, sourceLength: record.source.length,
      truncated: end < record.source.length,
      labelSource: record.labelSource,
    });
  }
  const floatMetrics = summarize(float);
  const quantizedMetrics = quantized ? summarize(quantized) : null;
  return {
    diagnostic: "bounded-tree-checkpoint", readOnly: true,
    checkpoint: checkpoint.path, runId: checkpoint.metadata.runId ?? null,
    context: checkpoint.model.stateInput ? "hybrid" : "tree",
    mode: config.teacher ? "float-teacher-only" : "float-vs-stored-quantized",
    checkpointLabelSource: checkpoint.metadata.labelSource,
    sample: {
      shard, selection: "deterministic-shard-prefix", isFullCorpusScore: false,
      maxParts, maxFiles, parts: data.tokenCount, files: files.length,
      limitReached: data.tokenCount >= maxParts || files.length >= maxFiles,
      truncatedFiles: files.filter((file) => file.truncated).length,
      labelSource: "direct-raw-shiki-sourceLabels", records: files,
    },
    notes: [
      "Bounded diagnostic sample, not a full-corpus score or a measured model capacity ceiling.",
      "Both variants use exactly the same records, prefix context, ranges, and labels; truncation changes tree context.",
      "Character units are UTF-16 code units. ASCII whitespace and unlabeled source positions are excluded; confidence does not filter raw-label scores.",
      "Styled spans are maximal same-class runs on the scored grid; whitespace and unlabeled gaps break spans. Boundaries exclude file edges and gaps.",
      "Region counts overlap and use raw Shiki auxiliary scope bits. Macro F1 averages all eight styled classes (absent classes contribute zero).",
      ...(config.teacher ? ["Offline float teacher: no quantization is performed and no deployed score is claimed."] : [
        "Quantized inference uses the checkpoint's stored packed artifact with the JavaScript CPU reference; this is not GPU parity verification or proof the run was promoted.",
        "Quantization gap is float minus quantized (percentage points); positive values mean degradation after quantization.",
      ]),
    ],
    float: { precision: "float32", ...floatMetrics },
    quantized: packed ? { precision: `int${packed.bits}`, artifact: packed.path, ...quantizedMetrics } : null,
    quantizationGap: packed ? {
      ...changes,
      partAccuracyPoints: gap(floatMetrics.parts.accuracy, quantizedMetrics.parts.accuracy),
      characterAccuracyPoints: gap(floatMetrics.characters.accuracy, quantizedMetrics.characters.accuracy),
      characterMacroF1Points: gap(floatMetrics.characters.macroF1, quantizedMetrics.characters.macroF1),
      spanF1Points: gap(floatMetrics.spans.f1, quantizedMetrics.spans.f1),
      boundaryF1Points: gap(floatMetrics.boundary.f1, quantizedMetrics.boundary.f1),
    } : null,
  };
}

function requireSource(record) {
  if (typeof record.source !== "string" || !Array.isArray(record.sourceLabels) ||
      record.ranges?.length !== record.features.length * 2) {
    throw new Error("loadTreeShard({ retainSource: true }) must retain source, sourceLabels, and flat ranges");
  }
  let end = 0;
  for (let index = 0; index < record.ranges.length; index += 2) {
    const from = record.ranges[index], to = record.ranges[index + 1];
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < end || to <= from || to > record.source.length) {
      throw new Error("diagnostic tree ranges must be sorted disjoint UTF-16 source ranges");
    }
    end = to;
  }
}

function validateCheckpoint({ metadata, model }) {
  if (metadata.model !== TREE_MODEL || ![TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION].includes(metadata.formatVersion) ||
      !isTreeFeatureVersion(metadata.featureVersion) || metadata.tokenizerVersion !== TREE_TOKENIZER_VERSION ||
      JSON.stringify(metadata.classNames) !== JSON.stringify(classNames)) {
    throw new Error("checkpoint is incompatible with current tree features, tokenizer, format, or classes");
  }
  if (metadata.labelSource !== "shiki-spans-v1") throw new Error("checkpoint lacks direct-label provenance");
  const config = {
    hiddenSize: metadata.hiddenSize, classifierSize: metadata.architecture?.classifierDimensions,
    hashBuckets: metadata.architecture?.lexemeHashBuckets,
    featureVersion: metadata.featureVersion,
    teacher: metadata.config?.teacherMode === true || metadata.precision === "float32" || metadata.config?.precision === "float32",
  };
  positiveInteger(config.hiddenSize, "checkpoint hiddenSize");
  positiveInteger(config.classifierSize, "checkpoint classifierSize");
  if (config.hiddenSize < 2 || (config.hiddenSize & (config.hiddenSize - 1)) ||
      !Number.isInteger(config.hashBuckets) ||
      metadata.inputSize !== treeInputSize(config.hashBuckets, metadata.featureVersion)) {
    throw new Error("invalid checkpoint tree dimensions");
  }
  const names = treeTensorNamesFor(model);
  if (JSON.stringify(Object.keys(model).sort()) !== JSON.stringify([...names].sort()) ||
      Boolean(model.stateInput) !== (metadata.formatVersion === TREE_HYBRID_FORMAT_VERSION)) {
    throw new Error("checkpoint tensor names do not match its tree/hybrid format");
  }
  let offset = 0;
  const layout = treeTensorLayout(model, { ...config, inputSize: metadata.inputSize });
  for (const [index, tensor] of layout.entries()) {
    const stored = metadata.tensorLayout[index];
    const length = tensor.shape.reduce((product, value) => product * value, 1);
    if (stored?.name !== tensor.name || stored.offset !== offset || stored.length !== length || model[tensor.name].length !== length ||
        model[tensor.name].some((value) => !Number.isFinite(value))) {
      throw new Error(`invalid checkpoint tensor ${tensor.name}`);
    }
    offset += length;
  }
  return config;
}

async function loadPackedModel(checkpoint) {
  const quantization = checkpoint.metadata.quantization;
  if (!quantization || ![4, 5, 6].includes(quantization.bits)) {
    throw new Error("non-teacher checkpoint requires stored int4/int5/int6 quantization metadata; no re-quantization fallback is allowed");
  }
  const names = treeTensorNamesFor(checkpoint.model);
  if (!Array.isArray(quantization.tensors) || quantization.tensors.length !== names.length) throw new Error("invalid packed tensor layout");
  let offset = 0;
  for (const [index, name] of names.entries()) {
    const tensor = quantization.tensors[index];
    if (tensor.name !== name || tensor.offset !== offset || tensor.length !== checkpoint.model[name].length ||
        !Number.isFinite(tensor.scale) || tensor.scale <= 0) throw new Error(`invalid packed tensor ${name}`);
    offset += tensor.length;
  }
  const byteLength = Math.ceil(offset * quantization.bits / 8);
  if (quantization.parameterCount !== offset || quantization.packedByteLength !== byteLength) throw new Error("invalid packed weight size metadata");
  const pattern = new RegExp(`^weights-int${quantization.bits}-.+\\.bin$`);
  const files = (await readdir(checkpoint.path)).filter((name) => pattern.test(name));
  if (files.length !== 1) throw new Error(`expected exactly one stored int${quantization.bits} artifact, found ${files.length}`);
  const path = resolve(checkpoint.path, files[0]);
  const data = await readFile(path);
  if (data.byteLength !== byteLength) throw new Error("packed weight artifact byte length does not match metadata");
  return { path, bits: quantization.bits, model: dequantizeTensors(data, quantization) };
}

function predict(model, record, config) {
  return treeProbabilities(model, record, config).map((row) => {
    if (row.length !== classNames.length || row.some((value) => !Number.isFinite(value))) throw new Error("non-finite or invalid model probabilities");
    let best = 0;
    for (let index = 1; index < row.length; index++) if (row[index] > row[best]) best = index;
    return best;
  });
}

function confusion() { return classNames.map(() => classNames.map(() => 0)); }
function counts() { return { matched: 0, expected: 0, predicted: 0 }; }
function accumulator() {
  return {
    parts: confusion(), characters: confusion(), spans: counts(), boundary: counts(),
    spanClasses: classNames.slice(1).map(() => counts()),
    byLanguage: new Map(), byRegion: new Map(),
  };
}
function scoredPart(record, aligned, index) {
  const kind = record.features[index][0];
  return kind !== TREE_PART_SPACE && kind !== TREE_PART_NEWLINE && aligned[index].coverage > 0;
}
function group(map, name) {
  if (!map.has(name)) map.set(name, { parts: confusion(), characters: confusion() });
  return map.get(name);
}
function scoreRecord(total, record, source, expected, aligned, predictions) {
  const spans = predictions.map((selected, index) => ({
    from: record.ranges[index * 2], to: record.ranges[index * 2 + 1], class: classNames[selected],
  }));
  const audit = auditSourceLabels(source, expected, spans);
  addMatrix(total.characters, audit.characters.confusion);
  addCounts(total.spans, audit.spans);
  addCounts(total.boundary, audit.boundary);
  audit.spans.perClass.forEach((value, index) => addCounts(total.spanClasses[index], value));
  const language = group(total.byLanguage, record.language ?? "unknown");
  addMatrix(language.characters, audit.characters.confusion);
  const regionNames = ["unscoped", ...auxiliaryNames];
  for (const [regionIndex, name] of regionNames.entries()) {
    const inRegion = (bits) => regionIndex === 0 ? !bits : Boolean(bits & (1 << (regionIndex - 1)));
    const labels = expected.filter((span) => inRegion(span.auxiliary ?? 0));
    const region = group(total.byRegion, name);
    if (labels.length) addMatrix(region.characters, auditSourceLabels(source, labels, spans).characters.confusion);
    for (let index = 0; index < predictions.length; index++) {
      if (scoredPart(record, aligned, index) && inRegion(aligned[index].auxiliary)) {
        region.parts[classNames.indexOf(aligned[index].class)][predictions[index]]++;
      }
    }
  }
  for (let index = 0; index < predictions.length; index++) {
    if (!scoredPart(record, aligned, index)) continue;
    const target = classNames.indexOf(aligned[index].class);
    total.parts[target][predictions[index]]++;
    language.parts[target][predictions[index]]++;
  }
}
function addMatrix(target, source) {
  for (let row = 0; row < target.length; row++) for (let column = 0; column < target.length; column++) target[row][column] += source[row][column];
}
function addCounts(target, source) { for (const key of ["matched", "expected", "predicted"]) target[key] += source[key]; }
function prf({ matched, expected, predicted }) {
  return { matched, expected, predicted, precision: matched / Math.max(1, predicted), recall: matched / Math.max(1, expected), f1: 2 * matched / Math.max(1, predicted + expected) };
}
function matrixMetrics(matrix) {
  const perClass = classNames.map((name, index) => {
    const expected = matrix[index].reduce((sum, count) => sum + count, 0);
    const predicted = matrix.reduce((sum, row) => sum + row[index], 0);
    const matched = matrix[index][index];
    return { class: name, support: expected, errors: expected - matched, ...prf({ matched, expected, predicted }) };
  });
  const total = perClass.reduce((sum, value) => sum + value.support, 0);
  const correct = perClass.reduce((sum, value) => sum + value.matched, 0);
  return {
    total, correct, errors: total - correct, accuracy: total ? correct / total : null,
    macroF1: perClass.slice(1).reduce((sum, value) => sum + value.f1, 0) / (classNames.length - 1),
    falseColorRate: perClass[0].support ? perClass[0].errors / perClass[0].support : null,
    confusion: matrix, perClass,
  };
}
function summarize(total) {
  const groups = (map) => Object.fromEntries([...map.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([name, value]) => [name, {
    parts: matrixMetrics(value.parts), characters: matrixMetrics(value.characters),
  }]));
  return {
    parts: matrixMetrics(total.parts), characters: matrixMetrics(total.characters),
    spans: { ...prf(total.spans), perClass: total.spanClasses.map((value, index) => ({ class: classNames[index + 1], ...prf(value) })) },
    boundary: prf(total.boundary), byLanguage: groups(total.byLanguage), byRegion: groups(total.byRegion),
  };
}
function gap(a, b) { return a == null || b == null ? null : 100 * (a - b); }
function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive finite safe integer`);
  return value;
}

export function parseArguments(args) {
  const options = {};
  const values = { run: "run", shard: "shard", "max-parts": "maxParts", "max-files": "maxFiles" };
  const flags = { help: "help" };
  for (let index = 0; index < args.length; index++) {
    if (args[index] === "--") continue;
    if (!args[index].startsWith("--")) {
      if (options.run) throw new Error(`unexpected argument ${args[index]}`);
      options.run = args[index];
      continue;
    }
    const match = /^--([^=]+)(?:=(.*))?$/.exec(args[index]);
    if (!match) throw new Error(`invalid argument ${args[index]}`);
    const [, name, inline] = match;
    if (flags[name]) {
      if (inline !== undefined) throw new Error(`--${name} does not take a value`);
      options[flags[name]] = true;
    } else if (values[name]) {
      const value = inline ?? args[++index];
      if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value`);
      options[values[name]] = name.startsWith("max-") ? positiveInteger(Number(value), name) : value;
    } else throw new Error(`unknown option --${name}`);
  }
  return options;
}

const usage = `Usage: node src/evaluate-tree.js --run <run-id-or-path> [--shard <verification.jsonl.gz>]
  --max-parts <n>              Part budget, including whitespace context (default ${defaults.maxParts})
  --max-files <n>              File budget (default ${defaults.maxFiles})

Read-only JSON diagnostic on a bounded prefix, NOT a full-corpus score. Float teacher
checkpoints are float-only. Other checkpoints require their stored packed weights.
No training, re-quantization, artifact writes, or promotion occurs.`;

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const options = parseArguments(process.argv.slice(2));
    if (options.help) console.log(usage);
    else console.log(JSON.stringify(await evaluateTree(options), null, 2));
  } catch (error) {
    console.error(`evaluate-tree: ${error.message}`);
    process.exitCode = 1;
  }
}

import {
  runtimeTreeFeatureLayout, unreachableTreeFeatureRows,
} from "./tree-features.js";

// Runtime order is independent from the training checkpoint schema.
//
// The classifier heads come first. The classifier matrix is the hottest loop
// of the runtime, and on Apple GPUs it runs about 40% slower when any of its
// vec4 indices reaches 4096 (byte offset 65,536): shifting the buffer in
// memory changes nothing, so the difference is in the code Metal generates
// for small index ranges. outputBias, the only tensor whose length is not a
// multiple of four, comes last so that every other tensor starts on a vec4
// boundary. The rest of the order has no measured effect on speed and was
// selected against the promoted model's Brotli-11 size, as built and as
// re-bundled by the benchmark's size script.
const RUNTIME_TENSOR_ORDER = [
  "auxiliaryOutput", "auxiliaryBias", "classifierInput", "classifierBias", "stateGate",
  "output", "downSkip", "localOffsetScale", "downOwnSelf", "mergeOwnLeft",
  "mergeOwnRight", "downRightBias", "mergeCrossRight", "mergeBias", "downOwnSibling",
  "downCrossParent", "downCrossSelf", "downOwnParent", "localNonspaceScale", "leafBias",
  "downLeftBias", "mergeCrossLeft", "stateInputBias", "downCrossSibling", "stateMix",
  "stateMixBias", "featureEmbedding", "stateInput", "stateGateBias", "outputBias",
];

export function isHybridTree(model) {
  return model?.model === "hierarchical-tree" && model.formatVersion === 9;
}

export function validTreeContext(model) {
  return isHybridTree(model) && model.architecture?.context === "local-affine-tree" &&
    model.architecture?.localRadius === 2;
}

export function validateHybridTensors(model) {
  if (!isHybridTree(model)) return;
  const h = model.hiddenSize;
  const lengths = {
    neighborScale: 3 * h, localOffsetScale: 5 * h, localNonspaceScale: 2 * h,
    stateInput: h * h, stateInputBias: h, stateGate: h * h, stateGateBias: h,
    stateMix: h * 2 * h, stateMixBias: h,
  };
  const tensors = model.quantization?.tensors ?? [];
  for (const [name, length] of Object.entries(lengths)) {
    const matches = tensors.filter((tensor) => tensor.name === name);
    const tensor = matches[0];
    if (matches.length !== 1 || tensor.length !== length || !Number.isInteger(tensor.offset) ||
        tensor.offset < 0 || tensor.offset + length > model.quantization.parameterCount) {
      throw new Error(`invalid hybrid runtime tensor ${name}`);
    }
  }
}

export function runtimeTensorLayout(model) {
  validateHybridTensors(model);
  const sourceTensors = model.quantization.tensors;
  const tensors = new Map(sourceTensors.map((tensor) => [tensor.name, tensor]));
  const expected = new Set([...RUNTIME_TENSOR_ORDER, "neighborScale"]);
  if (sourceTensors.some(({ name }) => !expected.has(name)) ||
      expected.size !== sourceTensors.length) {
    throw new Error("promoted hierarchical-tree tensor layout is incompatible with this runtime");
  }

  const hidden = model.hiddenSize;
  const hashBuckets = model.architecture.lexemeHashBuckets ?? 128;
  const compactFeatures = runtimeTreeFeatureLayout(hashBuckets, model.featureVersion);
  const omittedRows = unreachableTreeFeatureRows(hashBuckets, model.featureVersion);
  let offset = 0;
  return RUNTIME_TENSOR_ORDER.map((name) => {
    const source = tensors.get(name);
    if (!source) throw new Error(`promoted hierarchical-tree model is missing ${name}`);
    const sourceRanges = name === "featureEmbedding"
      ? embeddingSourceRanges(source, hidden, model.inputSize, omittedRows)
      : [source.offset, source.offset + source.length];
    const length = name === "featureEmbedding"
      ? compactFeatures.inputSize * hidden
      : source.length;
    const transpose = transposedShape(model, name);
    const quads = quadShape(model, name);
    const shape = transpose ?? quads;
    if (shape && shape[0] * shape[1] !== length) {
      throw new Error(`invalid runtime matrix shape for ${name}`);
    }
    if (quads && quads[0] % 4) {
      throw new Error(`runtime tensor ${name} needs a multiple of four output units`);
    }
    const tensor = { name, offset, length, scale: source.scale, sourceRanges, transpose, quads };
    offset += length;
    return tensor;
  });
}

// The shader reads these [rows][columns] matrices with the row (output unit)
// varying across adjacent threads. The runtime buffer stores them column-major
// so those threads read adjacent weights; values and summation order are
// unchanged.
function transposedShape(model, name) {
  const hidden = model.hiddenSize;
  if (name === "stateInput" || name === "stateGate") return [hidden, hidden];
  if (name === "stateMix") return [hidden, hidden * 2];
  if (name === "output") return [model.outputSize, model.architecture.classifierDimensions];
  return undefined;
}

// The classifier and auxiliary heads compute four output units per thread.
// Their [rows][columns] matrices are stored as [rows / 4][columns][4], so a
// thread reads its four units' weights for one input in a single vec4 load
// and walks one contiguous run of the buffer. Values and summation order are
// unchanged.
function quadShape(model, name) {
  const hidden = model.hiddenSize;
  const auxiliary = model.architecture.auxiliaryStates.length;
  if (name === "classifierInput") return [model.architecture.classifierDimensions, hidden * 2 + auxiliary];
  if (name === "auxiliaryOutput") return [auxiliary, hidden * 2];
  return undefined;
}

function embeddingSourceRanges(tensor, hidden, inputSize, omittedRows) {
  if (tensor.length !== inputSize * hidden) {
    throw new Error("invalid feature embedding tensor length");
  }
  const ranges = [];
  let firstRow = 0;
  for (const row of omittedRows) {
    if (row > firstRow) ranges.push(
      tensor.offset + firstRow * hidden,
      tensor.offset + row * hidden,
    );
    firstRow = row + 1;
  }
  if (firstRow < inputSize) ranges.push(
    tensor.offset + firstRow * hidden,
    tensor.offset + inputSize * hidden,
  );
  return ranges;
}

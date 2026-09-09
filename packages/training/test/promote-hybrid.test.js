import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { classNames } from "../src/classes.js";
import { promote, promotionRejection, validateModel } from "../src/promote.js";
import { quantizeTensors } from "../src/quantization.js";
import {
  createTreeModel, treeAuxiliaryNames, treeTensorNames, treeTensorNamesFor,
} from "../src/tree-model.js";
import { treeInputSize } from "../../core/src/tree-features.js";

const hybridLengths = (hidden) => ({
  localOffsetScale: 5 * hidden, localNonspaceScale: 2 * hidden,
  stateInput: hidden * hidden, stateInputBias: hidden,
  stateGate: hidden * hidden, stateGateBias: hidden,
  stateMix: hidden * 2 * hidden, stateMixBias: hidden,
});

function fixture({ formatVersion = 9, hiddenSize = 32, classifierSize = 72, hashBuckets = 128, bits = 4 } = {}) {
  const model = createTreeModel({ hiddenSize, classifierSize, hashBuckets,
    featureVersion: 2, scaleBuckets: 8, random: () => 0.5 });
  // Build the appended tensors independently so validation is checked against
  // the wire contract, not merely the trainer's own layout implementation.
  if (formatVersion === 9) for (const [name, length] of Object.entries(hybridLengths(hiddenSize))) {
    model[name] = new Float32Array(length);
  }
  const names = [...treeTensorNames, ...(formatVersion === 9 ? Object.keys(hybridLengths(hiddenSize)) : [])];
  const quantized = quantizeTensors(model, names, bits);
  const metadata = {
    formatVersion, model: "hierarchical-tree", featureVersion: 2,
    inputSize: treeInputSize(hashBuckets, 2), hiddenSize,
    config: { weightBits: bits },
    runtimeParameterCount: quantized.parameterCount, runtimeWeightBytes: quantized.data.byteLength,
    architecture: {
      direction: "bidirectional", tree: "scale-aware-butterfly-binary", blockParts: 32, scaleBuckets: 8,
      classifierDimensions: classifierSize, lexemeHashBuckets: hashBuckets,
      localNeighborParts: formatVersion === 9 ? 2 : 1,
      auxiliaryStates: treeAuxiliaryNames, classifierAuxiliaryStates: true,
      ...(formatVersion === 9 ? { context: "local-affine-tree", localRadius: 2 } : {}),
    },
    classNames, auxiliaryNames: treeAuxiliaryNames, quantization: quantized.metadata,
    verification: { accuracy: 0.8, macroF1: 0.7 },
  };
  return { metadata, weights: quantized.data, model, names };
}

for (const formatVersion of [8, 9]) for (const bits of [4, 5, 6]) {
  test(`promotion validates format ${formatVersion} int${bits} without generating artifacts`, () => {
    const { metadata, weights, model, names } = fixture({ formatVersion, bits });
    assert.deepEqual(treeTensorNamesFor(model), names);
    assert.deepEqual(treeTensorNamesFor({ treeContext: formatVersion === 9 ? "hybrid" : "tree" }), names);
    assert.doesNotThrow(() => validateModel(metadata, weights));
  });
}

test("promotion validates feature-v3 collision features and 12 scales", () => {
  const hiddenSize = 32, classifierSize = 72, hashBuckets = 256;
  const model = createTreeModel({ treeContext: "hybrid", hiddenSize, classifierSize, hashBuckets,
    featureVersion: 3, scaleBuckets: 12, random: () => 0.5 });
  const quantized = quantizeTensors(model, treeTensorNamesFor(model), 6);
  const metadata = {
    formatVersion: 9, model: "hierarchical-tree", featureVersion: 3,
    inputSize: treeInputSize(hashBuckets, 3), hiddenSize,
    runtimeParameterCount: quantized.parameterCount, runtimeWeightBytes: quantized.data.byteLength,
    architecture: {
      direction: "bidirectional", tree: "scale-aware-butterfly-binary", blockParts: 32, scaleBuckets: 12,
      classifierDimensions: classifierSize, lexemeHashBuckets: hashBuckets,
      secondaryLexemeHashBuckets: 128, neighborSymbolHashBuckets: 32,
      localNeighborParts: 2, context: "local-affine-tree", localRadius: 2,
      auxiliaryStates: treeAuxiliaryNames, classifierAuxiliaryStates: true,
    },
    classNames, auxiliaryNames: treeAuxiliaryNames, quantization: quantized.metadata,
    verification: { accuracy: 0.86, macroF1: 0.75 },
  };
  assert.equal(quantized.parameterCount, 41_609);
  assert.equal(quantized.data.byteLength, 31_207);
  assert.doesNotThrow(() => validateModel(metadata, quantized.data));
});

for (const [name] of Object.entries(hybridLengths(32))) {
  test(`hybrid promotion checks the ${name} dimension`, () => {
    const { metadata, weights } = fixture();
    metadata.quantization.tensors.find((tensor) => tensor.name === name).length += 1;
    assert.throws(() => validateModel(metadata, weights), new RegExp(`invalid tree tensor ${name}`));
  });
}

for (const mutation of ["missing", "reordered", "extra"]) {
  test(`hybrid promotion rejects ${mutation} tensors`, () => {
    const { metadata, weights } = fixture();
    const tensors = metadata.quantization.tensors;
    if (mutation === "missing") tensors.pop();
    if (mutation === "reordered") [tensors[23], tensors[24]] = [tensors[24], tensors[23]];
    if (mutation === "extra") tensors.push({ name: "unexpected", length: 32, offset: metadata.runtimeParameterCount, scale: 1 });
    assert.throws(() => validateModel(metadata, weights), /tensor layout/);
  });
}

test("format 8 accepts the new explicit legacy context metadata", () => {
  const { metadata, weights } = fixture({ formatVersion: 8 });
  Object.assign(metadata.architecture, { context: "tree", localRadius: 1 });
  assert.doesNotThrow(() => validateModel(metadata, weights));
});

test("hybrid promotion rejects the legacy neighbor radius", () => {
  const { metadata, weights } = fixture();
  metadata.architecture.localNeighborParts = 1;
  assert.throws(() => validateModel(metadata, weights), /architecture/);
});

for (const architecture of [{ context: "tree" }, { context: undefined }, { localRadius: 1 }, { localRadius: undefined }]) {
  test(`hybrid promotion rejects mismatched context ${JSON.stringify(architecture)}`, () => {
    const { metadata, weights } = fixture();
    Object.assign(metadata.architecture, architecture);
    assert.throws(() => validateModel(metadata, weights), /context.*format version/);
  });
}

test("format 8 cannot silently carry hybrid context or hybrid tensors", () => {
  const { metadata, weights } = fixture();
  metadata.formatVersion = 8;
  metadata.architecture.localNeighborParts = 1;
  assert.throws(() => validateModel(metadata, weights), /context.*format version/);
  delete metadata.architecture.context;
  delete metadata.architecture.localRadius;
  assert.throws(() => validateModel(metadata, weights), /tensor layout/);
});

for (const options of [
  { hiddenSize: 64, classifierSize: 72, bits: 6 },
  { hiddenSize: 64, classifierSize: 256, hashBuckets: 256, bits: 4 },
]) {
  test(`hybrid deploy budgets stay strict: ${JSON.stringify(options)}`, () => {
    const { metadata, weights } = fixture(options);
    if (options.bits === 6) {
      assert.ok(metadata.runtimeParameterCount <= 75_000);
      assert.ok(weights.length > 37_500);
    } else assert.ok(metadata.runtimeParameterCount > 75_000);
    assert.throws(() => validateModel(metadata, weights), /tree int[46] layout/);
  });
}

for (const marker of [
  { config: { teacherMode: true } },
  { precision: "float32" },
  { config: { precision: "float32" } },
]) {
  test(`teacher rejection precedes checkpoint reads, quantization and corpus evaluation: ${JSON.stringify(marker)}`, async () => {
    const { metadata } = fixture();
    Object.assign(metadata, marker, {
      acceptance: { accepted: false, criterion: "rejected-runtime-candidate" },
    });
    assert.match(promotionRejection(metadata), /teacher\/float32/);
    assert.throws(() => validateModel(metadata, null), /teacher\/float32/);
    const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-promote-teacher-"));
    try {
      // Deliberately omit the checkpoint and corpus: the metadata guard must
      // win before either can be opened. This run can never write artifacts.
      await writeFile(resolve(directory, "model-test.json"), JSON.stringify(metadata));
      await assert.rejects(promote(directory), /teacher\/float32/);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
}

for (const formatVersion of [8, 9]) for (const labelSource of [undefined, "shiki-spans-v1"]) {
  test(`format ${formatVersion} promotion requires explicit objective before evaluating ${labelSource ?? "legacy"} runs`, async () => {
    const { metadata, model, names } = fixture({ formatVersion });
    const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-promote-labels-"));
    try {
      const shard = resolve(directory, "verification.jsonl.gz");
      metadata.config.verificationShard = shard;
      metadata.labelSource = labelSource;
      // A mandatory post-evaluation rejection guarantees no generated files
      // are written even when the corpus is accepted.
      metadata.acceptance = { accepted: false, criterion: "test-only-never-promote" };
      let offset = 0;
      metadata.tensorLayout = names.map((name) => {
        const tensor = { name, offset, length: model[name].length };
        offset += tensor.length;
        return tensor;
      });
      const floats = new Float32Array(offset);
      for (const tensor of metadata.tensorLayout) floats.set(model[tensor.name], tensor.offset);
      await writeFile(resolve(directory, "model-test.json"), JSON.stringify(metadata));
      await writeFile(resolve(directory, "weights-f32-test.bin"), Buffer.from(floats.buffer));
      await writeFile(shard, gzipSync(JSON.stringify({
        source: "x", language: "javascript", sourceName: "fixture", path: "fixture.js",
        tokens: [{ from: 0, to: 1, class: "plain", auxiliary: 0, confidence: 255 }],
      }) + "\n"));
      // The objective guard now fails before shard access; direct-label rejection
      // with a valid objective/baseline is covered in language-objective.test.js.
      await assert.rejects(promote(directory), /invalid language objective provenance/);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
}

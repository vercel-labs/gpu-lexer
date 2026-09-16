import assert from "node:assert/strict";
import test from "node:test";
import { initialize, validate } from "wgslender";
import {
  BUFFER_HYBRID_LOCAL, BUFFER_LEAF_STATES, BUFFER_SCRATCH,
  createInspectableRuntime as createRuntime,
} from "../src/gpu.js";
import { runtimeTensorLayout } from "../src/model-layout.js";
import { createPromotedShader } from "../src/shader.min.generated.js";
import { createTreeShader } from "../src/tree-shader.js";
import { promotedModel } from "../src/model.generated.js";
import { treeInputSize } from "../src/tree-features.js";

const entries = ["hybrid_neighbor_blocks", "hybrid_neighbor_prefixes", "hybrid_scan_blocks",
  "hybrid_scan_prefixes", "hybrid_mix_tree_up", "tree_global", "tree_down_classify"];

function fixture(hiddenSize = 32, bits = 6, hybrid = true, featureVersion = 2) {
  const h = hiddenSize;
  const hashBuckets = featureVersion >= 3 ? 256 : 128;
  const scaleBuckets = featureVersion >= 3 ? 12 : 8;
  const inputSize = treeInputSize(hashBuckets, featureVersion);
  const lengths = {
    featureEmbedding: inputSize * h, neighborScale: 3 * h, leafBias: h,
    ...(hybrid ? {
      localOffsetScale: 5 * h, localNonspaceScale: 2 * h,
      stateInput: h * h, stateInputBias: h, stateGate: h * h, stateGateBias: h,
      stateMix: h * 2 * h, stateMixBias: h,
    } : {}),
    ...Object.fromEntries(["mergeOwnLeft", "mergeOwnRight", "mergeCrossLeft", "mergeCrossRight", "mergeBias",
      "downOwnParent", "downOwnSelf", "downOwnSibling", "downCrossParent", "downCrossSelf",
      "downCrossSibling", "downSkip", "downLeftBias", "downRightBias"].map((name) => [name, scaleBuckets * h])),
    classifierInput: 16 * (h * 2 + 3), classifierBias: 16, output: 9 * 16, outputBias: 9,
    auxiliaryOutput: 3 * h * 2, auxiliaryBias: 3,
  };
  let parameterCount = 0;
  const tensors = Object.entries(lengths).map(([name, length]) => {
    const tensor = { name, length, offset: parameterCount, scale: 0.01 };
    parameterCount += length;
    return tensor;
  });
  return {
    model: "hierarchical-tree", formatVersion: hybrid ? 9 : 8, featureVersion,
    inputSize, hiddenSize, outputSize: 9,
    config: { treeContext: hybrid ? "hybrid" : "tree" },
    architecture: {
      direction: "bidirectional", tree: "scale-aware-butterfly-binary", blockParts: 32, scaleBuckets,
      classifierDimensions: 16, auxiliaryStates: ["a", "b", "c"], classifierAuxiliaryStates: true,
      lexemeHashBuckets: hashBuckets,
      ...(featureVersion >= 3 ? { secondaryLexemeHashBuckets: 128, neighborSymbolHashBuckets: 32 } : {}),
      ...(hybrid ? { context: "local-affine-tree", localRadius: 2 } : {}),
    },
    quantization: { type: `symmetric-int${bits}-per-tensor`, bits, parameterCount, tensors },
    weightEncoding: "signed-symbol-v1", weights: "A".repeat(parameterCount),
  };
}

test("feature-v3 collision features and 12 scales produce valid WGSL", async () => {
  await initialize();
  const shader = createTreeShader(fixture(32, 6, true, 3));
  const result = validate(shader);
  assert.equal(result.valid, true, result.diagnostics.map(({ message }) => message).join("; "));
  assert.match(shader, /context >> 15u/);
  assert.match(shader, /context >> 22u/);
  assert.match(shader, /context >> 27u/);
});

for (const hidden of [32]) for (const f16 of [false, true]) {
  test(`format9 hybrid WGSL validates H=${hidden} f16=${f16}`, async () => {
    await initialize();
    const model = fixture(hidden);
    const shader = createTreeShader(model, { f16 });
    for (const entry of entries) assert.match(shader, new RegExp(`fn ${entry}\\(`));
    const result = validate(shader);
    assert.equal(result.valid, true, result.diagnostics.map(({ message }) => message).join("; "));
    assert.match(shader, /kind != 1u && kind != 2u/);
    assert.match(shader, /token \+ d >= start \+ 2u/);
    assert.match(shader, /@workgroup_size\(256\)\s+fn tree_global/);
    assert.match(shader, /bitcast<f32>\(previous\)/);
    assert.doesNotMatch(shader, /fn hybrid_local_states/);
    assert.doesNotMatch(shader, /var<storage, read_write> hybrid_pairs/);
    assert.match(shader, new RegExp(`var<workgroup> hybrid_work: array<vec4<f32>, ${32 * hidden}>`));
    assert.match(shader, /var<workgroup> classifier_projected/);
    assert.doesNotMatch(shader, /tree_down:|contexts:/);
    assert.equal(runtimeTensorLayout(model).some(({ name }) => name === "neighborScale"), false);
  });
}

test("format9 is opt-in and rejects mismatched metadata or hybrid tensors", () => {
  const mutations = [
    (m) => { m.architecture.context = "tree"; },
    (m) => { m.architecture.localRadius = 1; },
    (m) => { m.formatVersion = 8; },
    (m) => { m.quantization.tensors = m.quantization.tensors.filter(({ name }) => name !== "stateGate"); },
    (m) => { m.quantization.tensors.find(({ name }) => name === "stateMix").length--; },
    (m) => { m.hiddenSize = 64; },
  ];
  for (const mutate of mutations) {
    const model = fixture();
    mutate(model);
    assert.throws(() => createTreeShader(model), /incompatible|invalid hybrid/);
  }
});

test("runtime hybrid metadata does not require training-only config", () => {
  const model = fixture();
  delete model.config;
  assert.doesNotThrow(() => createTreeShader(model));
});

function fakeDevice(f16 = false) {
  const dispatches = [];
  const bindGroups = [];
  let computePasses = 0;
  let code;
  return {
    limits: { maxBufferSize: Infinity },
    dispatches, bindGroups, get code() { return code; }, get computePasses() { return computePasses; },
    features: new Set(f16 ? ["shader-f16"] : []),
    queue: { writeBuffer() {}, submit() {} },
    createShaderModule(descriptor) { code = descriptor.code; return {}; },
    createComputePipelineAsync({ compute: { entryPoint } }) {
      return { entryPoint, getBindGroupLayout() { return entryPoint; } };
    },
    createBuffer({ size }) {
      return { size, mapState: "unmapped", destroy() {},
        async mapAsync() { this.mapState = "mapped"; },
        getMappedRange() { return new ArrayBuffer(size); },
        unmap() { this.mapState = "unmapped"; } };
    },
    createBindGroup(descriptor) { bindGroups.push(descriptor); return descriptor; },
    createCommandEncoder() {
      return {
        beginComputePass() {
          computePasses += 1;
          let entry;
          return { setPipeline(p) { entry = p.entryPoint; }, setBindGroup() {},
            dispatchWorkgroups(x, y) { dispatches.push([entry, x, y]); }, end() {} };
        },
        clearBuffer() {}, copyBufferToBuffer() {}, finish() { return {}; },
      };
    },
  };
}

test("tree runtime wires the active hybrid passes and reuses buffers", async () => {
  const device = fakeDevice(true);
  const runner = createRuntime({ device, shader: createPromotedShader(true) });
  const streams = Uint32Array.of(0, 65, 65, 1);
  await runner.run(new Uint32Array(66 * 2), streams, (labels) => assert.equal(labels.length, 17));
  assert.deepEqual(device.dispatches, [
    ["a", 1, 1], ["b", 1, 1], ["c", 4, 1], ["d", 8, 1],
    ["e", 4, 1], ["f", 2, 1], ["g", 4, 1],
  ]);
  assert.equal(device.computePasses, 1);
  const up = device.bindGroups.find(({ layout }) => layout === "e");
  assert.equal(up.entries.some(({ binding }) => binding === 0), false);
  for (const group of device.bindGroups) {
    // The WebGPU baseline allows at most 8 storage buffers per stage.
    assert.ok(group.entries.filter(({ binding }) => binding !== 2).length <= 8);
  }
  assert.equal(runner.treeBindGroups[4], up);
  assert.equal(runner.buffers[BUFFER_SCRATCH][2], 4 * 32 * 16);
  assert.equal(runner.buffers[BUFFER_HYBRID_LOCAL][2], 66 * 32 * 4);
  assert.equal(runner.buffers[BUFFER_LEAF_STATES][2], 66 * 32 * 2);
  assert.match(device.code, /fn f\(/);
  const before = device.bindGroups.length;
  await runner.run(new Uint32Array(66 * 2), streams, () => {});
  assert.equal(device.bindGroups.length, before);
  await runner.run(new Uint32Array(200 * 2), Uint32Array.of(0, 200), () => {});
  assert.ok(device.bindGroups.length > before);
});

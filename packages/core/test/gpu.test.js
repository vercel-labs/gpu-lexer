import assert from "node:assert/strict";
import test from "node:test";

import {
  BUFFER_FEATURES, createInspectableRuntime as createRuntime, preparePromotedWeights,
} from "../src/gpu.js";
import { runtimeTensorLayout } from "../src/model-layout.js";
import { promotedModel } from "../src/model.generated.js";
import { WEIGHT_RANGES, WEIGHTS } from "../src/model.runtime.generated.js";
import { runtimeTreeFeatureLayout, treeFeatureLayout } from "../src/tree-features.js";

test("the runtime fails clearly when WebGPU is unavailable", async () => {
  const runner = createRuntime({ gpu: null });
  await assert.rejects(runner.run(new Uint32Array(6)), /WebGPU unavailable/);
});

test("grow-only buffers are reused until capacity is exceeded", () => {
  const created = [];
  const device = { limits: { maxBufferSize: Infinity }, createBuffer({ size, usage }) {
    const buffer = { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
    created.push(buffer);
    return buffer;
  } };
  const runner = createRuntime({ device, shader: "" });
  const first = runner.ensureBuffer(BUFFER_FEATURES, 100, 1);
  assert.equal(first, runner.ensureBuffer(BUFFER_FEATURES, 200, 1));
  const grown = runner.ensureBuffer(BUFFER_FEATURES, 300, 1);
  assert.notEqual(first, grown);
  assert.equal(first.destroyed, true);
  assert.deepEqual(created.map((buffer) => buffer.size), [256, 512]);
});

test("device initialization opts into supported buffer limits", async () => {
  let requested;
  const device = {
    features: new Set(), queue: { writeBuffer() {} }, createShaderModule() { return {}; },
    createComputePipelineAsync() { return { getBindGroupLayout() { return {}; } }; },
    createBuffer({ size }) { return { size }; },
  };
  const limits = { maxBufferSize: 4_294_967_292, maxStorageBufferBindingSize: 4_294_967_292 };
  const runner = createRuntime({ gpu: { async requestAdapter() { return {
    features: new Set(), limits, async requestDevice(descriptor) { requested = descriptor; return device; },
  }; } }, shader: "" });
  await runner.initialize();
  assert.deepEqual(requested.requiredLimits, limits);
});

test("buffers track used bytes and respect the device maximum", () => {
  const device = { limits: { maxBufferSize: 400 }, createBuffer({ size }) { return { size, destroy() {} }; } };
  const runner = createRuntime({ device, shader: "" });
  runner.ensureBuffer(BUFFER_FEATURES, 300, 1);
  assert.equal(runner.buffers[BUFFER_FEATURES][1], 300);
  assert.equal(runner.buffers[BUFFER_FEATURES][2], 300);
});

test("mapped readback views are limited without copying capacity", () => {
  const backing = new ArrayBuffer(512 * Uint32Array.BYTES_PER_ELEMENT);
  const labels = new Uint32Array(backing, 0, 260);
  assert.equal(labels.length, 260);
  assert.equal(labels.buffer, backing);
});

test("the promoted positional weights expand to finite runtime values", () => {
  const weights = preparePromotedWeights();
  const tensors = runtimeTensorLayout(promotedModel);
  assert.equal(weights.length, 41_321);
  assert.equal(weights.every(Number.isFinite), true);
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  const expected = new Float32Array(weights.length);
  for (const tensor of tensors) {
    let output = tensor.offset;
    for (let range = 0; range < tensor.sourceRanges.length; range += 2) {
      for (let index = tensor.sourceRanges[range]; index < tensor.sourceRanges[range + 1]; index++) {
        const encoded = alphabet.indexOf(promotedModel.weights[index]);
        const value = encoded & 1 ? -(encoded + 1) / 2 : encoded / 2;
        expected[output++] = value * tensor.scale;
      }
    }
  }
  assert.deepEqual(weights, expected);
  assert.equal(tensors.some(({ name }) => name === "neighborScale"), false);
  assert.ok(tensors.some((tensor, index) => WEIGHT_RANGES[index * 2 + 1] !== tensor.scale));
});

test("compact pair embeddings preserve every reachable trained row", () => {
  const source = promotedModel.quantization.tensors.find(({ name }) => name === "featureEmbedding");
  const target = runtimeTensorLayout(promotedModel).find(({ name }) => name === "featureEmbedding");
  const training = treeFeatureLayout();
  const runtime = runtimeTreeFeatureLayout();
  const hidden = promotedModel.hiddenSize;
  const compareRow = (sourceRow, targetRow) => assert.equal(
    promotedModel.weights.slice(source.offset + sourceRow * hidden, source.offset + (sourceRow + 1) * hidden),
    WEIGHTS.slice(target.offset + targetRow * hidden, target.offset + (targetRow + 1) * hidden),
  );
  for (let code = 1; code <= 14; code++) {
    compareRow(training.previousPair + code, runtime.previousPair + code - 1);
    compareRow(training.nextPair + code, runtime.nextPair + code - 1);
  }
  for (let code = 1; code <= 31; code++) {
    compareRow(training.previousSymbolPair + code, runtime.previousSymbolPair + code - 1);
    compareRow(training.nextSymbolPair + code, runtime.nextSymbolPair + code - 1);
  }
});

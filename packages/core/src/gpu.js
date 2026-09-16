import {
  HIDDEN_SIZE, WEIGHT_RANGES, WEIGHTS,
} from "./model.runtime.generated.js";
import { createPromotedShader, TREE_ENTRY_POINTS } from "./shader.min.generated.js";
import { prepareTreeSource, releaseTreePrepared } from "./prepare-tree.js";
import { spansFromRanges } from "./spans.js";
import { TREE_FEATURE_STRIDE } from "./constants.js";

const MAP_READ = 1;
const COPY_SRC = 4;
const COPY_DST = 8;
const UNIFORM = 64;
const STORAGE = 128;
const TREE_BLOCK_SIZE = 32;
const MAX_BATCH_TOKENS = 16_384;
export const PIPELINE_TREE_GLOBAL = 5;

export const BUFFER_FEATURES = 0;
export const BUFFER_LABELS = 1;
export const BUFFER_READBACK = 2;
export const BUFFER_PARAMS = 3;
export const BUFFER_STREAMS = 4;
export const BUFFER_BLOCKS = 5;
export const BUFFER_LEAF_STATES = 6;
export const BUFFER_TREE_UP = 7;
export const BUFFER_SCRATCH = 8;
export const BUFFER_HYBRID_NEIGHBORS = 9;
export const BUFFER_HYBRID_LOCAL = 10;
const BUFFER_WEIGHTS = -1;
const BUFFER_OBJECT = 0;
const BUFFER_SIZE = 1;
const BUFFER_BINDING_SIZE = 2;
const LAYOUT_STREAM_STORAGE = 0;
const LAYOUT_BLOCK_STORAGE = 1;
const LAYOUT_STREAMS = 2;
const LAYOUT_BLOCKS = 3;
const LAYOUT_COUNT = 4;
const LAYOUT_TREE_NODES = 5;
const LAYOUT_SOURCE = 6;

export function createRuntime(
  gpu = globalThis.navigator?.gpu,
  device,
  shader,
  runOverride,
) {
  let useF16 = false;
  let initializing;
  let pipelines;
  let weightBuffer;
  let treeBindGroups;
  let treeLayout;
  const buffers = [];
  const scanParams = new Uint32Array(4);
  const queue = [];
  let queuedTokens = 0;
  let scheduled = false;
  let flushing = false;
  let batchFeatures = new Uint32Array();
  let batchStreams = new Uint32Array();
  const run = runOverride ?? runGpu;
  async function initialize() {
    if (pipelines) return;
    if (initializing) return initializing;

    initializing = (async () => {
      if (!device) {
        if (!gpu) throw new Error("WebGPU unavailable");
        const adapter = await gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU unavailable");
        useF16 = adapter.features.has("shader-f16");
        const requiredLimits = {};
        for (const name of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
          requiredLimits[name] = adapter.limits[name];
        }
        device = await adapter.requestDevice({
          requiredFeatures: useF16 ? ["shader-f16"] : [],
          requiredLimits,
        });
      } else {
        useF16 = device.features.has("shader-f16");
      }

      if (shader === undefined) shader = createPromotedShader(useF16);
      const module = device.createShaderModule({ code: shader });
      const create = (entryPoint) => device.createComputePipelineAsync({
        layout: "auto", compute: { module, entryPoint },
      });
      pipelines = await Promise.all(Array.from(TREE_ENTRY_POINTS, create));

      const expanded = preparePromotedWeights();
      weightBuffer = device.createBuffer({
        size: expanded.byteLength,
        usage: STORAGE | COPY_DST,
      });
      device.queue.writeBuffer(weightBuffer, 0, expanded);
    })();

    try {
      await initializing;
    } catch (error) {
      initializing = undefined;
      throw error;
    }
  }

  async function runGpu(features, streams, consume) {
    await initialize();
    return runTree(features, streams, consume);
  }

  async function runTree(features, streams, consume) {
    const tokenCount = features.length / TREE_FEATURE_STRIDE;
    const labelBytes = (tokenCount + 3) & ~3;
    const featureBuffer = ensureBuffer(BUFFER_FEATURES, features.byteLength, STORAGE | COPY_DST);
    const labelBuffer = ensureBuffer(BUFFER_LABELS, labelBytes, STORAGE | COPY_SRC | COPY_DST);
    const readBuffer = ensureBuffer(BUFFER_READBACK, labelBytes, MAP_READ | COPY_DST);
    const paramsBuffer = ensureBuffer(BUFFER_PARAMS, 16, UNIFORM | COPY_DST);
    const layout = treeLayout = createTreeLayout(streams, treeLayout);
    const metadata = layout[LAYOUT_STREAMS];
    const blocks = layout[LAYOUT_BLOCKS];
    const blockCount = layout[LAYOUT_COUNT];
    const treeNodes = layout[LAYOUT_TREE_NODES];
    const streamBuffer = ensureBuffer(BUFFER_STREAMS, metadata.byteLength, STORAGE | COPY_DST);
    const blockBuffer = ensureBuffer(BUFFER_BLOCKS, blocks.byteLength, STORAGE | COPY_DST);
    const valueBytes = useF16 ? 2 : 4;
    const stateBytes = tokenCount * HIDDEN_SIZE * valueBytes;
    const treeBytes = treeNodes * HIDDEN_SIZE * valueBytes;
    ensureBuffer(BUFFER_LEAF_STATES, stateBytes, STORAGE);
    ensureBuffer(BUFFER_TREE_UP, treeBytes, STORAGE);
    ensureBuffer(BUFFER_SCRATCH,
      Math.max(blockCount * HIDDEN_SIZE * 16, treeNodes * HIDDEN_SIZE * 4), STORAGE);
    ensureBuffer(BUFFER_HYBRID_NEIGHBORS, blockCount * 8, STORAGE);
    ensureBuffer(BUFFER_HYBRID_LOCAL, tokenCount * HIDDEN_SIZE * 4, STORAGE);
    try {
      device.queue.writeBuffer(featureBuffer, 0, features);
      device.queue.writeBuffer(streamBuffer, 0, metadata);
      device.queue.writeBuffer(blockBuffer, 0, blocks);
      scanParams.set([streams.length / 2, tokenCount, blockCount, treeNodes]);
      device.queue.writeBuffer(paramsBuffer, 0, scanParams);
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(labelBuffer, 0, labelBytes);
      const groups = treeBindGroups ??= createTreeBindGroups();
      const pass = encoder.beginComputePass();
      const encode = (pipeline, bindGroup, workgroups) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        const x = Math.min(65_535, workgroups);
        pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
      };
      const counts = [(blockCount + 63) >> 6, (streams.length + 127) >> 7,
        blockCount, streams.length * 2, blockCount];
      counts.forEach((count, index) => encode(pipelines[index], groups[index], count));
      encode(pipelines[PIPELINE_TREE_GLOBAL], groups[PIPELINE_TREE_GLOBAL], streams.length / 2);
      encode(pipelines[6], groups[6], blockCount);
      pass.end();
      encoder.copyBufferToBuffer(labelBuffer, 0, readBuffer, 0, labelBytes);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(1);
      return consume(new Uint32Array(readBuffer.getMappedRange(), 0, labelBytes / 4));
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
    }
  }

  function ensureBuffer(slot, minimumSize, usage) {
    const current = buffers[slot];
    if (current?.[BUFFER_SIZE] >= minimumSize) {
      if (minimumSize > current[BUFFER_BINDING_SIZE]) {
        current[BUFFER_BINDING_SIZE] = minimumSize;
        treeBindGroups = undefined;
      }
      return current[BUFFER_OBJECT];
    }
    current?.[BUFFER_OBJECT].destroy();
    const maximum = device.limits.maxBufferSize;
    if (minimumSize > maximum) {
      throw new RangeError(`WebGPU buffer limit: ${minimumSize} > ${maximum}`);
    }
    const size = growCapacity(minimumSize, maximum);
    const buffer = device.createBuffer({ size, usage });
    buffers[slot] = [buffer, size, minimumSize];
    treeBindGroups = undefined;
    return buffer;
  }

  function createTreeBindGroups() {
    const resource = (slot) => slot === BUFFER_WEIGHTS
      ? { buffer: weightBuffer, size: weightBuffer.size }
      : { buffer: buffers[slot][BUFFER_OBJECT], size: buffers[slot][BUFFER_BINDING_SIZE] };
    const bind = (pipeline, entries) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map(([binding, slot]) => ({ binding, resource: resource(slot) })),
    });
    const entries = [
      [[0, BUFFER_FEATURES], [2, BUFFER_PARAMS], [9, BUFFER_BLOCKS], [12, BUFFER_HYBRID_NEIGHBORS]],
      [[2, BUFFER_PARAMS], [3, BUFFER_STREAMS], [12, BUFFER_HYBRID_NEIGHBORS]],
      [[0, BUFFER_FEATURES], [2, BUFFER_PARAMS], [3, BUFFER_STREAMS], [4, BUFFER_WEIGHTS],
        [9, BUFFER_BLOCKS], [10, BUFFER_HYBRID_LOCAL], [11, BUFFER_SCRATCH],
        [12, BUFFER_HYBRID_NEIGHBORS]],
      [[2, BUFFER_PARAMS], [3, BUFFER_STREAMS], [11, BUFFER_SCRATCH]],
      [[2, BUFFER_PARAMS], [3, BUFFER_STREAMS], [4, BUFFER_WEIGHTS],
        [5, BUFFER_LEAF_STATES], [6, BUFFER_TREE_UP], [9, BUFFER_BLOCKS],
        [10, BUFFER_HYBRID_LOCAL], [11, BUFFER_SCRATCH]],
      [[2, BUFFER_PARAMS], [3, BUFFER_STREAMS], [4, BUFFER_WEIGHTS],
        [6, BUFFER_TREE_UP], [11, BUFFER_SCRATCH]],
      [[0, BUFFER_FEATURES], [1, BUFFER_LABELS], [2, BUFFER_PARAMS], [4, BUFFER_WEIGHTS],
        [3, BUFFER_STREAMS], [5, BUFFER_LEAF_STATES], [9, BUFFER_BLOCKS], [11, BUFFER_SCRATCH]],
    ];
    return entries.map((entry, index) => bind(pipelines[index], entry));
  }

  function parse(source) {
    const prepared = prepareTreeSource(source);
    const [, ranges, , tokenCount] = prepared;
    if (tokenCount === 0) {
      const spans = spansFromRanges(ranges, new Uint32Array());
      releaseTreePrepared(prepared);
      return spans;
    }

    const promise = new Promise((resolve, reject) => queue.push([prepared, resolve, reject]));
    queuedTokens += tokenCount;
    if (queuedTokens >= MAX_BATCH_TOKENS) void flush();
    else schedule();
    return promise;
  }

  function schedule() {
    if (scheduled || flushing) return;
    scheduled = true;
    queueMicrotask(() => void flush());
  }

  async function flush() {
    if (flushing || queue.length === 0) return;
    flushing = true;
    scheduled = false;
    const requests = queue.splice(0);
    queuedTokens = 0;
    const tokenCount = requests.reduce((total, request) => total + request[0][3], 0);
    let features = requests[0][0][0];
    let streams = requests[0][0][2];
    if (requests.length > 1) {
      const featureLength = tokenCount * TREE_FEATURE_STRIDE;
      const streamLength = requests.reduce((total, request) => total + request[0][2].length, 0);
      batchFeatures = growArray(batchFeatures, featureLength);
      batchStreams = growArray(batchStreams, streamLength);
      features = batchFeatures.subarray(0, featureLength);
      streams = batchStreams.subarray(0, streamLength);
      let featureOffset = 0;
      let tokenOffset = 0;
      let streamOffset = 0;
      for (const request of requests) {
        const [data, , requestStreams, requestTokens] = request[0];
        features.set(data, featureOffset);
        for (let stream = 0; stream < requestStreams.length; stream += 2) {
          streams[streamOffset++] = tokenOffset + requestStreams[stream];
          streams[streamOffset++] = requestStreams[stream + 1];
        }
        featureOffset += data.length;
        tokenOffset += requestTokens;
      }
    }

    try {
      await run(features, streams, (labels) => {
        let labelOffset = 0;
        for (const request of requests) {
          const [prepared, resolve] = request;
          const [, ranges, , requestTokens] = prepared;
          resolve(spansFromRanges(ranges, labels, labelOffset));
          labelOffset += requestTokens;
        }
      });
    } catch (error) {
      for (const request of requests) request[2](error);
    } finally {
      for (const request of requests) releaseTreePrepared(request[0]);
      flushing = false;
      if (queue.length > 0) schedule();
    }
  }

  return {
    h: parse, i: initialize, r: run, e: ensureBuffer,
    g: createTreeBindGroups, b: buffers, p: scanParams,
    get d() { return device; },
    get f() { return useF16; },
    get c() { return pipelines; },
    get q() { return treeBindGroups; },
    set q(value) { treeBindGroups = value; },
    get l() { return treeLayout; },
    set l(value) { treeLayout = value; },
  };
}

// Readable access to runtime internals for tests and the benchmark profiler.
// The public build never imports this export, so bundlers remove it entirely.
export function createInspectableRuntime(options = {}) {
  const runtime = createRuntime(
    options.gpu === undefined ? globalThis.navigator?.gpu : options.gpu,
    options.device,
    options.shader,
    options.run,
  );
  return {
    parse: runtime.h,
    initialize: runtime.i,
    run: runtime.r,
    ensureBuffer: runtime.e,
    createTreeBindGroups: runtime.g,
    buffers: runtime.b,
    scanParams: runtime.p,
    get device() { return runtime.d; },
    get useF16() { return runtime.f; },
    get pipelines() { return runtime.c; },
    get treeBindGroups() { return runtime.q; },
    set treeBindGroups(value) { runtime.q = value; },
    get treeLayout() { return runtime.l; },
    set treeLayout(value) { runtime.l = value; },
  };
}

export function createTreeLayout(streams, reuse) {
  if (reuse?.[LAYOUT_SOURCE]?.length === streams.length) {
    let unchanged = true;
    for (let index = 0; index < streams.length; index++) {
      if (reuse[LAYOUT_SOURCE][index] !== streams[index]) { unchanged = false; break; }
    }
    if (unchanged) return reuse;
  }
  const streamCount = streams.length / 2;
  let blockCount = 0;
  let treeNodes = 0;
  for (let stream = 0; stream < streamCount; stream++) {
    const blocks = (streams[stream * 2 + 1] + TREE_BLOCK_SIZE - 1) >> 5;
    const power = nextPowerOfTwo(blocks);
    blockCount += blocks;
    treeNodes += power * 2 - 1;
  }
  const streamStorage = growArray(reuse?.[LAYOUT_STREAM_STORAGE], streamCount * 6);
  const blockStorage = growArray(reuse?.[LAYOUT_BLOCK_STORAGE], blockCount * 4);
  const metadata = streamStorage.subarray(0, streamCount * 6);
  const blocks = blockStorage.subarray(0, blockCount * 4);
  let block = 0;
  let treeOffset = 0;
  for (let stream = 0; stream < streamCount; stream++) {
    const start = streams[stream * 2];
    const tokens = streams[stream * 2 + 1];
    const count = (tokens + TREE_BLOCK_SIZE - 1) >> 5;
    const power = nextPowerOfTwo(count);
    metadata.set([start, tokens, block, count, treeOffset, power], stream * 6);
    for (let offset = 0; offset < tokens; offset += TREE_BLOCK_SIZE) {
      blocks.set([start + offset, Math.min(TREE_BLOCK_SIZE, tokens - offset), stream, offset >> 5], block * 4);
      block += 1;
    }
    treeOffset += power * 2 - 1;
  }
  // Compact internal tuple: backing stores, active metadata, counts and cache key.
  return [streamStorage, blockStorage, metadata, blocks, blockCount, treeNodes, streams.slice()];
}

function growArray(current, minimum) {
  if (current?.length >= minimum) return current;
  let capacity = 16;
  while (capacity < minimum) capacity *= 2;
  return new Uint32Array(capacity);
}

function nextPowerOfTwo(value) {
  return 2 ** (32 - Math.clz32(value - 1));
}

function growCapacity(minimum, maximum) {
  let size = 256;
  while (size < minimum && size <= maximum / 2) size *= 2;
  if (size < minimum) size = Math.ceil(minimum / 4) * 4;
  return size;
}

export function preparePromotedWeights() {
  const result = new Float32Array(WEIGHTS.length);
  let start = 0;
  for (let range = 0; range < WEIGHT_RANGES.length; range += 2) {
    const end = WEIGHT_RANGES[range];
    const scale = WEIGHT_RANGES[range + 1];
    for (let index = start; index < end; index++) {
      const code = WEIGHTS.charCodeAt(index);
      const value = code >= 97 ? code - 71 : code >= 65 ? code - 65 : code >= 48 ? code + 4 : code === 45 ? 62 : 63;
      result[index] = (value & 1 ? -(value + 1) / 2 : value / 2) * scale;
    }
    start = end;
  }
  return result;
}

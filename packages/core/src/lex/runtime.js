// WebGPU inference runtime shared by every lex model. Model-specific pieces
// (tokenizer, WGSL, weights and pipeline steps) are passed in by the model
// entry, so importing one model never bundles another.
//
// Every call made in one microtask turn is coalesced into a single command
// buffer with a single readback: the mapAsync round-trip dominates a call and
// is nearly flat in input size.

import { unpackWeights } from "./weights-codec.js";

const CLASS_NAMES = [
  "plain", "comment", "string", "number", "keyword",
  "type", "function", "constant", "operator",
];

const JOB_BYTES = 16; // Job { token_count, token_offset, stage, stride }
// Must match MAX_JOBS in the shader: a uniform array needs a compile-time size,
// so a larger burst of calls is flushed in chunks.
const MAX_JOBS = 64;
// Scratch regions per job, must match NREG in the shader.
const SCRATCH_REGIONS = 8;
// Wider models push per-workgroup scratch arrays past the 16 KiB default.
const WORKGROUP_STORAGE = 32 * 1024;

const STORAGE = 128;
const COPY_SRC = 4;
const COPY_DST = 8;
const MAP_READ = 1;
const UNIFORM = 64;

export function createParse(model) {
  let runtime;
  return async function parse(code) {
    const tokens = model.tokenize(code);
    if (!tokens.count) return [];
    runtime ??= createRuntime(model).catch((error) => {
      runtime = undefined;
      throw error;
    });
    const classes = await (await runtime)(tokens.packed, tokens.count);
    return toSpans(tokens, classes);
  };
}

// Adjacent tokens of the same class are merged. Whitespace is not classified;
// it joins the span before it (or the first span), so spans cover the source.
function toSpans(tokens, classes) {
  const spans = [];
  let current;
  for (let i = 0; i < tokens.count; i++) {
    if (tokens.kinds[i] === 1 || tokens.kinds[i] === 2) continue;
    const type = CLASS_NAMES[classes[i]] ?? "plain";
    if (current?.type === type) continue;
    const start = current ? tokens.starts[i] : 0;
    if (current) current.end = start;
    spans.push(current = { type, start, end: 0 });
  }
  if (!current) spans.push(current = { type: "plain", start: 0, end: 0 });
  current.end = tokens.ends[tokens.count - 1];
  return spans;
}

async function createRuntime({ shader, meta, pipeline, sym, f16 }) {
  const gpu = globalThis.navigator?.gpu;
  const adapter = await gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU unavailable");
  const requiredLimits = {};
  for (const name of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
    requiredLimits[name] = adapter.limits[name];
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize >= WORKGROUP_STORAGE) {
    requiredLimits.maxComputeWorkgroupStorageSize = WORKGROUP_STORAGE;
  }
  const device = await adapter.requestDevice({ requiredLimits });

  const module = device.createShaderModule({ code: shader });
  const info = await module.getCompilationInfo?.();
  const errors = info?.messages.filter((message) => message.type === "error") ?? [];
  if (errors.length) {
    throw new Error(`WGSL compilation failed: ${errors.map((message) => message.message).join("; ")}`);
  }

  const dim = meta.config.dim;
  const layout = device.createBindGroupLayout({
    entries: ["read-only-storage", "read-only-storage", "read-only-storage",
      "storage", "storage", "storage", "uniform"].map((type, binding) => ({
      binding, visibility: 4 /* COMPUTE */, buffer: { type },
    })),
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelines = await Promise.all(pipeline.map(({ entry }) => device.createComputePipelineAsync({
    layout: pipelineLayout, compute: { module, entryPoint: entry },
  })));

  const { planes, fp } = unpackWeights(sym, f16, meta);
  const buffers = {
    planes: upload(planes),
    fp: upload(fp),
    jobs: device.createBuffer({ size: MAX_JOBS * JOB_BYTES, usage: UNIFORM | COPY_DST }),
  };
  const capacity = { tokens: 0, hidden: 0 };
  let bindGroup;
  let pending = [];
  let inflight = Promise.resolve();

  function upload(data) {
    const buffer = device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4, usage: STORAGE | COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  function ensure(totalTokens, words, hiddenBytes) {
    if (totalTokens > capacity.tokens) {
      const n = Math.max(totalTokens, capacity.tokens * 2, 4096);
      for (const name of ["tokens", "out", "staging"]) buffers[name]?.destroy();
      buffers.tokens = device.createBuffer({ size: n * words * 4, usage: STORAGE | COPY_DST });
      buffers.out = device.createBuffer({ size: n * 4, usage: STORAGE | COPY_SRC | COPY_DST });
      buffers.staging = device.createBuffer({ size: n * 4, usage: MAP_READ | COPY_DST });
      capacity.tokens = n;
      bindGroup = undefined;
    }
    // Every job gets its own hidden-state slot so a batch runs as one dispatch.
    if (hiddenBytes > capacity.hidden) {
      const n = Math.max(hiddenBytes, capacity.hidden * 2, 64 * 1024);
      buffers.hidden?.destroy();
      buffers.scratch?.destroy();
      buffers.hidden = device.createBuffer({ size: n, usage: STORAGE });
      buffers.scratch = device.createBuffer({ size: n * SCRATCH_REGIONS, usage: STORAGE });
      capacity.hidden = n;
      bindGroup = undefined;
    }
    bindGroup ??= device.createBindGroup({
      layout,
      entries: [buffers.tokens, buffers.planes, buffers.fp, buffers.hidden,
        buffers.scratch, buffers.out, buffers.jobs].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
  }

  async function runBatch(jobs) {
    try {
      const words = jobs[0].packed.length / jobs[0].count;
      let total = 0;
      let maxTokens = 0;
      for (const job of jobs) {
        job.offset = total;
        total += job.count;
        maxTokens = Math.max(maxTokens, job.count);
      }
      const stride = maxTokens * dim;
      ensure(total, words, jobs.length * stride * 4);

      const tokenData = new Uint32Array(total * words);
      const jobData = new Uint32Array(MAX_JOBS * JOB_BYTES / 4);
      jobs.forEach((job, index) => {
        tokenData.set(job.packed, job.offset * words);
        jobData.set([job.count, job.offset, 0, stride], index * JOB_BYTES / 4);
      });
      device.queue.writeBuffer(buffers.tokens, 0, tokenData);
      device.queue.writeBuffer(buffers.jobs, 0, jobData);

      const readBytes = total * 4;
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(buffers.out, 0, readBytes);
      // One pass per step: dispatches within a pass are not guaranteed to see
      // each other's storage writes, and the steps are a dependency chain.
      pipeline.forEach((step, index) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipelines[index]);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(step.tile ? Math.ceil(maxTokens / step.tile) : 1, jobs.length);
        pass.end();
      });
      encoder.copyBufferToBuffer(buffers.out, 0, buffers.staging, 0, readBytes);
      device.queue.submit([encoder.finish()]);

      await buffers.staging.mapAsync(MAP_READ, 0, readBytes);
      const classes = new Uint32Array(buffers.staging.getMappedRange(0, readBytes).slice(0));
      buffers.staging.unmap();
      for (const job of jobs) job.resolve(classes.subarray(job.offset, job.offset + job.count));
    } catch (error) {
      for (const job of jobs) job.reject(error);
    }
  }

  function flush() {
    const queued = pending;
    pending = [];
    // Chained, not concurrent: there is one staging buffer to map.
    inflight = inflight.then(async () => {
      for (let i = 0; i < queued.length; i += MAX_JOBS) {
        await runBatch(queued.slice(i, i + MAX_JOBS));
      }
    });
  }

  return (packed, count) => new Promise((resolve, reject) => {
    if (pending.push({ packed, count, resolve, reject }) === 1) queueMicrotask(flush);
  });
}

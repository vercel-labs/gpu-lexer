const MAP_READ = 1;
const COPY_SRC = 4;
const COPY_DST = 8;
const UNIFORM = 64;
const STORAGE = 128;
const BLOCK_LABELS = 1024;
const pipelineCache = new WeakMap();

export const RLE_SHADER = /* wgsl */ `
struct Params { labels: u32, blocks: u32, block_size: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> runs: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

fn label_at(index: u32) -> u32 {
  return (labels[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn count_runs(@builtin(global_invocation_id) id: vec3<u32>) {
  let block = id.x;
  if (block >= params.blocks) { return; }
  let start = block * params.block_size;
  let end = min(start + params.block_size, params.labels);
  var count = 0u;
  var previous = 0xffffffffu;
  for (var index = start; index < end; index++) {
    let label = label_at(index);
    if (index == start || label != previous) { count += 1u; }
    previous = label;
  }
  counts[block] = count;
}

@compute @workgroup_size(64)
fn scatter_runs(@builtin(global_invocation_id) id: vec3<u32>) {
  let block = id.x;
  if (block >= params.blocks) { return; }
  let start = block * params.block_size;
  let end = min(start + params.block_size, params.labels);
  var output = offsets[block];
  var run_start = start;
  var previous = label_at(start);
  for (var index = start + 1u; index < end; index++) {
    let label = label_at(index);
    if (label != previous) {
      runs[output] = vec2<u32>(index, previous);
      output += 1u;
      run_start = index;
      previous = label;
    }
  }
  if (run_start < end) { runs[output] = vec2<u32>(end, previous); }
}
`;

export async function benchmarkLabelReadback(device, labelBuffer, tokenCount) {
  if (!tokenCount) return {
    parity: true,
    direct: { totalMs: 0, bytes: 0 },
    rle: { totalMs: 0, bytes: 0, runs: 0, compressionRatio: 1 },
  };
  const direct = await directReadback(device, labelBuffer, tokenCount);
  const rle = await rleReadback(device, labelBuffer, tokenCount);
  return {
    parity: arraysEqual(unpackLabels(direct.labels, tokenCount), expandRuns(rle.values, tokenCount)),
    direct: { totalMs: direct.totalMs, gpuMs: direct.gpuMs, mapMs: direct.mapMs, bytes: direct.labels.byteLength },
    rle: {
      totalMs: rle.totalMs,
      initializationMs: rle.initializationMs,
      countGpuMs: rle.countGpuMs,
      countMapMs: rle.countMapMs,
      prefixMs: rle.prefixMs,
      compactGpuMs: rle.compactGpuMs,
      compactMapMs: rle.compactMapMs,
      bytes: rle.values.byteLength,
      runs: rle.values.length / 2,
      compressionRatio: rle.values.byteLength / direct.labels.byteLength,
    },
  };
}

async function directReadback(device, labelBuffer, tokenCount) {
  const started = performance.now();
  const bytes = Math.ceil(tokenCount / 4) * 4;
  const read = device.createBuffer({ size: bytes, usage: MAP_READ | COPY_DST });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(labelBuffer, 0, read, 0, bytes);
    const gpuStarted = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - gpuStarted;
    const mapStarted = performance.now();
    await read.mapAsync(1);
    const labels = new Uint32Array(read.getMappedRange(), 0, bytes / 4).slice();
    const mapMs = performance.now() - mapStarted;
    return { labels, totalMs: performance.now() - started, gpuMs, mapMs };
  } finally {
    if (read.mapState === "mapped") read.unmap();
    read.destroy();
  }
}

async function rleReadback(device, labelBuffer, tokenCount) {
  const started = performance.now();
  const blockCount = Math.ceil(tokenCount / BLOCK_LABELS);
  const initStarted = performance.now();
  const [countPipeline, scatterPipeline] = await rlePipelines(device);
  const initializationMs = performance.now() - initStarted;
  const counts = device.createBuffer({ size: blockCount * 4, usage: STORAGE | COPY_SRC });
  const countRead = device.createBuffer({ size: blockCount * 4, usage: MAP_READ | COPY_DST });
  const offsets = device.createBuffer({ size: blockCount * 4, usage: STORAGE | COPY_DST });
  const params = device.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
  let runs;
  let runsRead;
  try {
    device.queue.writeBuffer(params, 0, Uint32Array.of(tokenCount, blockCount, BLOCK_LABELS, 0));
    const countGroup = device.createBindGroup({
      layout: countPipeline.getBindGroupLayout(0),
      entries: [
        binding(0, labelBuffer), binding(1, counts), binding(4, params),
      ],
    });
    const countEncoder = device.createCommandEncoder();
    dispatch(countEncoder, countPipeline, countGroup, blockCount);
    countEncoder.copyBufferToBuffer(counts, 0, countRead, 0, blockCount * 4);
    const countGpuStarted = performance.now();
    device.queue.submit([countEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const countGpuMs = performance.now() - countGpuStarted;
    const countMapStarted = performance.now();
    await countRead.mapAsync(1);
    const blockCounts = new Uint32Array(countRead.getMappedRange(), 0, blockCount);
    const countMapMs = performance.now() - countMapStarted;
    const prefixStarted = performance.now();
    const blockOffsets = new Uint32Array(blockCount);
    let runCount = 0;
    for (let index = 0; index < blockCount; index++) {
      blockOffsets[index] = runCount;
      runCount += blockCounts[index];
    }
    const prefixMs = performance.now() - prefixStarted;
    countRead.unmap();
    const runBytes = runCount * 8;
    runs = device.createBuffer({ size: runBytes, usage: STORAGE | COPY_SRC });
    runsRead = device.createBuffer({ size: runBytes, usage: MAP_READ | COPY_DST });
    device.queue.writeBuffer(offsets, 0, blockOffsets);
    const scatterGroup = device.createBindGroup({
      layout: scatterPipeline.getBindGroupLayout(0),
      entries: [
        binding(0, labelBuffer), binding(2, offsets), binding(3, runs), binding(4, params),
      ],
    });
    const scatterEncoder = device.createCommandEncoder();
    dispatch(scatterEncoder, scatterPipeline, scatterGroup, blockCount);
    scatterEncoder.copyBufferToBuffer(runs, 0, runsRead, 0, runBytes);
    const compactGpuStarted = performance.now();
    device.queue.submit([scatterEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const compactGpuMs = performance.now() - compactGpuStarted;
    const compactMapStarted = performance.now();
    await runsRead.mapAsync(1);
    const values = new Uint32Array(runsRead.getMappedRange(), 0, runCount * 2).slice();
    const compactMapMs = performance.now() - compactMapStarted;
    return {
      values, initializationMs, countGpuMs, countMapMs, prefixMs, compactGpuMs, compactMapMs,
      totalMs: performance.now() - started,
    };
  } finally {
    if (countRead.mapState === "mapped") countRead.unmap();
    if (runsRead?.mapState === "mapped") runsRead.unmap();
    for (const buffer of [counts, countRead, offsets, params, runs, runsRead]) buffer?.destroy();
  }
}

function rlePipelines(device) {
  let pipelines = pipelineCache.get(device);
  if (pipelines) return pipelines;
  const module = device.createShaderModule({ code: RLE_SHADER });
  const create = (entryPoint) => {
    const descriptor = { layout: "auto", compute: { module, entryPoint } };
    return device.createComputePipelineAsync
      ? device.createComputePipelineAsync(descriptor)
      : device.createComputePipeline(descriptor);
  };
  pipelines = Promise.all([create("count_runs"), create("scatter_runs")]);
  pipelineCache.set(device, pipelines);
  return pipelines;
}

function binding(index, buffer) { return { binding: index, resource: { buffer } }; }

function dispatch(encoder, pipeline, bindGroup, count) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
}

export function expandRuns(values, tokenCount) {
  const labels = new Uint32Array(tokenCount);
  let start = 0;
  for (let index = 0; index < values.length; index += 2) {
    const end = values[index];
    labels.fill(values[index + 1], start, end);
    start = end;
  }
  if (start !== tokenCount) throw new Error(`RLE covered ${start} of ${tokenCount} labels`);
  return labels;
}

function arraysEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function unpackLabels(packed, tokenCount) {
  return Uint32Array.from({ length: tokenCount }, (_, index) =>
    packed[index >> 2] >> ((index & 3) * 8) & 255);
}

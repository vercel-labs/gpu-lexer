import {
  BUFFER_BLOCKS, BUFFER_FEATURES, BUFFER_HYBRID_LOCAL, BUFFER_HYBRID_NEIGHBORS, BUFFER_LABELS,
  BUFFER_LEAF_STATES, BUFFER_PARAMS, BUFFER_READBACK, BUFFER_STREAMS,
  BUFFER_SCRATCH, BUFFER_TREE_UP, PIPELINE_TREE_GLOBAL,
  createInspectableRuntime as createRuntime, createTreeLayout,
} from "../../core/src/gpu.js";
import { HIDDEN_SIZE } from "../../core/src/model.runtime.generated.js";
import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { spansFromRanges } from "../../core/src/spans.js";
import { benchmarkLabelReadback } from "./readback.js";
import { TREE_FEATURE_STRIDE } from "../../core/src/tree-features.js";

const MAP_READ = 1;
const COPY_SRC = 4;
const COPY_DST = 8;
const UNIFORM = 64;
const STORAGE = 128;
const QUERY_RESOLVE = 512;
const TREE_PASS_NAMES = [
  "hybrid_neighbor_blocks", "hybrid_neighbor_prefixes", "hybrid_scan_blocks",
  "hybrid_scan_prefixes", "hybrid_mix_tree_up", "tree_global", "tree_down_classify",
];

/** Benchmark-only phase profiler. Nothing in this module is bundled by core. */
export async function profileHighlight(source, {
  runner = createRuntime(),
  target,
  compareReadback = true,
  profileGpuPasses = false,
} = {}) {
  const totalStarted = performance.now();
  let started = performance.now();
  const prepared = prepareTreeSource(source);
  const [features, ranges, streams, tokenCount] = prepared;
  const preparationMs = performance.now() - started;
  try {
    started = performance.now();
    await runner.initialize();
    const initializationMs = performance.now() - started;
    if (!tokenCount) {
      const spans = spansFromRanges(ranges, new Uint32Array());
      return { phases: { preparationMs, initializationMs }, spans };
    }
    const inference = await runProfiledInference(runner, features, streams, profileGpuPasses);
    started = performance.now();
    const spans = spansFromRanges(ranges, inference.labels);
    const reconstructionMs = performance.now() - started;
    started = performance.now();
    if (target) renderSpans(target, source, spans);
    const domMs = performance.now() - started;
    const totalMs = performance.now() - totalStarted;
    // Run experimental comparisons after the primary timing so their extra
    // pipelines and synchronization cannot contaminate the highlight total.
    const readbackComparison = compareReadback
      ? await benchmarkLabelReadback(runner.device, inference.labelBuffer, tokenCount)
      : null;
    return {
      tokens: tokenCount,
      spans: spans.length,
      phases: {
        preparationMs,
        initializationMs,
        bufferSetupMs: inference.bufferSetupMs,
        uploadMs: inference.uploadMs,
        commandEncodingMs: inference.commandEncodingMs,
        gpuMs: inference.gpuMs,
        readbackMs: inference.readbackMs,
        ...(inference.gpuPasses ? { gpuPasses: inference.gpuPasses } : {}),
        reconstructionMs,
        domMs,
        totalMs,
      },
      readbackComparison,
    };
  } finally {
    releaseTreePrepared(prepared);
  }
}

async function runProfiledInference(runner, features, streams, profileGpuPasses) {
  return runProfiledTree(runner, features, streams, profileGpuPasses);
}

async function runProfiledTree(runner, features, streams, profileGpuPasses) {
  const { device } = runner;
  const tokenCount = features.length / TREE_FEATURE_STRIDE;
  const labelBytes = Math.ceil(tokenCount / 4) * 4;
  let started = performance.now();
  const featureBuffer = runner.ensureBuffer(BUFFER_FEATURES, features.byteLength, STORAGE | COPY_DST);
  const labelBuffer = runner.ensureBuffer(BUFFER_LABELS, labelBytes, STORAGE | COPY_SRC | COPY_DST);
  const readBuffer = runner.ensureBuffer(BUFFER_READBACK, labelBytes, MAP_READ | COPY_DST);
  const paramsBuffer = runner.ensureBuffer(BUFFER_PARAMS, 16, UNIFORM | COPY_DST);
  const layout = runner.treeLayout = createTreeLayout(streams, runner.treeLayout);
  const [, , metadata, blocks, blockCount, treeNodes] = layout;
  const streamBuffer = runner.ensureBuffer(BUFFER_STREAMS, metadata.byteLength, STORAGE | COPY_DST);
  const blockBuffer = runner.ensureBuffer(BUFFER_BLOCKS, blocks.byteLength, STORAGE | COPY_DST);
  const valueBytes = runner.useF16 ? 2 : 4;
  const stateBytes = tokenCount * HIDDEN_SIZE * valueBytes;
  const treeBytes = treeNodes * HIDDEN_SIZE * valueBytes;
  runner.ensureBuffer(BUFFER_LEAF_STATES, stateBytes, STORAGE);
  runner.ensureBuffer(BUFFER_TREE_UP, treeBytes, STORAGE);
  runner.ensureBuffer(BUFFER_SCRATCH,
    Math.max(blockCount * HIDDEN_SIZE * 16, treeNodes * HIDDEN_SIZE * 4), STORAGE);
  runner.ensureBuffer(BUFFER_HYBRID_NEIGHBORS, blockCount * 8, STORAGE);
  runner.ensureBuffer(BUFFER_HYBRID_LOCAL, tokenCount * HIDDEN_SIZE * 4, STORAGE);
  const timestampCount = TREE_PASS_NAMES.length * 2;
  const timestampsEnabled = profileGpuPasses && device.features?.has?.("timestamp-query");
  const timestampSet = timestampsEnabled
    ? device.createQuerySet({ type: "timestamp", count: timestampCount })
    : null;
  const timestampResolve = timestampsEnabled
    ? runner.ensureBuffer(13, timestampCount * 8, QUERY_RESOLVE | COPY_SRC)
    : null;
  const timestampReadback = timestampsEnabled
    ? runner.ensureBuffer(14, timestampCount * 8, MAP_READ | COPY_DST)
    : null;
  const bufferSetupMs = performance.now() - started;

  started = performance.now();
  device.queue.writeBuffer(featureBuffer, 0, features);
  device.queue.writeBuffer(streamBuffer, 0, metadata);
  device.queue.writeBuffer(blockBuffer, 0, blocks);
  runner.scanParams.set([streams.length / 2, tokenCount, blockCount, treeNodes]);
  device.queue.writeBuffer(paramsBuffer, 0, runner.scanParams);
  const uploadMs = performance.now() - started;

  started = performance.now();
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(labelBuffer, 0, labelBytes);
  const groups = runner.treeBindGroups ??= runner.createTreeBindGroups();
  const counts = [Math.ceil(blockCount / 64), Math.ceil(streams.length / 128),
    blockCount, streams.length * 2, blockCount];
  const dispatches = counts.map((count, index) => [runner.pipelines[index], groups[index], count]);
  dispatches.push(
    [runner.pipelines[PIPELINE_TREE_GLOBAL], groups[PIPELINE_TREE_GLOBAL], streams.length / 2],
    [runner.pipelines[6], groups[6], blockCount],
  );
  if (timestampSet) {
    dispatches.forEach(([pipeline, group, count], index) =>
      encode(encoder, pipeline, group, count, timestampSet, index));
  } else {
    const pass = encoder.beginComputePass();
    dispatches.forEach(([pipeline, group, count]) => dispatch(pass, pipeline, group, count));
    pass.end();
  }
  encoder.copyBufferToBuffer(labelBuffer, 0, readBuffer, 0, labelBytes);
  if (timestampSet) {
    encoder.resolveQuerySet(timestampSet, 0, timestampCount, timestampResolve, 0);
    encoder.copyBufferToBuffer(timestampResolve, 0, timestampReadback, 0, timestampCount * 8);
  }
  const commands = encoder.finish();
  const commandEncodingMs = performance.now() - started;

  started = performance.now();
  device.queue.submit([commands]);
  await device.queue.onSubmittedWorkDone();
  const gpuMs = performance.now() - started;
  started = performance.now();
  await readBuffer.mapAsync(1);
  const labels = new Uint32Array(readBuffer.getMappedRange(), 0, labelBytes / 4).slice();
  const readbackMs = performance.now() - started;
  readBuffer.unmap();
  let gpuPasses;
  if (timestampReadback) {
    await timestampReadback.mapAsync(1);
    const values = new BigUint64Array(timestampReadback.getMappedRange(), 0, timestampCount);
    gpuPasses = Object.fromEntries(TREE_PASS_NAMES.map((name, index) => [
      name,
      Number(values[index * 2 + 1] - values[index * 2]) / 1e6,
    ]));
    timestampReadback.unmap();
    timestampSet.destroy();
  }
  return { labels, labelBuffer, bufferSetupMs, uploadMs, commandEncodingMs, gpuMs, readbackMs, gpuPasses };
}

function encode(encoder, pipeline, bindGroup, workgroups, timestampSet, passIndex) {
  const pass = encoder.beginComputePass(timestampSet ? {
    timestampWrites: {
      querySet: timestampSet,
      beginningOfPassWriteIndex: passIndex * 2,
      endOfPassWriteIndex: passIndex * 2 + 1,
    },
  } : undefined);
  dispatch(pass, pipeline, bindGroup, workgroups);
  pass.end();
}

function dispatch(pass, pipeline, bindGroup, workgroups) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const x = Math.min(65_535, workgroups);
  pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}

function renderSpans(target, source, spans) {
  const fragment = document.createDocumentFragment();
  for (const span of spans) {
    const value = source.slice(span.start, span.end);
    if (span.type === "plain") fragment.append(document.createTextNode(value));
    else {
      const element = document.createElement("span");
      element.className = `syntax-${span.type}`;
      element.textContent = value;
      fragment.append(element);
    }
  }
  target.replaceChildren(fragment);
}

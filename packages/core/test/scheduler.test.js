import assert from "node:assert/strict";
import test from "node:test";

import { createInspectableRuntime as createRuntime } from "../src/gpu.js";
import { TREE_FEATURE_STRIDE, TREE_FEATURE_VERSION } from "../src/tree-features.js";

const syntax = { number: 3, keyword: 4, operator: 8 };

function createTestRuntime(runOverride) {
  const calls = [];
  const streamsSeen = [];
  const run = runOverride ?? (async (features, streams, consume) => {
    calls.push(features);
    streamsSeen.push(streams);
    const count = features.length / TREE_FEATURE_STRIDE;
    const labels = new Uint32Array(Math.ceil(count / 4));
    for (let index = 0; index < count; index++) {
      const packed = features[index * TREE_FEATURE_STRIDE];
      const kind = packed & 3;
      const first = (packed >>> 5) & 127;
      let label = 0;
      if (first >= 48 && first <= 57) label = syntax.number;
      if (kind === 3) label = syntax.operator;
      labels[index >> 2] |= label << ((index & 3) * 8);
    }
    return consume(labels);
  });
  return { runtime: createRuntime({ run }), calls, streams: streamsSeen };
}

test("concurrent requests are coalesced into one tree runner call", async () => {
  const { runtime, calls, streams } = createTestRuntime();
  await Promise.all([runtime.parse("a + 1"), runtime.parse("b * 2")]);
  assert.equal(calls.length, 1);
  assert.deepEqual([...streams[0]], [0, 5, 5, 5]);
});

test("a single request reaches the runner without copying its feature buffer", async () => {
  const { runtime, calls } = createTestRuntime();
  await runtime.parse("const answer = 42");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].length, 7 * TREE_FEATURE_STRIDE);
});

test("mapped labels are consumed synchronously inside the runner callback", async () => {
  let consumed = false;
  const { runtime } = createTestRuntime(async (_features, _streams, consume) => {
    const result = consume(Uint32Array.of(syntax.keyword));
    consumed = true;
    return result;
  });
  const spans = await runtime.parse("value");
  assert.equal(consumed, true);
  assert.equal(spans[0].type, "keyword");
});

test("parsing reconstructs merged source spans", async () => {
  const source = "value + 42";
  const { runtime } = createTestRuntime();
  const spans = await runtime.parse(source);
  assert.deepEqual(Object.keys(spans[0]), ["type", "start", "end"]);
  assert.equal(spans.map((span) => source.slice(span.start, span.end)).join(""), source);
  assert.deepEqual(spans.map((span) => span.type), ["plain", "operator", "plain", "number"]);
});

test("the public module exports only parse", async () => {
  assert.deepEqual(Object.keys(await import("../src/index.js")), ["parse"]);
});

test("requests queued during a dispatch are flushed next", async () => {
  let releaseFirst;
  const firstRun = new Promise((resolve) => { releaseFirst = resolve; });
  let calls = 0;
  const { runtime } = createTestRuntime(async (features, _streams, consume) => {
    calls += 1;
    if (calls === 1) await firstRun;
    return consume(new Uint32Array(Math.ceil(features.length / TREE_FEATURE_STRIDE / 4)));
  });
  const first = runtime.parse("first");
  await new Promise((resolve) => queueMicrotask(resolve));
  const second = runtime.parse("second");
  releaseFirst();
  await Promise.all([first, second]);
  assert.equal(calls, 2);
});

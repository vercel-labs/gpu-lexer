import assert from "node:assert/strict";
import test from "node:test";

import { createInspectableRuntime as createRuntime } from "../src/inspect.js";
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

test("labels that arrive in chunks rebuild the same spans as labels that arrive whole", async () => {
  const sources = ["let total = 10 + 20 + 30", "x", "a   +   b * 42 - c / 7 + d", "9 9 9 + + + 9"];
  // Every label is derived from the part itself, so each chunk can be labeled alone.
  const labelsFor = (features, from, to) => {
    const word = from & ~3;
    const labels = new Uint32Array(Math.ceil((to - word) / 4));
    for (let part = from; part < to; part++) {
      const packed = features[part * TREE_FEATURE_STRIDE];
      const first = (packed >>> 5) & 127;
      let label = (packed & 3) === 3 ? syntax.operator : first >= 48 && first <= 57 ? syntax.number : 0;
      // Lower the confidence of some parts so that merged spans carry a minimum across chunks.
      label |= (part % 5 === 0 ? 6 : 15) << 4;
      labels[(part - word) >> 2] |= label << (((part - word) & 3) * 8);
    }
    return [labels, -word, from, to];
  };
  const whole = createTestRuntime(async (features, _streams, consume) =>
    consume(labelsFor(features, 0, features.length / TREE_FEATURE_STRIDE)[0]));
  const expected = await Promise.all(sources.map((source) => whole.runtime.parse(source)));
  // Chunk boundaries fall inside requests, inside label words, and inside merged spans.
  for (const size of [1, 3, 5, 6, 11, 1000]) {
    let calls = 0;
    const chunked = createTestRuntime(async (features, _streams, consume) => {
      const count = features.length / TREE_FEATURE_STRIDE;
      for (let from = 0; from < count; from += size) {
        calls += 1;
        consume(...labelsFor(features, from, Math.min(count, from + size)));
      }
    });
    assert.deepEqual(await Promise.all(sources.map((source) => chunked.runtime.parse(source))), expected, `size ${size}`);
    assert.ok(calls >= 1);
  }
  assert.ok(expected[0].some((span) => span.confidence < 1));
});

test("parsing reconstructs merged source spans", async () => {
  const source = "value + 42";
  const { runtime } = createTestRuntime();
  const spans = await runtime.parse(source);
  assert.deepEqual(Object.keys(spans[0]), ["type", "start", "end", "confidence"]);
  assert.equal(spans.map((span) => source.slice(span.start, span.end)).join(""), source);
  assert.deepEqual(spans.map((span) => span.type), ["plain", "operator", "plain", "number"]);
});

test("a span reports the lowest confidence level among its merged parts", async () => {
  // "a b + c": words a, b at levels 15 and 9, operator at 3, word c at 15;
  // whitespace reports the top level and must not lower a span.
  const levels = { a: 15, b: 9, "+": 3, c: 15 };
  const source = "a b + c";
  const { runtime } = createTestRuntime(async (features, _streams, consume) => {
    const count = features.length / TREE_FEATURE_STRIDE;
    const labels = new Uint32Array(Math.ceil(count / 4));
    let offset = 0;
    for (const [index, text] of ["a", " ", "b", " ", "+", " ", "c"].entries()) {
      assert.equal(source.slice(offset, offset + text.length), text);
      offset += text.length;
      const label = (text === "+" ? syntax.operator : 0) | ((levels[text] ?? 15) << 4);
      labels[index >> 2] |= label << ((index & 3) * 8);
    }
    assert.equal(count, 7);
    return consume(labels);
  });
  assert.deepEqual(await runtime.parse(source), [
    { type: "plain", start: 0, end: 4, confidence: 10 / 16 },
    { type: "operator", start: 4, end: 5, confidence: 4 / 16 },
    { type: "plain", start: 5, end: 7, confidence: 1 },
  ]);
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

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createParse } from "../src/lex/runtime.js";
import { tokenize } from "../src/lex/lite/tokenizer.js";
import { META, PIPELINE, WEIGHTS_F16_B85, WEIGHTS_SYM } from "../src/lex/lite/weights.js";
import { CLASS, installFakeGpu, setGpu } from "./fake-gpu.js";

const DIM = 4;
const STEPS = [{ entry: "tiled", tile: 16 }, { entry: "whole", tile: 0 }];

function fakeModel() {
  return createParse({
    tokenize, shader: "", pipeline: STEPS, sym: "", f16: "",
    meta: { config: { dim: DIM }, plane_words: 0, sym_tensors: {}, f16_bytes: 0 },
  });
}

const text = (source, spans) => spans.map(({ type, start, end }) => [type, source.slice(start, end)]);

test("empty input resolves without touching WebGPU", async () => {
  setGpu(undefined);
  assert.deepEqual(await fakeModel()(""), []);
});

test("the runtime fails clearly when WebGPU is unavailable, then retries", async () => {
  setGpu(undefined);
  const parse = fakeModel();
  await assert.rejects(parse("x"), /WebGPU unavailable/);
  setGpu({ async requestAdapter() { return null; } });
  await assert.rejects(parse("x"), /WebGPU unavailable/);
  const log = installFakeGpu();
  assert.deepEqual(text("x", await parse("x")), [["plain", "x"]]);
  assert.equal(log.adapters, 1);
});

test("device initialization opts into supported adapter limits", async () => {
  const log = installFakeGpu();
  await fakeModel()("x");
  assert.deepEqual(log.requestedLimits, {
    maxBufferSize: 4_294_967_292, maxStorageBufferBindingSize: 4_294_967_292,
    maxComputeWorkgroupStorageSize: 32_768,
  });
});

test("the workgroup storage limit is only requested when the adapter supports it", async () => {
  const log = installFakeGpu({ adapterLimits: { maxBufferSize: 1, maxStorageBufferBindingSize: 1,
    maxComputeWorkgroupStorageSize: 16_384 } });
  await fakeModel()("x");
  assert.deepEqual(log.requestedLimits, { maxBufferSize: 1, maxStorageBufferBindingSize: 1 });
});

test("WGSL compilation errors reject with their diagnostics", async () => {
  installFakeGpu({ compileMessages: [{ type: "warning", message: "meh" }, { type: "error", message: "bad token" }] });
  await assert.rejects(fakeModel()("x"), /WGSL compilation failed: bad token/);
});

test("spans merge same-class tokens and cover the source in order", async () => {
  installFakeGpu();
  const parse = fakeModel();
  const source = "  value + 42  \n1 2 x";
  const spans = await parse(source);
  assert.deepEqual(Object.keys(spans[0]), ["type", "start", "end"]);
  assert.deepEqual(text(source, spans), [
    ["plain", "  value "], ["operator", "+ "], ["number", "42  \n1 2 "], ["plain", "x"],
  ]);
  for (let i = 1; i < spans.length; i++) assert.equal(spans[i].start, spans[i - 1].end);
});

test("whitespace-only input is one plain span", async () => {
  installFakeGpu();
  assert.deepEqual(await fakeModel()(" \t\r\n "), [{ type: "plain", start: 0, end: 5 }]);
});

test("unknown class ids fall back to plain", async () => {
  installFakeGpu({ classify: (packed) => new Uint32Array(packed.length / 2).fill(99) });
  assert.deepEqual(await fakeModel()("a b"), [{ type: "plain", start: 0, end: 3 }]);
});

test("concurrent requests share one submit and one readback", async () => {
  const log = installFakeGpu();
  const parse = fakeModel();
  const sources = ["a + 1", "x".repeat(40).split("").join(" "), "7"];
  const results = await Promise.all(sources.map(parse));
  assert.equal(log.submits, 1);
  assert.equal(log.maps, 1);
  assert.deepEqual(text(sources[0], results[0]), [["plain", "a "], ["operator", "+ "], ["number", "1"]]);
  assert.deepEqual(text(sources[2], results[2]), [["number", "7"]]);

  // Tiled steps dispatch one workgroup per tile of the longest job; jobs ride y.
  const longest = tokenize(sources[1]).count;
  assert.deepEqual(log.dispatches, [["tiled", Math.ceil(longest / 16), 3], ["whole", 1, 3]]);
  assert.equal(log.passes, STEPS.length);

  const jobs = log.writes.find(({ size }) => size === 64 * 16);
  const job = new Uint32Array(jobs.bytes.buffer);
  const counts = sources.map((source) => tokenize(source).count);
  assert.deepEqual([...job.subarray(0, 12)], [
    counts[0], 0, 0, longest * DIM,
    counts[1], counts[0], 0, longest * DIM,
    counts[2], counts[0] + counts[1], 0, longest * DIM,
  ]);
});

test("sequential requests each get their own flush", async () => {
  const log = installFakeGpu();
  const parse = fakeModel();
  await parse("a");
  await parse("b");
  assert.equal(log.submits, 2);
});

test("requests queued during a flush run after it", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const log = installFakeGpu();
  const parse = fakeModel();
  await parse("warm");
  let maps = 0;
  for (const buffer of log.buffers) {
    buffer.mapAsync = async function () { if (maps++ === 0) await gate; };
  }
  const first = parse("1");
  await new Promise((resolve) => setTimeout(resolve));
  const second = parse("+");
  await new Promise((resolve) => setTimeout(resolve));
  assert.equal(maps, 1);
  release();
  assert.deepEqual(await first, [{ type: "number", start: 0, end: 1 }]);
  assert.deepEqual(await second, [{ type: "operator", start: 0, end: 1 }]);
  assert.equal(maps, 2);
});

test("more than 64 concurrent requests are split into chunks", async () => {
  const log = installFakeGpu();
  const parse = fakeModel();
  const results = await Promise.all(Array.from({ length: 130 }, (_, i) => parse(String(i))));
  assert.equal(log.submits, 3);
  assert.deepEqual(log.dispatches.filter(([entry]) => entry === "whole").map(([, , jobs]) => jobs), [64, 64, 2]);
  assert.ok(results.every((spans) => spans.length === 1 && spans[0].type === "number"));
});

test("a failed batch rejects its requests and later requests still run", async () => {
  let fail = true;
  installFakeGpu({ classify: (packed) => {
    if (fail) throw new Error("device lost");
    return new Uint32Array(packed.length / 2).fill(CLASS.keyword);
  } });
  const parse = fakeModel();
  const results = await Promise.allSettled([parse("a"), parse("b")]);
  assert.deepEqual(results.map(({ status }) => status), ["rejected", "rejected"]);
  assert.match(results[0].reason.message, /device lost/);
  fail = false;
  assert.deepEqual(await parse("a"), [{ type: "keyword", start: 0, end: 1 }]);
});

test("buffers and the bind group are reused until capacity is exceeded", async () => {
  const log = installFakeGpu();
  const parse = fakeModel();
  await parse("a b c");
  const created = log.buffers.length;
  const groups = log.bindGroups.length;
  await parse("d e f");
  assert.equal(log.buffers.length, created);
  assert.equal(log.bindGroups.length, groups);

  const before = log.buffers.slice();
  await parse("x ".repeat(3000));
  assert.ok(log.buffers.length > created);
  assert.ok(log.bindGroups.length > groups);
  const destroyed = before.filter(({ destroyed }) => destroyed);
  assert.ok(destroyed.length > 0);
  for (const buffer of destroyed) {
    assert.equal(log.bindGroups.at(-1).entries.some(({ resource }) => resource.buffer === buffer), false);
  }
});

test("the lite model wires every pipeline step and uploads its weights", async () => {
  const log = installFakeGpu();
  const shader = await readFile(new URL("../src/lex/lite/shader.wgsl", import.meta.url), "utf8");
  const parse = createParse({
    tokenize, shader, meta: META, pipeline: PIPELINE, sym: WEIGHTS_SYM, f16: WEIGHTS_F16_B85,
  });
  await parse("const answer = 42");
  assert.equal(log.code, shader);
  assert.deepEqual(log.entries, PIPELINE.map(({ entry }) => entry));
  const [planes, fp] = log.writes;
  assert.equal(planes.size, META.plane_words * 4);
  assert.equal(fp.size, Math.ceil(META.f16_count * 4 / 4) * 4);
  assert.equal(log.bindGroups.at(-1).entries.length, 7);
});

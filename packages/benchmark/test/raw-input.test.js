import assert from "node:assert/strict";
import test from "node:test";

import { packUtf16, packUtf8, rangesFromTape, RawInputWorkspace, __testing } from "../src/raw-input.js";
import { expandRuns } from "../src/readback.js";

test("UTF-16 transport packs two exact code units per word", () => {
  const words = packUtf16("A🌍z", new RawInputWorkspace());
  assert.equal(words.length, 2);
  assert.equal(words[0] & 0xffff, 65);
  assert.equal(words[0] >>> 16, "🌍".charCodeAt(0));
  assert.equal(words[1] & 0xffff, "🌍".charCodeAt(1));
  assert.equal(words[1] >>> 16, 122);
});

test("UTF-8 encodeInto workspace reports Unicode edge cases", () => {
  const workspace = new RawInputWorkspace();
  assert.deepEqual([...packUtf8("café", workspace).bytes], [...new TextEncoder().encode("café")]);
  assert.equal(packUtf8("🌍", workspace).hasLoneSurrogate, false);
  assert.equal(packUtf8("\ud800", workspace).hasLoneSurrogate, true);
  assert.equal(__testing.hasAstralCodePoint("🌍"), true);
});

test("prepass tapes reconstruct non-whitespace token ranges", () => {
  const item = (kind, boundary, width = 1) => kind | (Number(boundary) << 8) | (width << 16);
  assert.deepEqual(rangesFromTape(Uint32Array.of(
    item(1, true), item(1, false), item(0, true), item(5, true), item(5, false), item(1, true),
  ), false), [0, 2, 3, 5, 5, 6]);
});

test("block-local RLE expands exactly across duplicate boundary runs", () => {
  assert.deepEqual([...expandRuns(Uint32Array.of(2, 4, 4, 4, 7, 1), 7)], [4, 4, 4, 4, 1, 1, 1]);
});

test("pipeline namespaces reuse one shader module across entry points", async () => {
  let modules = 0;
  let pipelines = 0;
  const device = {
    createShaderModule({ code }) { modules += 1; return { code }; },
    createComputePipeline({ compute }) { pipelines += 1; return compute; },
  };
  const first = await __testing.cachedPipeline(device, 0, "shader", "first");
  const second = await __testing.cachedPipeline(device, 0, "shader", "second");
  assert.equal(await __testing.cachedPipeline(device, 0, "shader", "first"), first);
  assert.notEqual(first, second);
  assert.equal(modules, 1);
  assert.equal(pipelines, 2);
});

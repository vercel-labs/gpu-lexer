import assert from "node:assert/strict";
import test from "node:test";

import { createTreeLayout } from "../src/gpu.js";
import { prepareTreeSource, releaseTreePrepared } from "../src/prepare-tree.js";
import {
  runtimeTreeFeatureLayout, treeFeatureLayout, unreachableTreeFeatureRows,
} from "../src/tree-features.js";

test("simple tree parts keep words, spaces and CRLF but split every symbol", () => {
  const source = "wsl:${fp}\r\n café++";
  const prepared = prepareTreeSource(source);
  try {
    const [, ranges, streams, tokenCount] = prepared;
    const parts = Array.from({ length: tokenCount }, (_, index) =>
      source.slice(ranges[index * 2], ranges[index * 2 + 1]));
    assert.deepEqual(parts,
      ["wsl", ":", "$", "{", "fp", "}", "\r\n", " ", "café", "+", "+"]);
    assert.equal(parts.join(""), source);
    assert.deepEqual([...streams], [0, tokenCount]);
  } finally {
    releaseTreePrepared(prepared);
  }
});

test("non-ASCII spelling is normalized without changing source ranges", () => {
  const first = prepareTreeSource("café");
  const second = prepareTreeSource("caf_");
  try {
    assert.deepEqual(first[0], second[0]);
    assert.deepEqual([...first[1]], [0, 4]);
  } finally {
    releaseTreePrepared(first);
    releaseTreePrepared(second);
  }
});

test("tree layout preserves request boundaries without semantic segmentation", () => {
  const layout = createTreeLayout(Uint32Array.of(0, 300, 300, 10));
  const [, , streams, blocks, count, treeNodes] = layout;
  assert.equal(count, 11);
  assert.equal(treeNodes, 32);
  assert.deepEqual([...streams], [0, 300, 0, 10, 0, 16, 300, 10, 10, 1, 31, 1]);
  assert.deepEqual([...blocks.slice(0, 12)], [0, 32, 0, 0, 32, 32, 0, 1, 64, 32, 0, 2]);
  assert.deepEqual([...blocks.slice(-8)], [288, 12, 0, 9, 300, 10, 1, 0]);
});

test("runtime features omit only unreachable pair sentinel rows", () => {
  const training = treeFeatureLayout();
  const runtime = runtimeTreeFeatureLayout();
  assert.deepEqual(unreachableTreeFeatureRows(), [659, 674, 675, 690, 691, 723]);
  assert.equal(runtime.inputSize, training.inputSize - 6);
  assert.equal(runtime.previousPair, 659);
  assert.equal(runtime.nextPair, 673);
  assert.equal(runtime.previousSymbolPair, 687);
  assert.equal(runtime.nextSymbolPair, 718);
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  TREE_REPLAY_FAMILIES, mineTreeReplay, treeReplayFraction,
} from "../src/tree-replay.js";
import { classNames } from "../src/classes.js";

test("automatic tree replay is capped at one percent", () => {
  assert.equal(treeReplayFraction(undefined, { autoMining: true }), 0.01);
  assert.equal(treeReplayFraction(0.005, { autoMining: true }), 0.005);
  assert.equal(treeReplayFraction(0.008), 0.008);
  assert.equal(treeReplayFraction(0, { autoMining: true }), 0);
  assert.throws(() => treeReplayFraction(0.011, { autoMining: true }), /capped at 0.01/);
  assert.throws(() => treeReplayFraction(0.251), /between 0 and 0.25/);
});

test("tree mining ranks real failures, covers focused families, and weights only failing parts", () => {
  const record = (family, sourceName, predictions) => ({
    family, language: family, sourceName,
    features: predictions.map(() => Uint16Array.of(0)),
    targets: new Uint8Array(predictions.length),
    auxiliary: new Uint16Array(predictions.length),
    supervisionWeights: new Uint8Array(predictions.length).fill(255),
    lossWeights: new Uint8Array(predictions.length).fill(1),
    predictions,
  });
  const records = [
    record("css", "css-repo", [1, 0, 1, 0, 1, 0, 0, 0]),
    record("lua", "lua-repo", [0, 1, 0, 0, 0, 1, 0, 0]),
    record("javascript", "js-repo", [1, 1, 1, 1]),
  ];
  const probability = (prediction) => Float32Array.from({ length: classNames.length }, (_, index) =>
    index === prediction ? 0.9 : 0.1 / (classNames.length - 1));
  const result = mineTreeReplay(records, null, {}, {
    families: TREE_REPLAY_FAMILIES,
    windowParts: 4, strideParts: 2, contextParts: 1,
    maxParts: 8, maxWindowsPerSource: 1, failureWeight: 2,
    predict: ({ predictions }) => predictions.map(probability),
  });

  assert.equal(result.records.length, 2);
  assert.deepEqual(new Set(result.records.map(({ family }) => family)), new Set(["css", "lua"]));
  assert.ok(result.records.every(({ family }) => family !== "javascript"));
  for (const replay of result.records) {
    assert.equal(replay.targets.length, 4);
    assert.ok([...replay.lossWeights].includes(2));
    for (let part = 0; part < replay.lossWeights.length; part++) {
      if (replay.lossWeights[part] === 2) assert.notEqual(replay.predictions[replay.miningWindow.from + part], 0);
    }
  }
  assert.equal(result.stats.selectedParts, 8);
  assert.equal(result.stats.perFamily.css.selectedWindows, 1);
  assert.equal(result.stats.perFamily.lua.selectedWindows, 1);
});

test("tree mining rejects prediction alignment changes", () => {
  const record = {
    family: "markdown", features: [Uint16Array.of(0)], targets: Uint8Array.of(0),
    supervisionWeights: Uint8Array.of(255), lossWeights: Uint8Array.of(1),
  };
  assert.throws(() => mineTreeReplay([record], null, {}, { predict: () => [] }), /length mismatch/);
});

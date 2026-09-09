import assert from "node:assert/strict";
import test from "node:test";
import {
  agreementLanguageMultipliers, countClasses, enrollMatureLanguageGuards, focusedReplayConfig,
  majorLanguageErrorRows,
  shouldAutoPromoteTreeRun, treeTrainingSchedule,
  treeWarmStartMismatch, validateTreeTeacher,
} from "../src/train-tree.js";
import { createTreeModel, treeAuxiliaryNames, treeTensorNamesFor } from "../src/tree-model.js";
import { classNames } from "../src/classes.js";
import { treeTrainingPolicy, assertObjectiveCoverage } from "../src/tree-training-policy.js";
import { experimentPlan } from "../src/tree-experiments.js";

test("policy uses explicit population weights, validates calibration, and refuses partial coverage", async () => {
  const policy = await treeTrainingPolicy({ classWeightPower: 0.25, calibrationEpochs: 2 });
  assert.equal(policy.selectionMetric, "accuracy");
  assert.equal(policy.fixedBaseline, null);
  assert.equal(policy.classWeightPower, 0.25);
  assert.equal((await treeTrainingPolicy()).calibrationEpochs, 2);
  assert.ok(policy.languageObjective.familyWeights.javascript > policy.languageObjective.familyWeights.diff);
  assert.equal((await treeTrainingPolicy({ selectionMetric: "accuracy" })).selectionMetric, "accuracy");
  assert.throws(() => assertObjectiveCoverage([], policy.languageObjective, "test"), /coverage incomplete/);
  await assert.rejects(treeTrainingPolicy({ calibrationEpochs: 5 }), /calibration/);
  await assert.rejects(treeTrainingPolicy({ epochs: 8, fineTuneEpochs: 4, agreementEpochs: 5 }), /agreement/);
  await assert.rejects(treeTrainingPolicy({ classWeightPower: NaN }), /class-weight-power/);
  await assert.rejects(treeTrainingPolicy({ selectionMetric: "loss" }), /selection metric/);
});

test("low verification support is advisory while missing family coverage fails", async () => {
  const { languageObjective } = await treeTrainingPolicy();
  const objective = { ...languageObjective, familyWeights: { diff: 1 }, supplementalWeights: { diff: 1 }, protectedFamilies: ["diff"] };
  const targets = new Uint8Array(299).fill(1);
  targets.fill(0, 0, 62);
  const records = [{ family: "diff", targets, supervisionWeights: new Uint8Array(299).fill(255) }];
  assert.deepEqual(assertObjectiveCoverage(records, objective, "training"), []);
  assert.deepEqual(assertObjectiveCoverage(records, objective, "verification"), ["diff plain support 62 < 100"]);
  assert.equal(objective.minPlainSupport, 100);
  targets.fill(0, 0, 100);
  assert.deepEqual(assertObjectiveCoverage(records, objective, "verification"), []);
});

test("isolated capacity experiments have exact int6 costs", () => {
  assert.deepEqual(experimentPlan().comparisons.map(({ parameters, packedBytes }) => [parameters, packedBytes]),
    [[37513, 28135], [39673, 29755], [41609, 31207], [43769, 32827]]);
});

test("class balance excludes context-only whitespace and unlabeled positions", () => {
  const counts = countClasses([{ targets: Uint8Array.of(0, 0, 1, 2, 0),
    supervisionWeights: Uint8Array.of(0, 255, 128, 255, 0) }]);
  assert.deepEqual([...counts], [1, 1, 1, 0, 0, 0, 0, 0, 0]);
});

test("weak-language curriculum is linear, capped, and leaves guarded languages natural", () => {
  const actual = agreementLanguageMultipliers({ perLanguage: {
    javascript: { support: 100, errors: 20 },
    mature: { support: 100, errors: 5 },
    weak: { support: 100, errors: 50 },
    empty: { support: 0, accuracy: 0.8 },
  } }, ["javascript"]);
  assert.equal(actual.javascript, 1);
  assert.equal(actual.mature, 1);
  assert.equal(actual.weak, 3);
  assert.ok(Math.abs(actual.empty - 2) < 1e-12);
  const targeted = agreementLanguageMultipliers({ perLanguage: {
    weak: { support: 100, errors: 50 }, other: { support: 100, errors: 50 },
  } }, [], 0.9, ["weak"]);
  assert.deepEqual(targeted, { other: 1, weak: 3 });
  assert.doesNotThrow(() => agreementLanguageMultipliers({ perLanguage: {} }, [], 0.9, ["tsx"]));
  assert.throws(() => agreementLanguageMultipliers({ perLanguage: { bad: { support: 1, errors: 2 } } }),
    /invalid baseline/);
});

test("targeted replay options survive tree configuration", () => {
  assert.deepEqual(focusedReplayConfig({ focusedReplay: true, focusedReplaySteps: 2,
    focusedReplayLearningRate: 0.002, focusedReplayFinalLearningRate: 0.0004 }), {
    focusedReplay: true, focusedReplaySteps: 2,
    focusedReplayLearningRate: 0.002, focusedReplayFinalLearningRate: 0.0004,
  });
  assert.deepEqual(focusedReplayConfig(), { focusedReplay: false, focusedReplaySteps: 1,
    focusedReplayLearningRate: 0.001, focusedReplayFinalLearningRate: 0.0005 });
  assert.throws(() => focusedReplayConfig({ focusedReplaySteps: 0 }), /between 1 and 4/);
  assert.throws(() => focusedReplayConfig({ focusedReplayLearningRate: 0.0001,
    focusedReplayFinalLearningRate: 0.001 }), /final must not exceed initial/);
});

test("major languages below ten percent error become persistent one-point guards", async () => {
  const objective = (await treeTrainingPolicy()).languageObjective;
  const verification = { perLanguage: {
    javascript: { support: 1000, errors: 50, plainSupport: 500 },
    jsx: { support: 1000, errors: 99, plainSupport: 500 },
    scss: { support: 1000, errors: 100, plainSupport: 500 },
    svelte: { support: 1000, errors: 20, plainSupport: 50 },
  } };
  const updated = enrollMatureLanguageGuards(objective, verification, ["java"]);
  assert.deepEqual(updated.matureLanguages, ["java", "jsx"]);
  assert.equal(updated.maxMatureErrorIncrease, 0.01);
  assert.equal(updated.maxMatureFalseColorIncrease, 0.01);
});

test("major-language summary reports exact-language hard guards in objective order", () => {
  const rows = majorLanguageErrorRows({ perLanguage: {
    scss: { support: 100, errors: 20 },
    javascript: { support: 200, errors: 10 },
    jsx: { support: 50, errors: 1 },
    ruby: { support: 100, errors: 90 },
  } }, {
    protectedFamilies: ["javascript", "css"],
    strictLanguages: ["javascript", "css"],
    familyWeights: { javascript: 0.8, css: 0.2, ruby: 0 },
  });
  assert.deepEqual(rows, [
    { language: "javascript", family: "javascript", support: 200, errorRate: 0.05, hardGuard: true },
    { language: "jsx", family: "javascript", support: 50, errorRate: 0.02, hardGuard: false },
    { language: "scss", family: "css", support: 100, errorRate: 0.2, hardGuard: false },
  ]);
});

test("default hybrid student shapes fit the existing packed budget", () => {
  for (const [bits, hashBuckets, classifierSize] of [[6, 256, 72], [5, 256, 84], [4, 256, 176]]) {
    const model = createTreeModel({ treeContext: "hybrid", hiddenSize: 32, classifierSize, hashBuckets });
    const count = treeTensorNamesFor(model).reduce((n, name) => n + model[name].length, 0);
    assert.ok(count <= 75000);
    assert.ok(Math.ceil(count * bits / 8) <= 37500);
  }
});

test("tree warm start accepts the promoted shape and explains architecture changes", () => {
  const metadata = {
    model: "hierarchical-tree", formatVersion: 9, labelSource: "shiki-spans-v1",
    featureVersion: 3, tokenizerVersion: 2, hiddenSize: 32,
    classNames, auxiliaryNames: treeAuxiliaryNames,
    config: { treeContext: "hybrid" },
    architecture: { classifierDimensions: 72, lexemeHashBuckets: 256, scaleBuckets: 12 },
  };
  assert.equal(treeWarmStartMismatch(metadata), null);
  assert.match(treeWarmStartMismatch(metadata, { hiddenSize: 64 }), /hidden 32 -> 64/);
  assert.match(treeWarmStartMismatch(metadata, { treeContext: "tree", hashBuckets: 128 }),
    /context hybrid -> tree, hash 256 -> 128/);
});

test("compatible warm starts use a short polish schedule while fresh runs retain full training", () => {
  assert.deepEqual(treeTrainingSchedule({}, true),
    { epochs: 10, fineTuneEpochs: 2, agreementEpochs: 6, calibrationEpochs: 2 });
  assert.deepEqual(treeTrainingSchedule({}, false),
    { epochs: 32, fineTuneEpochs: 4, agreementEpochs: 6, calibrationEpochs: 2 });
  assert.deepEqual(treeTrainingSchedule({}, false, true),
    { epochs: 32, fineTuneEpochs: 4, agreementEpochs: 0, calibrationEpochs: 2 });
  assert.deepEqual(treeTrainingSchedule({ epochs: 10, calibrationEpochs: 3 }, true),
    { epochs: 10, fineTuneEpochs: 3, agreementEpochs: 6, calibrationEpochs: 3 });
});

test("a compatible consistency teacher remains valid when verification data changes", () => {
  const config = {
    labelSource: "shiki-spans-v1", inputSize: 755, hashBuckets: 256, scaleBuckets: 12,
  };
  const model = createTreeModel({ treeContext: "hybrid", hiddenSize: 32, classifierSize: 72, hashBuckets: 256 });
  const teacher = { model, metadata: {
    model: "hierarchical-tree", formatVersion: 9, labelSource: config.labelSource,
    featureVersion: 3, tokenizerVersion: 2, inputSize: config.inputSize,
    classNames, auxiliaryNames: treeAuxiliaryNames,
    architecture: { lexemeHashBuckets: 256, scaleBuckets: 12, tree: "scale-aware-butterfly-binary" },
    corpus: { verification: { sha256: "old-verification-digest" } },
  } };
  assert.doesNotThrow(() => validateTreeTeacher(teacher, config));
  teacher.metadata.architecture.lexemeHashBuckets = 128;
  assert.throws(() => validateTreeTeacher(teacher, config), /model features/);
});

test("tree runs auto-promote only for a strict untouched accuracy improvement", () => {
  const result = (accuracy, baselineAccuracy, overrides = {}) => ({
    selection: { status: "eligible" },
    metadata: {
      verification: { accuracy },
      config: { teacherMode: false, fixedBaselineMetrics: { accuracy: baselineAccuracy } },
    },
    ...overrides,
  });
  assert.equal(shouldAutoPromoteTreeRun(result(0.834, 0.833)), true);
  assert.equal(shouldAutoPromoteTreeRun(result(0.833, 0.833)), false);
  assert.equal(shouldAutoPromoteTreeRun(result(0.832, 0.833)), false);
  assert.equal(shouldAutoPromoteTreeRun(result(0.834, 0.833, { selection: { status: "diagnostic-only" } })), false);
  const teacher = result(0.9, 0.833);
  teacher.metadata.config.teacherMode = true;
  assert.equal(shouldAutoPromoteTreeRun(teacher), false);
  const unpinned = result(0.9, 0.833);
  unpinned.metadata.config.fixedBaselineMetrics = null;
  assert.equal(shouldAutoPromoteTreeRun(unpinned), false);
});

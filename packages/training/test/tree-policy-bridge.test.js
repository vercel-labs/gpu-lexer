import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { runTorchTraining, trainingDatasetPlan } from "../src/torch-runner.js";
import { createTreeModel, treeAuxiliaryNames } from "../src/tree-model.js";
import { classNames } from "../src/classes.js";
import { treeInputSize } from "../../core/src/tree-features.js";
import { treeTrainingPolicy } from "../src/tree-training-policy.js";

test("polish exports the natural corpus without a duplicate fine dataset", () => {
  assert.deepEqual(trainingDatasetPlan({ polishMode: true, epochs: 8, fineTuneEpochs: 2 }),
    { usesNaturalTrain: true, hasFineDataset: false });
  assert.deepEqual(trainingDatasetPlan({ polishMode: false, epochs: 32, fineTuneEpochs: 4 }),
    { usesNaturalTrain: true, hasFineDataset: true });
  assert.deepEqual(trainingDatasetPlan({ polishMode: false, epochs: 6, fineTuneEpochs: 6 }),
    { usesNaturalTrain: false, hasFineDataset: false });
});

for (const fail of [false, true]) test(`staging bridge preserves policy and recovery files (failure=${fail}) without training`, async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "tree-policy-bridge-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const capture = resolve(directory, "captured.json"), script = resolve(directory, "capture.mjs");
  // Node impersonates only the file-based protocol; no optimizer or model loop.
  await writeFile(script, `import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
const directory=process.argv[process.argv.indexOf('--directory')+1];
const config=await readFile(resolve(directory,'config.json'));
await writeFile(${JSON.stringify(capture)},config);
const initial=await readFile(resolve(directory,'initial.f32'));
for(const name of ['selected','best-overall','best-mixed']) await writeFile(resolve(directory,name+'.f32'),initial);
const candidate={epoch:0,metrics:{weightedError:0.2}};
await writeFile(resolve(directory,'result.json'),JSON.stringify({history:[],selected:candidate,bestOverall:candidate,bestMixed:candidate,selection:{status:'diagnostic-only',failures:['css error: regression']}}));
${fail ? 'process.exit(7);' : ''}
`);
  // runTorchTraining invokes executable with trainer path first. Use a tiny
  // executable shim to replace that argument with the protocol fixture.
  const shim = resolve(directory, "protocol");
  await writeFile(shim, `#!/bin/sh\nshift\nexec '${process.execPath.replaceAll("'", "'\\''")}' '${script}' "$@"\n`, { mode: 0o700 });
  const policy = await treeTrainingPolicy({ classWeightPower: .25, calibrationEpochs: 1 });
  const config = { ...policy, model: "hierarchical-tree", treeContext: "hybrid", hiddenSize: 2,
    classifierSize: 2, inputSize: treeInputSize(0), weightBits: 6, epochs: 2, fineTuneEpochs: 1,
    agreementEpochs: 0, agreementLearningRate: 0.00005, agreementFinalLearningRate: 0.00001,
    calibrationLearningRate: 0.000001, calibrationFinalLearningRate: 0.0000002,
    focusedReplay: true, focusedReplaySteps: 1, initialIsBaseline: true,
    focusedReplayLearningRate: 0.001, focusedReplayFinalLearningRate: 0.0005,
    trainingLanguageMultipliers: { javascript: 1, css: 3 },
    fixedBaselineMetrics: { weightedError: .3, perFamily: {} }, migrationExpectedAccuracy: .8,
    repeatHardReplay: true, maxHardReplayRepeats: 2, boundaryLossMultiplier: 1.25 };
  const model = createTreeModel({ ...config, hashBuckets: 0, random: () => .5 });
  const records = [{ features: [Uint16Array.of(0)], targets: Uint8Array.of(0),
    auxiliary: Uint16Array.of(0), supervisionWeights: Uint8Array.of(255), family: "javascript" }];
  let staging;
  t.after(async () => { if (staging) await rm(staging, { recursive: true, force: true }); });
  const operation = runTorchTraining({ runtime: { executable: shim, device: "cpu", torch: "protocol-fixture" }, config, model,
    pretrainRecords: records, fineTuneRecords: records, hardRecords: records, requiredRecords: records,
    verificationRecords: records,
    pretrainClassWeights: classNames.map(() => 1), fineTuneClassWeights: classNames.map(() => 1),
    pretrainAuxiliaryPositiveWeights: treeAuxiliaryNames.map(() => 1), fineTuneAuxiliaryPositiveWeights: treeAuxiliaryNames.map(() => 1),
    classNames, auxiliaryNames: treeAuxiliaryNames, log: (line) => {
      if (line.startsWith("PyTorch recovery files: ")) staging = line.slice("PyTorch recovery files: ".length);
    } });
  if (fail) await assert.rejects(operation, /exit 7; recovery files retained at/);
  else {
    const result = await operation;
    assert.equal(result.recoveryDirectory, staging);
    assert.deepEqual(result.selection, { status: "diagnostic-only", failures: ["css error: regression"] });
  }
  assert.ok((await readFile(resolve(staging, "selected.f32"))).length > 0);
  assert.ok(JSON.parse(await readFile(resolve(staging, "result.json"))).selected);
  const actual = JSON.parse(await readFile(capture, "utf8"));
  for (const key of ["languageObjective", "classWeightPower", "agreementEpochs", "agreementLearningRate",
    "agreementFinalLearningRate", "calibrationEpochs", "calibrationLearningRate",
    "calibrationFinalLearningRate", "fixedBaselineMetrics",
    "migrationExpectedAccuracy", "selectionMetric"]) {
    assert.deepEqual(actual[key], config[key], key);
  }
  assert.equal(actual.repeatHardReplay, true);
  assert.equal(actual.focusedReplay, true);
  assert.equal(actual.focusedReplaySteps, 1);
  assert.equal(actual.focusedReplayLearningRate, 0.001);
  assert.equal(actual.focusedReplayFinalLearningRate, 0.0005);
  assert.equal(actual.initialIsBaseline, true);
  assert.equal(actual.maxHardReplayRepeats, 2);
  assert.equal(actual.boundaryLossMultiplier, 1.25);
  assert.deepEqual(actual.trainingLanguageMultipliers, config.trainingLanguageMultipliers);
  assert.equal(actual.hasHardRecords, true);
  assert.equal(actual.hasRequiredRecords, true);
});

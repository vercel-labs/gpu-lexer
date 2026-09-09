import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { classNames } from "../src/classes.js";
import { addFailureToBank, loadFailureBank, selectDiverseFailures } from "../src/failure-bank.js";
import { focusedFailureRecord, mismatchReport, parseFineTuneArguments } from "../src/fine-tune.js";
import { createTreeModel, createTreeRecord } from "../src/tree-model.js";

test("fine-tune arguments describe one constrained snippet run", () => {
  assert.deepEqual(parseFineTuneArguments([
    "--", "--lang", "javascript", "--file=failure.js", "--epochs", "5",
    "--head-epochs", "2", "--failure-weight", "3",
    "--calibration-epochs", "1", "--failure-steps", "2",
    "--failure-lr", "0.002", "--failure-final-lr", "0.0004",
    "--bank", "failures.json", "--consistency-weight", "0.2",
  ]), {
    lang: "javascript", file: "failure.js", epochs: 5, headTuneEpochs: 2,
    calibrationEpochs: 1, focusedReplaySteps: 2,
    focusedReplayLearningRate: 0.002, focusedReplayFinalLearningRate: 0.0004,
    failureWeight: 3, bank: "failures.json", consistencyWeight: 0.2,
  });
  assert.throws(() => parseFineTuneArguments(["--bogus", "1"]), /unknown option/);
});

test("targeted replay supervises only the active model's failing parts", () => {
  const record = {
    features: [Uint16Array.of(1), Uint16Array.of(2), Uint16Array.of(3)],
    targets: Uint8Array.of(1, 2, 3), auxiliary: Uint16Array.of(0, 1, 0),
    supervisionWeights: Uint8Array.of(255, 128, 255), lossWeights: Uint8Array.of(1, 1, 1),
  };
  const focused = focusedFailureRecord(record, [{ part: 1 }], 3);
  assert.deepEqual([...focused.supervisionWeights], [0, 128, 0]);
  assert.deepEqual([...focused.lossWeights], [1, 3, 1]);
  assert.deepEqual([...record.supervisionWeights], [255, 128, 255]);
  assert.throws(() => focusedFailureRecord(record, [{ part: 3 }]), /invalid failing part/);
});

test("failure bank persists unique snippets and retains family diversity", async (context) => {
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-failure-bank-"));
  const path = resolve(directory, "bank.json");
  context.after(() => rm(directory, { recursive: true, force: true }));
  const entry = (language, family, source) => ({
    language, family, source, sourceFile: `${family}.txt`,
    sourceLabels: [{ from: 0, to: source.length, class: "plain", confidence: 1 }],
  });
  await addFailureToBank(entry("javascript", "javascript", "one"), path);
  await addFailureToBank(entry("javascript", "javascript", "one"), path);
  await addFailureToBank(entry("python", "python", "two"), path);
  await addFailureToBank(entry("javascript", "javascript", "three"), path);
  const bank = await loadFailureBank(path);
  assert.equal(bank.entries.length, 3);
  assert.deepEqual(new Set(bank.entries.map(({ family }) => family)), new Set(["javascript", "python"]));
  assert.deepEqual(new Set(selectDiverseFailures(bank.entries, { maxEntries: 2 }).map(({ family }) => family)),
    new Set(["javascript", "python"]));
  assert.ok(bank.entries.every(({ sourceSha256 }) => /^[a-f0-9]{64}$/.test(sourceSha256)));
});

test("direct Shiki spans become an exact tree record and mismatch report", () => {
  const source = "const answer = 42";
  const keyword = classNames.indexOf("keyword");
  const { record, classCounts } = createTreeRecord({
    source, language: "javascript", family: "javascript", sourceLabelsVersion: 1,
    sourceLabels: [
      { from: 0, to: 5, class: "keyword", confidence: 1 },
      { from: 5, to: source.length, class: "plain", confidence: 1 },
    ],
  }, 0, { retainSource: true });
  const model = createTreeModel({ hiddenSize: 2, classifierSize: 2, hashBuckets: 0, random: () => 0.5 });
  model.outputBias.fill(0);
  model.outputBias[0] = 10;
  const report = mismatchReport(model, record, { hiddenSize: 2, classifierSize: 2 }, source);
  assert.equal(record.source, source);
  assert.equal(classCounts[keyword], 1);
  assert.equal(report.supervised, 4);
  assert.deepEqual(report.mismatches[0], {
    part: 0, from: 0, to: 5, text: "const", expected: "keyword", predicted: "plain",
  });
});

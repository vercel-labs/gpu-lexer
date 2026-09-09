import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { treeFeatureIndices, treeInputSize } from "../../core/src/tree-features.js";
import {
  createTreeModel, loadTreeShard, migrateTreeFeatureV2Model, prepareTreeSourceV2,
  treeAuxiliaryNames, treeProbabilities, treeTensorLayout,
} from "../src/tree-model.js";
import { classNames } from "../src/classes.js";
import { writeTorchDataset } from "../src/torch-data.js";
import { __testing as torchTesting } from "../src/torch-runner.js";

const repositoryRoot = resolve(new URL("../../../", import.meta.url).pathname);
const python = resolve(repositoryRoot, ".venv/bin/python");
const trainer = new URL("../torch/train.py", import.meta.url).pathname;

test("richer hybrid tree remains inside the packed runtime budget", () => {
  const model = createTreeModel({ treeContext: "hybrid", hiddenSize: 32, classifierSize: 72,
    hashBuckets: 256, random: () => 0.5 });
  const layout = treeTensorLayout(model, { inputSize: treeInputSize(256), hiddenSize: 32, classifierSize: 72 });
  const parameters = layout.reduce((sum, tensor) => sum + tensor.length, 0);
  assert.equal(parameters, 41_609);
  assert.ok(parameters < 50_000);
});

test("v3 adds independent word and generic neighboring-symbol hashes", () => {
  const packedWord = 0 | (3 << 2) | (97 << 5) | (42 << 12) | (1 << 19);
  const context = (37 << 15) | (9 << 22) | (17 << 27);
  const data = Uint32Array.of(packedWord, context);
  assert.equal(treeFeatureIndices(data, 0, 64).length, treeFeatureIndices(data, 0, 0).length + 2);
  assert.equal(treeFeatureIndices(data, 0, 64, 2).length,
    treeFeatureIndices(data, 0, 0, 2).length + 1);
  assert.equal(treeInputSize(256), 755);
  assert.equal(treeInputSize(128, 2), 435);
});

test("feature-v2 migration preserves predictions while expanding hashes and scales", () => {
  let state = 29;
  const random = () => { state = Math.imul(state, 1664525) + 1013904223; return (state >>> 0) / 4294967296; };
  const sourceModel = createTreeModel({ treeContext: "hybrid", hiddenSize: 4, classifierSize: 5,
    hashBuckets: 128, featureVersion: 2, scaleBuckets: 8, random });
  const migrated = migrateTreeFeatureV2Model(sourceModel);
  const source = Array.from({ length: 48 }, (_, index) =>
    `const value${index}=thing.member(${index})/* note */;\n`).join("");
  const before = prepareTreeSourceV2(source);
  const after = prepareTreeSource(source);
  try {
    assert.deepEqual(before[1], after[1]);
    const record = (prepared, version, buckets) => ({ features: Array.from(
      { length: prepared[3] }, (_, index) =>
        Uint16Array.from(treeFeatureIndices(prepared[0], index, buckets, version)),
    ) });
    const expected = treeProbabilities(sourceModel, record(before, 2, 128),
      { hiddenSize: 4, classifierSize: 5 });
    const actual = treeProbabilities(migrated, record(after, 3, 256),
      { hiddenSize: 4, classifierSize: 5 });
    for (let part = 0; part < expected.length; part++) for (let label = 0; label < expected[part].length; label++) {
      assert.ok(Math.abs(expected[part][label] - actual[part][label]) < 1e-7,
        `part ${part}, label ${label}`);
    }
  } finally {
    releaseTreePrepared(before);
    releaseTreePrepared(after);
  }
});

test("tree reference returns one normalized distribution per part", () => {
  const model = createTreeModel({ hiddenSize: 2, classifierSize: 3, hashBuckets: 0, random: () => 0.5 });
  const record = { features: [Uint16Array.of(0), Uint16Array.of(3), Uint16Array.of(0)] };
  const probabilities = treeProbabilities(model, record, { hiddenSize: 2, classifierSize: 3 });
  assert.equal(probabilities.length, 3);
  for (const row of probabilities) assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
});

test("tree corpus adds language-neutral nesting targets before closing delimiters", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-tree-nesting-"));
  const path = resolve(directory, "fixture.jsonl.gz");
  const item = {
    source: "{{x}}", language: "fixture", sourceName: "fixture", path: "fixture.txt",
    sourceLabelsVersion: 1,
    sourceLabels: [{ from: 0, to: 5, class: "plain", auxiliary: 0, confidence: 1 }],
  };
  try {
    await writeFile(path, gzipSync(`${JSON.stringify(item)}\n`));
    const { records } = await loadTreeShard(path, Infinity, 0);
    const curlyOne = 1 << 10;
    const curlyTwo = 1 << 11;
    assert.deepEqual([...records[0].auxiliary], [0, curlyOne, curlyOne | curlyTwo, curlyOne | curlyTwo, curlyOne]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("direct tree loading masks uncovered parts", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-direct-tree-"));
  const path = resolve(directory, "fixture.jsonl.gz");
  const item = {
    source: "a+b ?", sourceName: "fixture", path: "fixture.txt",
    sourceLabelsVersion: 1,
    sourceLabels: [
      { from: 0, to: 1, class: "string", confidence: 1 },
      { from: 1, to: 2, class: "operator", confidence: 1 },
      { from: 2, to: 3, class: "keyword", confidence: 1 },
    ],
  };
  try {
    await writeFile(path, gzipSync(`${JSON.stringify(item)}\n`));
    const data = await loadTreeShard(path, Infinity, 0, { requireSourceLabels: true });
    assert.deepEqual([...data.records[0].targets], [2, 8, 4, 0, 0]);
    assert.deepEqual([...data.records[0].supervisionWeights], [255, 255, 255, 0, 0]);
    assert.equal(data.classCounts[0], 0);
    delete item.sourceLabels;
    await writeFile(path, gzipSync(`${JSON.stringify(item)}\n`));
    await assert.rejects(loadTreeShard(path, Infinity, 0, { requireSourceLabels: true }), /missing sourceLabels/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("tree loading skips only explicitly excluded families", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-tree-excluded-family-"));
  const path = resolve(directory, "fixture.jsonl.gz");
  const item = (family) => ({
    source: "x", language: family, family, sourceName: "fixture", path: `${family}.txt`,
    sourceLabelsVersion: 1, sourceLabels: [{ from: 0, to: 1, class: "plain", confidence: 1 }],
  });
  try {
    await writeFile(path, gzipSync(`${JSON.stringify(item("lisp"))}\n${JSON.stringify(item("javascript"))}\n`));
    const data = await loadTreeShard(path, Infinity, 0,
      { requireSourceLabels: true, excludeFamilies: ["lisp"] });
    assert.equal(data.records.length, 1);
    assert.equal(data.records[0].family, "javascript");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("tree loading excludes website verification sources by exact origin and path", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-tree-excluded-source-"));
  const path = resolve(directory, "fixture.jsonl.gz");
  const item = (name) => ({
    source: "x", language: "javascript", family: "javascript", origin: "git",
    sourceName: "owner/repo", path: `${name}.js`, sourceLabelsVersion: 1,
    sourceLabels: [{ from: 0, to: 1, class: "plain", confidence: 1 }],
  });
  try {
    await writeFile(path, gzipSync(`${JSON.stringify(item("demo"))}\n${JSON.stringify(item("train"))}\n`));
    const data = await loadTreeShard(path, Infinity, 0, {
      requireSourceLabels: true,
      excludeSources: new Set(["git\0owner/repo\0demo.js"]),
    });
    assert.equal(data.records.length, 1);
    assert.equal(data.records[0].path, "train.js");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

for (const treeContext of ["tree", "hybrid"]) test(`PyTorch ${treeContext} matches the JavaScript reference`, async (context) => {
  try { await access(python); }
  catch { context.skip("optional PyTorch environment is not installed"); return; }
  const config = { model: "hierarchical-tree", treeContext, localRadius: 2, inputSize: treeInputSize(0), hiddenSize: 4, classifierSize: 4, weightBits: 6 };
  let state = 17;
  const random = () => { state = Math.imul(state, 1664525) + 1013904223; return (state >>> 0) / 4294967296; };
  const model = createTreeModel({ treeContext, hiddenSize: 4, classifierSize: 4, hashBuckets: 0, random });
  const teacherConfig = { model: "hierarchical-tree", inputSize: treeInputSize(0), hiddenSize: 8, classifierSize: 6 };
  const teacher = createTreeModel({ hiddenSize: 8, classifierSize: 6, hashBuckets: 0, random });
  const record = {
    features: [Uint16Array.of(0, 5), Uint16Array.of(3, 19), Uint16Array.of(0, 7, 140)],
    targets: Uint8Array.of(4, 8, 0), auxiliary: Uint16Array.of(1, 4, 0),
    supervisionWeights: Uint8Array.of(255, 255, 255), lossWeights: Uint8Array.of(1, 1, 1),
    language: "fixture", family: "fixture",
  };
  const layout = torchTesting.createTensorLayout(model, config, classNames.length, treeAuxiliaryNames.length);
  const teacherLayout = torchTesting.createTensorLayout(
    teacher, teacherConfig, classNames.length, treeAuxiliaryNames.length,
  );
  const directory = await mkdtemp(resolve(tmpdir(), "gpu-lexer-tree-test-"));
  try {
    await writeTorchDataset(directory, "train", [record]);
    await writeFile(resolve(directory, "initial.f32"), torchTesting.concatenateModel(model, layout));
    await writeFile(resolve(directory, "teacher.f32"), torchTesting.concatenateModel(teacher, teacherLayout));
    await writeFile(resolve(directory, "config.json"), JSON.stringify({
      ...config, device: "cpu", tensorLayout: layout,
      languageObjective: { familyWeights: { fixture: 1 } },
      classWeights: new Array(classNames.length).fill(1),
      auxiliaryPositiveWeights: new Array(treeAuxiliaryNames.length).fill(1), auxiliaryLossWeight: 0.15,
      hasTeacher: true, teacherModel: "hierarchical-tree", teacherWeightBits: 6,
      teacherTensorLayout: teacherLayout, distillationWeight: 0.25, distillationTemperature: 2,
    }));
    const result = spawnSync(python, [trainer, "--directory", directory, "--check"], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const actual = JSON.parse(await readFile(resolve(directory, "check.json"), "utf8")).probabilities[0];
    const expected = treeProbabilities(model, record, config);
    for (let token = 0; token < expected.length; token++) for (let output = 0; output < expected[token].length; output++) {
      assert.ok(Math.abs(actual[token][output] - expected[token][output]) < 2e-6,
        `token ${token} class ${output}: PyTorch ${actual[token][output]}, JavaScript ${expected[token][output]}`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

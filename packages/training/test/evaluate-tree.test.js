import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { classNames } from "../src/classes.js";
import { defaults, evaluateTree, parseArguments } from "../src/evaluate-tree.js";
import { quantizeTensors } from "../src/quantization.js";
import {
  createTreeModel, treeTensorLayout, treeTensorNamesFor,
  TREE_FEATURE_VERSION, TREE_FORMAT_VERSION, TREE_HYBRID_FORMAT_VERSION, TREE_TOKENIZER_VERSION,
} from "../src/tree-model.js";
import { treeInputSize } from "../../core/src/tree-features.js";

const cli = new URL("../src/evaluate-tree.js", import.meta.url).pathname;

async function fixture(t, { teacher = false, hybrid = false, packedClass = "plain" } = {}) {
  const directory = await mkdtemp(join(tmpdir(), "gpu-lexer-evaluate-tree-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const config = { inputSize: treeInputSize(0), hiddenSize: 2, classifierSize: 3, hashBuckets: 0 };
  const model = createTreeModel({ ...config, treeContext: hybrid ? "hybrid" : "tree", random: () => 0.5 });
  model.outputBias[classNames.indexOf("keyword")] = 2;
  const names = treeTensorNamesFor(model);
  const layout = treeTensorLayout(model, config);
  const values = new Float32Array(layout.reduce((sum, tensor) => sum + tensor.length, 0));
  for (const tensor of layout) values.set(model[tensor.name], tensor.offset);
  // Deliberately different stored predictions prove the evaluator reads the
  // deployed bytes instead of silently re-quantizing the float checkpoint.
  const deployed = Object.fromEntries(names.map((name) => [name, model[name].slice()]));
  deployed.outputBias.fill(0);
  deployed.outputBias[classNames.indexOf(packedClass)] = 2;
  const packed = teacher ? null : quantizeTensors(deployed, names, 6);
  const metadata = {
    model: "hierarchical-tree", runId: "fixture",
    formatVersion: hybrid ? TREE_HYBRID_FORMAT_VERSION : TREE_FORMAT_VERSION,
    featureVersion: TREE_FEATURE_VERSION, tokenizerVersion: TREE_TOKENIZER_VERSION,
    labelSource: "shiki-spans-v1",
    precision: teacher ? "float32" : "int6", config: { teacherMode: teacher },
    inputSize: config.inputSize, hiddenSize: config.hiddenSize,
    architecture: { classifierDimensions: config.classifierSize, lexemeHashBuckets: 0 },
    classNames, tensorLayout: layout, quantization: packed?.metadata ?? null,
  };
  const item = {
    source: "if x", language: "javascript", path: "fixture.js", sourceName: "fixture",
    sourceLabelsVersion: 1, sourceLabels: [
      { from: 0, to: 2, class: "keyword", auxiliary: 1 << 9, confidence: 1 },
      { from: 2, to: 3, class: "plain", confidence: 1 },
      { from: 3, to: 4, class: "plain", confidence: 1 },
    ],
  };
  const shard = join(directory, "fixture.jsonl.gz");
  await writeFile(join(directory, "model-fixture.json"), JSON.stringify(metadata));
  await writeFile(join(directory, "weights-f32-fixture.bin"), new Uint8Array(values.buffer));
  if (packed) await writeFile(join(directory, "weights-int6-fixture.bin"), packed.data);
  await writeFile(shard, gzipSync(`${JSON.stringify(item)}\n${JSON.stringify({ ...item, path: "second.js" })}\n`));
  return { directory, shard, metadata, options: { run: directory, shard } };
}

test("CLI options are bounded and reject malformed/unbounded limits", () => {
  assert.equal(defaults.maxParts, 10_000);
  assert.deepEqual(parseArguments(["--", "run-name", "--max-parts=4", "--max-files", "2"]), {
    run: "run-name", maxParts: 4, maxFiles: 2,
  });
  for (const value of ["0", "-1", "Infinity", "NaN", "1.5", "12garbage", "9007199254740992"]) {
    assert.throws(() => parseArguments(["--max-parts", value]), /positive finite safe integer/);
  }
  assert.throws(() => parseArguments(["--max-files"]), /requires a value/);
  assert.throws(() => parseArguments(["--train"]), /unknown option/);
});

test("CLI help performs no checkpoint or training work", () => {
  const result = spawnSync(process.execPath, [cli, "--help"], { encoding: "utf8", timeout: 10_000 });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /NOT a full-corpus score/);
  assert.match(result.stdout, /default 10000/);
  const invalid = spawnSync(process.execPath, [cli, "--max-parts", "Infinity"], { encoding: "utf8", timeout: 10_000 });
  assert.equal(invalid.status, 1);
  assert.match(invalid.stderr, /positive finite safe integer/);
});

test("float and stored quantized predictions score the same bounded raw-label sample, read-only", async (t) => {
  const { directory, options } = await fixture(t);
  const before = await snapshot(directory);
  const report = await evaluateTree({ ...options, maxFiles: 1 });
  assert.deepEqual(await snapshot(directory), before);
  assert.equal(report.sample.files, 1);
  assert.equal(report.sample.parts, 3);
  assert.equal(report.sample.isFullCorpusScore, false);
  assert.equal(report.sample.labelSource, "direct-raw-shiki-sourceLabels");
  assert.equal(report.float.parts.total, 2);
  assert.equal(report.quantized.parts.total, 2);
  assert.equal(report.float.characters.total, 3);
  assert.equal(report.float.characters.correct, 2);
  assert.equal(report.quantized.characters.correct, 1);
  assert.equal(report.float.characters.perClass.find((row) => row.class === "keyword").support, 2);
  assert.equal(report.float.byRegion["clause-region"].characters.correct, 2);
  assert.equal(report.quantized.byRegion["clause-region"].characters.errors, 2);
  assert.equal(report.float.byLanguage.javascript.parts.total, 2);
  assert.equal(report.float.spans.expected, 1);
  assert.equal(report.float.spans.matched, 1);
  assert.equal(report.float.spans.predicted, 2);
  assert.equal(report.quantized.spans.predicted, 0);
  assert.equal(report.quantizationGap.disagreements, 2);
  assert.equal(report.quantizationGap.floatOnlyCorrect, 1);
  assert.equal(report.quantizationGap.quantizedOnlyCorrect, 1);
  assert.ok(Math.abs(report.quantizationGap.characterAccuracyPoints - 100 / 3) < 1e-10);
});

test("part budget clips raw spans and scores no source after the common prefix", async (t) => {
  const { options } = await fixture(t);
  const report = await evaluateTree({ ...options, maxParts: 1 });
  assert.equal(report.sample.parts, 1);
  assert.equal(report.sample.files, 1);
  assert.equal(report.sample.truncatedFiles, 1);
  assert.equal(report.sample.records[0].evaluatedSourceEnd, 2);
  assert.equal(report.float.characters.total, 2);
  assert.equal(report.float.characters.accuracy, 1);
  assert.equal(report.quantized.characters.accuracy, 0);
  assert.equal(report.quantizationGap.characterAccuracyPoints, 100);
});

test("hybrid float teacher is evaluated without any packed artifact or quantization", async (t) => {
  const { directory, options, metadata } = await fixture(t, { teacher: true, hybrid: true });
  // Even stale packed metadata must not turn an offline teacher into a deployed score.
  metadata.quantization = { bits: 6, parameterCount: 999 };
  await writeFile(join(directory, "model-fixture.json"), JSON.stringify(metadata));
  const report = await evaluateTree({ ...options, maxFiles: 1 });
  assert.equal(report.context, "hybrid");
  assert.equal(report.mode, "float-teacher-only");
  assert.equal(report.quantized, null);
  assert.equal(report.quantizationGap, null);
  assert.equal(report.float.characters.total, 3);
});

test("missing/corrupt packed artifacts fail instead of re-quantizing floats", async (t) => {
  const { directory, options } = await fixture(t);
  const path = join(directory, "weights-int6-fixture.bin");
  await writeFile(path, new Uint8Array(1));
  await assert.rejects(evaluateTree(options), /byte length/);
  await rm(path);
  await assert.rejects(evaluateTree(options), /exactly one stored int6 artifact/);
});

test("incompatible feature versions are rejected", async (t) => {
  const { directory, options, metadata } = await fixture(t, { teacher: true });
  metadata.featureVersion = -1;
  await writeFile(join(directory, "model-fixture.json"), JSON.stringify(metadata));
  await assert.rejects(evaluateTree(options), /incompatible/);
});

async function snapshot(directory) {
  return Promise.all((await readdir(directory)).sort().map(async (name) => [name, (await readFile(join(directory, name))).toString("base64")]));
}

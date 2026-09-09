import assert from "node:assert/strict";
import test from "node:test";
import { readFile, mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { auditLabelRecord, auditRepresentativeFiles, representativeManifest } from "../src/source-label-baseline.js";
import { auditSourceLabels } from "../src/source-label-audit.js";
import { createLabeler, sourceLabelsFromTokens } from "../src/label.js";
import { teacherLanguages } from "../src/corpus.js";
import { alignTreeLabels } from "../src/tree-label-alignment.js";
import { detectSyntaxConstructs } from "../src/syntax-constructs.js";

const span = (from, to, name) => ({ from, to, class: name });

test("representative command reports bounded direct alignment diagnostics, not model scores", async () => {
  const report = await auditRepresentativeFiles();
  assert.equal(report.ok, true);
  assert.equal(report.isModelScore, false);
  assert.equal(report.isFullCorpusScore, false);
  assert.equal(report.groups.direct.length, 5);
  assert.deepEqual(report.skipped, []);
  assert.deepEqual(report.malformed, []);
  assert.deepEqual(report.incompleteLabels, []);
  assert.match(report.teacher.version, /^4\./);
  const batch = report.groups.direct.find((record) => record.language === "bat");
  assert.equal(batch.directTreeProjection.characters.accuracy, 1);
  for (const record of report.groups.direct) {
    assert.equal(record.labelSource, "shiki-source");
    assert.match(record.sha256, /^[a-f0-9]{64}$/);
    assert.equal(record.directTreeProjection.teacherCoverage.coverage, 1);
    assert.ok(record.directTreeProjection.characters.total > 0);
    assert.equal(record.plainBaseline.characters.predictionCoverage, 1);
  }
});

test("representative teachers preserve selected semantic spans across Batch ShaderLab Make and embedded languages", async () => {
  const entries = JSON.parse(await readFile(representativeManifest, "utf8"));
  const labeler = await createLabeler({ langs: teacherLanguages });
  const probes = {
    bat: [["representative comment", "comment"], ["echo", "keyword"], ["world", "string"]],
    shaderlab: [["Shader", "keyword"], ["Custom/Example", "string"], ["embedded comment", "comment"], ["return", "keyword"], ["2.0", "number"]],
    make: [["representative comment", "comment"], ["all", "function"], ["$(CC)", "string"]],
    markdown: [["Representative heading", "keyword"], ["const", "keyword", 16], ["42", "number", 16], ["print", "function", 16], ["world", "string", 16]],
    html: [["div", "type", 8], ["const", "keyword", 16], ["call", "function", 16], ["42", "number", 16], ["color", "type", 64]],
  };
  try {
    for (const entry of entries) {
      const source = await readFile(resolve(dirname(representativeManifest), entry.path), "utf8");
      // Original-source offsets also survive Windows line endings and astral prefix.
      for (const text of [source, source.replaceAll("\n", "\r\n")]) {
        const labels = labeler.labelSource(text, entry.language);
        for (const [word, expected, auxiliary] of probes[entry.language]) {
          const from = text.indexOf(word);
          assert.ok(from >= 0);
          const [actual] = alignTreeLabels(labels, [from, from + word.length]);
          assert.equal(actual.class, expected, `${entry.language}: ${word}`);
          assert.equal(actual.coverage, 1);
          if (auxiliary) assert.ok(actual.auxiliary & auxiliary, `${entry.language}: ${word} scope`);
        }
      }
    }
  } finally { labeler.dispose(); }
});

test("missing and malformed explanations never silently become high-confidence plain labels", () => {
  for (const explanation of [undefined, [], {}, [{ content: "x" }], [{ content: "x", scopes: [null] }]]) {
    assert.throws(() => sourceLabelsFromTokens("x", [[{ offset: 0, content: "x", explanation }]]), /explanation/);
  }
  assert.throws(() => sourceLabelsFromTokens("x", [[null]]), /malformed/);
  assert.throws(() => sourceLabelsFromTokens("x", [[{ offset: 0, content: 1 }]]), /malformed/);
});

test("unlabeled non-whitespace is visible even when accuracy on the surviving labels is perfect", () => {
  const result = auditSourceLabels("a x\r\n😀", [span(0, 1, "keyword")], [span(0, 1, "keyword")]);
  assert.equal(result.characters.accuracy, 1);
  assert.deepEqual(result.teacherCoverage, { eligible: 4, labeled: 1, unlabeled: 3, coverage: 0.25, uncoveredRanges: [{ from: 2, to: 3 }, { from: 5, to: 7 }] });
});

test("missing or malformed direct labels fail", () => {
  const old = { source: "foo" };
  assert.throws(() => auditLabelRecord(old), /missing sourceLabels/);
  assert.throws(() => auditLabelRecord({ ...old, sourceLabels: [null] }), /sourceLabels/);
});

test("explicit-file audit reports skipped files and malformed labels without contaminating scores", async () => {
  const dir = await mkdtemp(`${tmpdir()}/source-label-audit-`);
  try {
    const manifest = `${dir}/manifest.json`;
    await writeFile(`${dir}/valid.js`, "const x = 1;\n");
    await writeFile(manifest, JSON.stringify([
      { path: "valid.js", language: "javascript" },
      { path: "missing.js", language: "javascript" },
      { path: "valid.js", language: "not-a-grammar" },
      { language: "javascript" },
    ]));
    const report = await auditRepresentativeFiles({ manifest });
    assert.equal(report.ok, false);
    assert.equal(report.groups.direct.length, 1);
    assert.equal(report.skipped.length, 1);
    assert.equal(report.malformed.length, 2);
    const bounded = await auditRepresentativeFiles({ manifest, maxFiles: 1 });
    assert.equal(bounded.skipped.length, 3);
    assert.ok(bounded.skipped.every(({ reason }) => reason === "max-files"));
    const oversized = await auditRepresentativeFiles({ manifest, maxBytes: 1 });
    assert.equal(oversized.groups.direct.length, 0);
    assert.equal(oversized.skipped[0].reason, "max-bytes");
    await assert.rejects(auditRepresentativeFiles({ maxFiles: NaN }), /positive integer/);
  } finally { await rm(dir, { recursive: true }); }
});

test("construct metadata recognizes fenced Markdown, ShaderLab and Batch continuations", () => {
  for (const source of ["```javascript\nconst n = 1;\n```", "~~~python\nprint(7)\n~~~"]) {
    assert.ok(detectSyntaxConstructs(source, "markdown").includes("embedded-language"));
    assert.ok(!detectSyntaxConstructs(source, "plain").includes("embedded-language"));
  }
  assert.ok(!detectSyntaxConstructs("```\nplain\n```", "markdown").includes("embedded-language"));
  for (const source of ["CGPROGRAM\nfloat x;\nENDCG", "HLSLINCLUDE\nfloat x;\nENDHLSL"]) {
    assert.ok(detectSyntaxConstructs(source, "shaderlab").includes("embedded-language"));
  }
  assert.ok(detectSyntaxConstructs("echo hello ^\r\n world", "bat").includes("line-continuation"));
});

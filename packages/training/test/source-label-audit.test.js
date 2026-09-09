import assert from "node:assert/strict";
import test from "node:test";

import { auditCorpusLabels, auditSourceLabels } from "../src/source-label-audit.js";

const span = (from, to, name) => ({ from, to, class: name });

test("common character/span diagnostics are invariant to tokenizer segmentation", () => {
  const source = "foo+42";
  const teacher = [span(0, 3, "function"), span(3, 4, "operator"), span(4, 6, "number")];
  const split = [span(0, 1, "function"), span(1, 3, "function"), span(3, 4, "operator"), span(4, 5, "number"), span(5, 6, "number")];
  const direct = auditSourceLabels(source, teacher, teacher);
  assert.deepEqual(auditSourceLabels(source, teacher, split), direct);
  assert.equal(direct.characters.accuracy, 1);
  assert.equal(direct.characters.total, 6);
  assert.equal(direct.spans.matched, 3);
  assert.equal(direct.spans.f1, 1);
  assert.equal(direct.boundary.f1, 1);
});

test("character confusion, false-color, exact spans and boundary errors expose different failures", () => {
  const result = auditSourceLabels("abcdef", [span(0, 2, "plain"), span(2, 6, "string")], [span(0, 1, "keyword"), span(1, 3, "plain"), span(3, 6, "string")]);
  assert.equal(result.characters.total, 6);
  assert.equal(result.characters.correct, 4);
  assert.equal(result.characters.falseColorRate, 0.5);
  assert.deepEqual(result.characters.majorityBaseline, { class: "string", accuracy: 4 / 6 });
  assert.equal(result.characters.confusion[0][4], 1);
  assert.equal(result.characters.perClass[2].f1, 6 / 7);
  assert.equal(result.spans.matched, 0);
  assert.equal(result.spans.expected, 1);
  assert.equal(result.spans.predicted, 2);
  assert.equal(result.boundary.matched, 0);
  assert.equal(result.boundary.expected, 1);
  assert.equal(result.boundary.predicted, 2);
});

test("CRLF, surrogate pairs, unlabeled gaps and whitespace have explicit common-grid semantics", () => {
  const source = "😀\r\nx y";
  const teacher = [span(0, 2, "string"), span(4, 7, "comment")];
  const result = auditSourceLabels(source, teacher, teacher);
  assert.equal(result.coordinateUnit, "utf16-code-unit");
  assert.equal(result.characters.total, 4);
  assert.equal(result.spans.expected, 3); // whitespace breaks maximal scored runs
  assert.equal(result.boundary.expected, 0); // no synthetic transition across CRLF gap
  assert.equal(auditSourceLabels(source, teacher, teacher, { includeWhitespace: true }).characters.total, 5);
  const missing = auditSourceLabels(source, teacher, []);
  assert.equal(missing.characters.predictionCoverage, 0);
  assert.equal(missing.characters.accuracy, 0);
  assert.equal(missing.spans.f1, 0);
  assert.equal(auditSourceLabels("", [], []).characters.accuracy, 0);
});

test("audits require direct labels and reject overlapping predictions", () => {
  const old = { source: "foo" };
  assert.throws(() => auditCorpusLabels(old, []), /missing sourceLabels/);
  assert.throws(() => auditSourceLabels("foo", [span(0, 3, "plain")],
    [span(0, 2, "plain"), span(1, 3, "plain")]), /non-overlapping/);
});

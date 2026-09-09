import assert from "node:assert/strict";
import test from "node:test";

import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { createLabeler, sourceLabelsFromTokens, sourceParts } from "../src/label.js";
import { alignTreeLabels, alignCorpusTreeLabels, clipSourceLabels, corpusSourceLabels } from "../src/tree-label-alignment.js";

const span = (from, to, name, confidence = 1, auxiliary = 0) => ({ from, to, class: name, confidence, auxiliary });

test("tree alignment sums class overlaps, unions bits, attenuates confidence and handles gaps", () => {
  const labels = [span(0, 2, "string", 1, 2), span(2, 5, "plain", 1, 16), span(5, 7, "string", 0.5, 2)];
  const [whole, gap, partial] = alignTreeLabels(labels, Uint32Array.of(0, 7, 7, 9, 5, 9));
  assert.equal(whole.class, "string"); // Two string spans beat the largest individual plain span.
  assert.equal(whole.auxiliary, 18);
  assert.equal(whole.confidence, Math.round(255 * 3 / 7));
  assert.equal(whole.coverage, 1);
  assert.equal(gap.class, "plain");
  assert.equal(gap.confidence, 0);
  assert.equal(gap.coverage, 0);
  assert.equal(partial.class, "string");
  assert.equal(partial.coverage, 0.5);
  assert.equal(partial.confidence, 64);
});

test("ties use source order and overlapping/unsorted tree ranges are independent", () => {
  const labels = [span(0, 2, "keyword"), span(2, 4, "string")];
  const result = alignTreeLabels(labels, [{ from: 2, to: 4 }, { from: 0, to: 4 }, { from: 1, to: 3 }]);
  assert.deepEqual(result.map((label) => label.class), ["string", "keyword", "keyword"]);
  assert.deepEqual(result.map((label) => label.confidence), [255, 128, 128]);
});

test("malformed source overlaps, offsets, classes, and confidence fail clearly", () => {
  assert.throws(() => alignTreeLabels([span(0, 3, "plain"), span(2, 4, "string")], [0, 4]), /non-overlapping/);
  assert.throws(() => alignTreeLabels([span(1, 3, "plain"), span(0, 1, "plain")], [0, 4]), /sorted/);
  assert.throws(() => alignTreeLabels([span(0, 3, "plain", 255)], [0, 4]), /confidence/);
  assert.throws(() => alignTreeLabels([span(0, 3, "bogus")], [0, 4]), /unknown/);
  assert.throws(() => alignTreeLabels([], [0]), /pairs/);
  assert.throws(() => alignTreeLabels([], [0, 0]), /invalid tree/);
  assert.throws(() => alignTreeLabels([span(0, 4, "plain")], [0, 3], { sourceLength: 3 }), /within the source/);
});

test("corpus serialization clips teacher spans after source selection without changing originals", () => {
  const labels = [span(0, 2, "keyword"), span(2, 8, "string"), span(8, 9, "operator")];
  assert.deepEqual(clipSourceLabels(labels, 5), [labels[0], span(2, 5, "string")]);
  assert.equal(labels[1].to, 8);
  assert.deepEqual(clipSourceLabels(labels, 0), []);
  const item = JSON.parse(JSON.stringify({ source: "abcde", sourceLabelsVersion: 1, sourceLabels: clipSourceLabels(labels, 5) }));
  assert.equal(alignCorpusTreeLabels(item, [2, 5]).labels[0].class, "string");
});

test("corpus shards require direct source labels", () => {
  const old = { path: "old.js", source: "foo" };
  assert.throws(() => alignCorpusTreeLabels(old, [0, 3]), /old.js.*missing sourceLabels/);
  const fresh = { ...old, sourceLabelsVersion: 1, sourceLabels: [span(0, 3, "function")] };
  const direct = alignCorpusTreeLabels(fresh, [0, 3]);
  assert.equal(direct.labelSource, "shiki-source");
  assert.equal(direct.labels[0].class, "function");
  assert.throws(() => corpusSourceLabels({ ...fresh, sourceLabelsVersion: 2 }), /unsupported/);
  assert.throws(() => corpusSourceLabels({ ...fresh, sourceLabels: "bad" }), /array/);
});

test("Shiki extraction validates explanation partitions and uses original offsets", () => {
  const scopes = [{ scopeName: "keyword.control.js" }];
  assert.deepEqual(sourceLabelsFromTokens("\r\nif", [[{ offset: 2, content: "if", explanation: [{ content: "if", scopes }] }]]), [span(2, 4, "keyword")]);
  assert.throws(() => sourceLabelsFromTokens("foo", [[{ offset: 0, content: "foo", explanation: [{ content: "f", scopes }] }]]), /partition/);
  assert.throws(() => sourceLabelsFromTokens("foo", [[{ offset: 1, content: "foo" }]]), /offset mismatch/);
});

test("real Shiki spans directly label runtime parts, comments, strings, interpolation and markup", async () => {
  const labeler = await createLabeler({ langs: ["javascript", "html"] });
  try {
    const source = '// 😀 comment\r\n\r\nconst text = "hello";\r\nconst out = `hi ${call(42)}`;\n';
    const sourceLabels = labeler.labelSource(source, "javascript");
    assert.ok(sourceParts(source, sourceLabels).length > 0);
    const prepared = prepareTreeSource(source);
    try {
      const direct = alignTreeLabels(sourceLabels, prepared[1], { sourceLength: source.length });
      const at = (text) => direct.find((label) => label.from === source.indexOf(text));
      assert.equal(at("comment").class, "comment");
      assert.equal(at("const").class, "keyword");
      assert.equal(at("hello").class, "string");
      assert.equal(at("hi").class, "string");
      assert.equal(at("${").class, "operator");
      assert.equal(at("call").class, "function");
      assert.equal(at("call").auxiliary & 2, 0);
      assert.ok(at("call").auxiliary & 16);
      assert.equal(at("42").class, "number");
      assert.equal(at("}").class, "operator");
      assert.equal(at("\r\n").confidence, 0);
      assert.ok(sourceLabels.every((label) => !source.slice(label.from, label.to).includes("\r\n")));
    } finally { releaseTreePrepared(prepared); }
    const html = '<!-- note -->\r\n<div title="hello"><script>const x = 42;</script></div>';
    const spans = labeler.labelSource(html, "html");
    const word = (value) => {
      const from = html.indexOf(value);
      return alignTreeLabels(spans, [from, from + value.length])[0];
    };
    assert.equal(word("note").class, "comment");
    assert.equal(word("div").class, "type");
    assert.ok(word("div").auxiliary & 8);
    assert.equal(word("hello").class, "string");
    assert.equal(word("const").class, "keyword");
    assert.equal(word("42").class, "number");
    assert.deepEqual(sourceParts("", labeler.labelSource("", "javascript")), []);
  } finally { labeler.dispose(); }
});

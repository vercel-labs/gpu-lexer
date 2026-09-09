import assert from "node:assert/strict";
import test from "node:test";

import {
  auxiliaryFromScopes, classFromScopes, confidenceFromScopes, normalizeScopes,
} from "../src/classes.js";
import { sourceParts } from "../src/label.js";

test("common TextMate scopes collapse to the small taxonomy", () => {
  assert.equal(classFromScopes(["source.ts", "comment.line.double-slash.ts"]), "comment");
  assert.equal(classFromScopes(["entity.name.function.ts"]), "function");
  assert.equal(classFromScopes(["variable.other.readwrite.ts"]), "plain");
  assert.equal(classFromScopes(["meta.function.js", "variable.other.readwrite.js"]), "plain");
  assert.equal(classFromScopes(["keyword.operator.assignment.js"]), "operator");
  assert.equal(classFromScopes(["variable.other.constant.js"]), "plain");
  assert.equal(classFromScopes(["constant.numeric.decimal.js"]), "number");
  assert.equal(classFromScopes(["entity.name.tag.html"]), "type");
  assert.equal(classFromScopes(["markup.heading.markdown"]), "keyword");
  assert.equal(classFromScopes(["markup.inline.raw.string.markdown"]), "string");
  assert.equal(classFromScopes(["support.constant.property-value.css"]), "constant");
});

test("auxiliary scope labels retain lexical and CSS region distinctions", () => {
  const comment = auxiliaryFromScopes(["comment.block.css"]);
  const selector = auxiliaryFromScopes(["meta.selector.css", "entity.other.attribute-name.class.css"]);
  const property = auxiliaryFromScopes(["meta.property-list.css", "support.type.property-name.css"]);
  assert.ok(comment & 1);
  assert.ok(selector & (1 << 5));
  assert.ok(property & (1 << 6));
});

test("auxiliary labels cover generic markup, values, members, and clauses", () => {
  assert.ok(auxiliaryFromScopes(["entity.other.attribute-name.html"]) & (1 << 3));
  assert.ok(auxiliaryFromScopes(["meta.property-value.css"]) & (1 << 7));
  assert.ok(auxiliaryFromScopes(["variable.other.property.ts"]) & (1 << 8));
  assert.ok(auxiliaryFromScopes(["meta.where.sql"]) & (1 << 9));
});

test("scope normalization makes labels stable and conflicting scopes less trusted", () => {
  assert.deepEqual(normalizeScopes([" Keyword.Control.JS ", "keyword.control.js"]), ["keyword.control.js"]);
  assert.equal(classFromScopes([" KEYWORD.CONTROL.JS "]), "keyword");
  assert.equal(confidenceFromScopes(["keyword.control.js"]), 1);
  assert.equal(classFromScopes(["keyword.operator.assignment.js"]), "operator");
  assert.equal(classFromScopes(["keyword.operator"]), "operator");
  assert.equal(confidenceFromScopes(["keyword.operator.assignment.js"]), 1);
  assert.equal(confidenceFromScopes(["keyword.operator"]), 1);
  assert.equal(confidenceFromScopes(["keyword.control.js", "string.quoted.js"]), 0.5);
  assert.equal(confidenceFromScopes(["invalid.illegal.js"]), 0.25);
});

test("template expressions do not inherit their outer string class", () => {
  const expression = ["string.template.js", "meta.template.expression.js", "meta.embedded.line.js", "variable.other.readwrite.js"];
  const boundary = ["string.template.js", "meta.template.expression.js", "punctuation.definition.template-expression.begin.js"];
  assert.equal(classFromScopes(expression), "plain");
  assert.equal(classFromScopes(boundary), "operator");
  assert.equal(auxiliaryFromScopes(expression) & (1 << 1), 0);
  assert.ok(auxiliaryFromScopes(expression) & (1 << 4));
});

test("Shiki-like spans align to runtime part offsets", () => {
  const code = "const answer = 42";
  const labels = sourceParts(code, [
    { from: 0, to: 5, class: "keyword" },
    { from: 6, to: 12, class: "plain" },
    { from: 15, to: 17, class: "number" },
  ]);
  assert.deepEqual(labels.map((token) => token.class), ["keyword", "plain", "plain", "number"]);
  assert.deepEqual(labels.map(({ confidence }) => confidence), [255, 255, 0, 255]);
});

test("non-ASCII whitespace-like code units follow runtime placeholder boundaries", () => {
  const code = "\ufeffconst";
  const labels = sourceParts(code, [
    { from: 0, to: 1, class: "plain" },
    { from: 1, to: 6, class: "keyword" },
  ]);
  assert.deepEqual(labels.map(({ value, from, to, class: name }) => [value, from, to, name]), [
    ["\ufeffconst", 0, 6, "keyword"],
  ]);
});

test("mixed Shiki labels down-weight ambiguous runtime parts", () => {
  const [label] = sourceParts("mixed", [
    { from: 0, to: 3, class: "keyword", confidence: 1 },
    { from: 3, to: 5, class: "plain", confidence: 1 },
  ]);
  assert.equal(label.class, "keyword");
  assert.equal(label.confidence, 153);
});

test("template-expression labels align to split interpolation tokens", () => {
  const code = "`wsl:${fp}`";
  const labels = sourceParts(code, [
    { from: 0, to: 5, class: "string" },
    { from: 5, to: 7, class: "operator" },
    { from: 7, to: 9, class: "plain" },
    { from: 9, to: 10, class: "operator" },
    { from: 10, to: 11, class: "string" },
  ]);
  assert.deepEqual(labels.map(({ value, class: name }) => [value, name]), [
    ["`", "string"], ["wsl", "string"], [":", "string"], ["$", "operator"],
    ["{", "operator"], ["fp", "plain"], ["}", "operator"], ["`", "string"],
  ]);
});

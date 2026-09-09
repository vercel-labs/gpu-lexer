import assert from "node:assert/strict";
import test from "node:test";

import { labelsFromHighlightedHtml } from "../src/html-labels.js";

test("highlight HTML is reconstructed and collapsed into visual classes", () => {
  const source = 'const x = "hi" < 2';
  const html = '<span class="token keyword">const</span> x <span class="token operator">=</span> ' +
    '<span class="token string">&quot;hi&quot;</span> <span class="token operator">&lt;</span> ' +
    '<span class="token number">2</span>';
  assert.deepEqual(labelsFromHighlightedHtml(source, html, "prism"), [
    { from: 0, to: 5, class: "keyword", confidence: 1 },
    { from: 5, to: 8, class: "plain", confidence: 1 },
    { from: 8, to: 9, class: "operator", confidence: 1 },
    { from: 9, to: 10, class: "plain", confidence: 1 },
    { from: 10, to: 14, class: "string", confidence: 1 },
    { from: 14, to: 15, class: "plain", confidence: 1 },
    { from: 15, to: 16, class: "operator", confidence: 1 },
    { from: 16, to: 17, class: "plain", confidence: 1 },
    { from: 17, to: 18, class: "number", confidence: 1 },
  ]);
});

test("inner plain tokens override styled wrappers", () => {
  const source = "`value ${name}`";
  const html = '<span class="token template-string">`value <span class="token interpolation">' +
    '<span class="token interpolation-punctuation punctuation">${</span>name' +
    '<span class="token interpolation-punctuation punctuation">}</span></span>`</span>';
  const labels = labelsFromHighlightedHtml(source, html, "prism");
  assert.deepEqual(labels.map(({ class: name }) => name), [
    "string", "operator", "plain", "operator", "string",
  ]);
});

test("library-specific names collapse to equivalent Markdown classes", () => {
  assert.deepEqual(labelsFromHighlightedHtml(
    "> quote\n# heading\n`code`",
    '<span class="hljs-quote">&gt; quote</span>\n' +
      '<span class="hljs-section"># heading</span>\n<span class="hljs-code">`code`</span>',
    "highlight.js",
  ).map(({ class: name }) => name), ["plain", "keyword", "plain", "string"]);
});

test("overloaded CSS token names use CSS semantics", () => {
  assert.deepEqual(labelsFromHighlightedHtml(
    'a { color: url("x") }',
    '<span class="hljs-selector-tag">a</span> { <span class="hljs-attribute">color</span>: ' +
      '<span class="hljs-built_in">url</span>(<span class="hljs-string">&quot;x&quot;</span>) }',
    "highlight.js",
    { family: "css" },
  ).map(({ class: name }) => name), ["type", "plain", "type", "plain", "function", "plain", "string", "plain"]);
  assert.equal(labelsFromHighlightedHtml(
    "color",
    '<span class="token property">color</span>',
    "prism.js",
    { family: "css" },
  )[0].class, "type");
  assert.equal(labelsFromHighlightedHtml(
    "color",
    '<span class="sh__token--property">color</span>',
    "sugar-high",
    { family: "css" },
  )[0].class, "type");
});

test("symbols and markup containers follow the shared Shiki taxonomy", () => {
  assert.equal(labelsFromHighlightedHtml(
    ":ready", '<span class="token symbol">:ready</span>', "prism.js",
  )[0].class, "string");
  assert.equal(labelsFromHighlightedHtml(
    "&amp;", '<span class="hljs-symbol">&amp;amp;</span>', "highlight.js", { family: "html" },
  )[0].class, "plain");
  assert.equal(labelsFromHighlightedHtml(
    "null", '<span class="sh__token--class">null</span>', "sugar-high",
  )[0].class, "constant");
});

test("Starry Night PrettyLights classes collapse to the shared taxonomy", () => {
  const source = 'const answer = "yes" + 42 // ok';
  const html = '<span class="pl-k">const</span> answer <span class="pl-kos">=</span> ' +
    '<span class="pl-s">&quot;yes&quot;</span> <span class="pl-kos">+</span> ' +
    '<span class="pl-c1">42</span> <span class="pl-c">// ok</span>';
  assert.deepEqual(labelsFromHighlightedHtml(source, html, "starry-night")
    .map(({ class: name }) => name),
  ["keyword", "plain", "operator", "plain", "string", "plain", "operator", "plain", "number", "plain", "comment"]);
});

test("source divergence is rejected", () => {
  assert.throws(() => labelsFromHighlightedHtml("source", "different", "highlight.js"), /diverged/);
});

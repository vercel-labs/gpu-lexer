import assert from "node:assert/strict";
import test from "node:test";

import { tokenize } from "../src/lex/lite/tokenizer.js";

const parts = (source) => {
  const tokens = tokenize(source);
  return Array.from({ length: tokens.count }, (_, i) => source.slice(tokens.starts[i], tokens.ends[i]));
};

const field = (tokens, i) => {
  const [a, b] = [tokens.packed[i * 2], tokens.packed[i * 2 + 1]];
  return {
    kind: a & 3, length: (a >>> 2) & 7, first: (a >>> 5) & 127, last: (a >>> 12) & 127,
    flags: (a >>> 19) & 255, symbolNext: a >>> 27,
    hash1: b & 1023, hash2: (b >>> 10) & 127, transitionPrevious: (b >>> 17) & 15,
    transitionNext: (b >>> 21) & 15, symbolPrevious: b >>> 25,
  };
};

test("words, space runs and CRLF stay whole while every symbol splits", () => {
  const source = "wsl:${fp}\r\n\t café++";
  assert.deepEqual(parts(source),
    ["wsl", ":", "$", "{", "fp", "}", "\r\n", "\t ", "café", "+", "+"]);
  assert.equal(parts(source).join(""), source);
  assert.deepEqual([...tokenize(source).kinds], [0, 3, 3, 3, 0, 3, 2, 1, 0, 3, 3]);
});

test("empty input has no tokens", () => {
  const tokens = tokenize("");
  assert.equal(tokens.count, 0);
  assert.equal(tokens.packed.length, 0);
});

test("each token packs into two words", () => {
  const tokens = tokenize("Foo_1 BAR\n  \\//x");
  assert.equal(tokens.packed.length, tokens.count * 2);

  const foo = field(tokens, 0);
  assert.deepEqual([foo.kind, foo.length, foo.first, foo.last], [0, 2, 70, 49]);
  // line start | lower | upper | digit | underscore
  assert.equal(foo.flags, 16 | 1 | 2 | 4 | 8);

  // All-caps words set the constant-case flag.
  assert.equal(field(tokens, 2).flags & 128, 128);

  const indent = field(tokens, 4);
  assert.equal(indent.kind, 1);
  assert.equal(indent.flags & 16, 16);

  const backslash = field(tokens, 5);
  assert.equal(backslash.flags & 64, 64);

  // The "//" transition is recorded on both sides of the pair.
  assert.equal(field(tokens, 6).transitionNext, 1);
  assert.equal(field(tokens, 7).transitionPrevious, 1);
  assert.ok(field(tokens, 6).symbolNext > 0);
  assert.equal(field(tokens, 6).symbolNext, field(tokens, 7).symbolPrevious);
});

test("word hashes depend only on spelling, non-ASCII folds to one bucket", () => {
  const a = tokenize("café");
  const b = tokenize("cafÿ");
  assert.deepEqual([...a.packed], [...b.packed]);
  assert.notDeepEqual([...tokenize("cafe").packed], [...a.packed]);
  const repeated = tokenize("name name");
  assert.equal(field(repeated, 0).hash1, field(repeated, 2).hash1);
  assert.equal(field(repeated, 0).hash2, field(repeated, 2).hash2);
});

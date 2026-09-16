import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { initialize, minify, validate } from "wgslender";

import { decodeBase85, halfToFloat, unpackWeights } from "../src/lex/weights-codec.js";
import { META, PIPELINE, WEIGHTS_F16_B85, WEIGHTS_SYM } from "../src/lex/lite/weights.js";

const shader = await readFile(new URL("../src/lex/lite/shader.wgsl", import.meta.url), "utf8");
const constant = (name) => Number(shader.match(new RegExp(`const ${name}: u32 = (\\d+)u;`))[1]);

test("base85 decodes full and short groups", () => {
  assert.deepEqual([...decodeBase85("")], []);
  // Python: base64.b85encode(bytes([222, 173, 190, 239])) == b'-mSjx'
  assert.deepEqual([...decodeBase85("-mSjx")], [222, 173, 190, 239]);
  // Python: base64.b85encode(bytes([0, 1, 2, 3, 255])) == b'009C6{{'
  assert.deepEqual([...decodeBase85("009C6{{")], [0, 1, 2, 3, 255]);
});

test("half floats decode zero, signs, subnormals and infinity", () => {
  const values = halfToFloat(Uint16Array.of(0x0000, 0x8000, 0x3c00, 0xc000, 0x3555, 0x0001, 0x7c00));
  assert.equal(Object.is(values[1], -0), true);
  assert.deepEqual([...values.subarray(2, 4)], [1, -2]);
  assert.equal(values[4], Math.fround(0.33325195));
  assert.equal(values[5], Math.fround(2 ** -14 / 1024));
  assert.equal(values[6], Infinity);
});

test("the lite weights unpack to the declared shapes with finite values", () => {
  const { planes, fp } = unpackWeights(WEIGHTS_SYM, WEIGHTS_F16_B85, META);
  assert.equal(planes.length, META.plane_words);
  assert.equal(fp.length, META.f16_count);
  assert.equal(fp.every(Number.isFinite), true);
  assert.ok(planes.some((word) => word !== 0));
  for (const [name, tensor] of Object.entries(META.sym_tensors)) {
    assert.ok(tensor.plane_offset + tensor.bits * tensor.words <= META.plane_words, name);
    assert.ok(tensor.offset + tensor.words * 32 <= WEIGHTS_SYM.length, name);
  }
});

test("the lite shader matches the model and runtime layout", () => {
  assert.equal(constant("DIM"), META.config.dim);
  assert.equal(constant("NCLASS"), 9);
  // Runtime constants in src/lex/runtime.js.
  assert.equal(constant("MAX_JOBS"), 64);
  assert.equal(constant("NREG"), 8);
  const bindings = [...shader.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map((match) => Number(match[1]));
  assert.deepEqual(bindings, [0, 1, 2, 3, 4, 5, 6]);
  for (const { entry, tile } of PIPELINE) {
    assert.match(shader, new RegExp(`fn ${entry}\\(`));
    assert.ok(tile === 0 || tile === constant("TILE") || tile === constant("TILE_POOL"), entry);
  }
});

test("the lite shader validates before and after build minification", async () => {
  await initialize();
  const source = validate(shader);
  assert.equal(source.valid, true, source.diagnostics.map(({ message }) => message).join("; "));
  const { code, errors } = minify(shader, {
    minifyWhitespace: true, minifyIdentifiers: true, minifySyntax: true, treeShaking: true,
  });
  assert.deepEqual(errors, []);
  assert.equal(validate(code).valid, true);
  for (const { entry } of PIPELINE) assert.match(code, new RegExp(`fn ${entry}\\(`));
});

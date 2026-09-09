import assert from "node:assert/strict";
import test from "node:test";

import { dequantizeTensors, encodeWeightSymbols, packSigned, quantizeTensors } from "../src/quantization.js";

test("signed int6 values pack four values into three bytes", () => {
  const values = Int8Array.from([-31, -1, 0, 1, 30, 31, -16, 16]);
  const packed = packSigned(values, 6);

  assert.equal(packed.length, 6);
  assert.deepEqual(unpack(packed, values.length), [...values]);
});

test("int6 quantization records parameter and packed byte counts separately", () => {
  const quantized = quantizeTensors({ value: Float32Array.from([-1, 0, 1, 0.5, -0.5]) }, ["value"]);

  assert.equal(quantized.parameterCount, 5);
  assert.equal(quantized.data.length, 4);
  assert.equal(quantized.metadata.parameterCount, 5);
  assert.equal(quantized.metadata.packedByteLength, 4);
  assert.deepEqual(unpack(quantized.data, 5), [-31, 0, 31, 16, -15]);
  assert.deepEqual(
    [...dequantizeTensors(quantized.data, quantized.metadata).value].map((value) => Number(value.toFixed(3))),
    [-1, 0, 1, 0.516, -0.484],
  );
});

function unpack(bytes, length) {
  return Array.from({ length }, (_, index) => {
    const bitOffset = index * 6;
    const byteOffset = bitOffset >> 3;
    const shift = bitOffset & 7;
    const encoded = ((bytes[byteOffset] | ((bytes[byteOffset + 1] ?? 0) << 8)) >> shift) & 63;
    return encoded & 32 ? encoded - 64 : encoded;
  });
}

for (const bits of [4, 5, 6]) test(`signed int${bits} round-trips every deployed level`, () => {
  const level = 2 ** (bits - 1) - 1;
  const values = Int8Array.from({ length: level * 2 + 1 }, (_, index) => index - level);
  const packed = packSigned(values, bits);
  const quantization = {
    bits, parameterCount: values.length,
    tensors: [{ name: "value", offset: 0, length: values.length, scale: 1 }],
  };
  assert.deepEqual([...dequantizeTensors(packed, quantization).value], [...values]);
  assert.equal(encodeWeightSymbols(packed, values.length, bits).length, values.length);
});

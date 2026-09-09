export const RUNTIME_WEIGHT_BITS = 6;
export const WEIGHT_SYMBOL_ENCODING = "signed-symbol-v1";
const WEIGHT_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

export function quantizeTensors(model, names, bits = RUNTIME_WEIGHT_BITS) {
  const levels = 2 ** (bits - 1) - 1;
  const parameterCount = names.reduce((sum, name) => sum + model[name].length, 0);
  const values = new Int8Array(parameterCount);
  const tensors = [];
  let offset = 0;

  for (const name of names) {
    const tensorValues = model[name];
    let maximum = 0;
    for (const value of tensorValues) maximum = Math.max(maximum, Math.abs(value));
    const scale = maximum ? maximum / levels : 1;
    for (let index = 0; index < tensorValues.length; index++) {
      values[offset + index] = Math.max(-levels, Math.min(levels, Math.round(tensorValues[index] / scale)));
    }
    tensors.push({ name, offset, length: tensorValues.length, scale });
    offset += tensorValues.length;
  }

  const data = packSigned(values, bits);
  return {
    data,
    parameterCount,
    metadata: {
      type: `symmetric-int${bits}-per-tensor`,
      bits,
      parameterCount,
      packedByteLength: data.byteLength,
      tensors,
    },
  };
}

export function packSigned(values, bits) {
  const mask = 2 ** bits - 1;
  const packed = new Uint8Array(Math.ceil(values.length * bits / 8));
  let bitOffset = 0;
  for (const value of values) {
    const encoded = value & mask;
    const byteOffset = bitOffset >> 3;
    const shift = bitOffset & 7;
    packed[byteOffset] |= encoded << shift;
    if (shift + bits > 8) packed[byteOffset + 1] |= encoded >> (8 - shift);
    bitOffset += bits;
  }
  return packed;
}

/** Encode packed signed integers as one compression-friendly ASCII symbol each. */
export function encodeWeightSymbols(data, parameterCount, bits) {
  if (![4, 5, 6].includes(bits) || data.length !== Math.ceil(parameterCount * bits / 8)) {
    throw new Error("packed weights do not match their int width");
  }
  let result = "";
  for (let index = 0; index < parameterCount; index++) {
    const value = unpackSigned(data, index, bits);
    result += WEIGHT_SYMBOLS[value < 0 ? -value * 2 - 1 : value * 2];
  }
  return result;
}

export function dequantizeTensors(data, quantization) {
  const values = new Float32Array(quantization.parameterCount);
  const mask = 2 ** quantization.bits - 1;
  const sign = 2 ** (quantization.bits - 1);
  for (const tensor of quantization.tensors) {
    for (let index = 0; index < tensor.length; index++) {
      const valueIndex = tensor.offset + index;
      const bitOffset = valueIndex * quantization.bits;
      const byteOffset = bitOffset >> 3;
      const shift = bitOffset & 7;
      const encoded = ((data[byteOffset] | ((data[byteOffset + 1] ?? 0) << 8)) >> shift) & mask;
      values[valueIndex] = (encoded & sign ? encoded - mask - 1 : encoded) * tensor.scale;
    }
  }
  return Object.fromEntries(quantization.tensors.map((tensor) => [
    tensor.name,
    values.slice(tensor.offset, tensor.offset + tensor.length),
  ]));
}

function unpackSigned(data, index, bits) {
  const bitOffset = index * bits;
  const byteOffset = bitOffset >> 3;
  const shift = bitOffset & 7;
  const mask = 2 ** bits - 1;
  const sign = 2 ** (bits - 1);
  const encoded = ((data[byteOffset] | ((data[byteOffset + 1] ?? 0) << 8)) >> shift) & mask;
  return encoded & sign ? encoded - mask - 1 : encoded;
}

// Decodes the embedded weight blob. Zero dependencies and no network fetch: the
// model is ~35 KB, so inlining it costs less than the round-trip it saves and
// keeps `lex` usable from a single import.
//
// Quantized codes (the bit-packed weight planes) and everything else (scales,
// biases, norm gains -- shipped as fp16) use different encodings on purpose.
// Codes are one character per value (SYM below) rather than base85 of the
// packed bytes: packing densely is *smaller* uncompressed but compresses far
// worse in isolation, since dense bits look close to random noise to a
// general-purpose compressor while long runs of repeated small values
// (common in low-bit codes) are exactly what it can exploit -- the SYM string
// alone was 26.8 KB after Brotli against 37.2 KB for base85 of the same
// bytes. That gain does not survive being shipped in the same file as the
// f16 base85 string and the JSON metadata, though: end to end, the real
// generated weights.js Brotli-compresses to about the same size either way,
// because Brotli fits one shared entropy model to the whole file and mixing
// SYM's 16-symbol alphabet with base85's 90-symbol one mostly erases SYM's
// own advantage (see bundle_lex.py's module docstring for the numbers). This
// is kept anyway because the code is headed upstream to a project that may
// end up serving the weight blob as its own request rather than inlined,
// where the isolated-stream number would actually apply. The two strings
// stay separate either way because concatenating them compresses worse than
// keeping them apart, even before metadata is added.

// 85 symbols packing 4 bytes into 5 characters (25% overhead) instead of
// base64's 3-into-4 (33%) -- the same alphabet and padding rule as Python's
// stdlib base64.b85encode/b85decode, so bundle_lex.py can just call that
// directly rather than shipping a second implementation to stay in sync with.
const B85 =
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~';
const B85_LOOKUP = new Uint8Array(128);
for (let i = 0; i < B85.length; i++) B85_LOOKUP[B85.charCodeAt(i)] = i;

/**
 * base85 -> bytes. A short final group is padded with '~' (the alphabet's
 * highest-value character) up to 5 characters before decoding, then the
 * corresponding number of trailing bytes is dropped -- the exact inverse of
 * how b85encode pads with zero bytes and truncates output characters.
 */
export function decodeBase85(str) {
  const padding = (5 - (str.length % 5)) % 5;
  const n = str.length + padding;
  const out = new Uint8Array((n / 5) * 4);
  let o = 0;
  for (let i = 0; i < n; i += 5) {
    let acc = 0;
    for (let j = 0; j < 5; j++) {
      const k = i + j;
      acc = acc * 85 + B85_LOOKUP[k < str.length ? str.charCodeAt(k) : 126]; // 126 = '~'
    }
    out[o++] = (acc >>> 24) & 255;
    out[o++] = (acc >>> 16) & 255;
    out[o++] = (acc >>> 8) & 255;
    out[o++] = acc & 255;
  }
  return out.subarray(0, out.length - padding);
}

// One character per quantized code (0-15): every tensor in this model is 1,
// 3, or 4 bits, so hex digits cover it exactly with no wasted alphabet --
// must match SYMBOL_ALPHABET in train/bundle_lex.py exactly.
const SYM_LOOKUP = new Uint8Array(128);
for (let i = 0; i < 16; i++) SYM_LOOKUP['0123456789ABCDEF'.charCodeAt(i)] = i;

/** hex-per-code string -> Uint8Array of 0-15 code values, one per character. */
function decodeSymbols(str) {
  const out = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) out[i] = SYM_LOOKUP[str.charCodeAt(i)];
  return out;
}

/**
 * Codes -> densely bit-packed planes, the exact inverse of train/quant.py's
 * pack_bitplanes: plane b holds bit b of code i at word i>>5, bit i&31, and
 * the output is laid out bit-plane-major (all of plane 0's words, then all of
 * plane 1's, ...) to match `planes2d.ravel()` there.
 */
function packBitplanes(codes, bits) {
  const words = Math.ceil(codes.length / 32);
  const out = new Uint32Array(bits * words);
  for (let i = 0; i < codes.length; i++) {
    const w = i >>> 5;
    const bit = i & 31;
    const v = codes[i];
    for (let b = 0; b < bits; b++) {
      if ((v >>> b) & 1) out[b * words + w] |= (1 << bit);
    }
  }
  return out;
}

/**
 * IEEE half -> float. The model file stores scales, biases, norm gains and decay
 * logits as fp16; the shader reads f32, so this runs once at load over a couple
 * of thousand values.
 */
export function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x3ff;
    if (exp === 0) {
      if (man === 0) {
        u32[0] = sign;
      } else {
        // Subnormal: renormalize into a float32 exponent.
        exp = 1;
        while ((man & 0x400) === 0) {
          man <<= 1;
          exp--;
        }
        man &= 0x3ff;
        u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
      }
    } else if (exp === 0x1f) {
      u32[0] = sign | 0x7f800000 | (man << 13);
    } else {
      u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    out[i] = f32[0];
  }
  return out;
}

/**
 * Reassemble the two GPU buffers the shader binds from the hex-per-code
 * string (quantized weights) and the base85 string (everything else, fp16).
 * `meta.sym_tensors` records where each tensor's codes live in `sym` and
 * where its packed planes belong in the output, written by bundle_lex.py in
 * the same pass that produced `sym` so the two always agree.
 */
export function unpackWeights(sym, f16b85, meta) {
  const planes = new Uint32Array(meta.plane_words);
  for (const t of Object.values(meta.sym_tensors)) {
    const codes = decodeSymbols(sym.substr(t.offset, t.words * 32));
    planes.set(packBitplanes(codes, t.bits), t.plane_offset);
  }
  const f16Bytes = decodeBase85(f16b85);
  const halves = new Uint16Array(
    f16Bytes.buffer.slice(f16Bytes.byteOffset, f16Bytes.byteOffset + meta.f16_bytes));
  return { planes, fp: halfToFloat(halves) };
}

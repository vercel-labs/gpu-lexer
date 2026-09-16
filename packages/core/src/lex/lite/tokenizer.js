// CPU pre-tokenizer. Splits source into runs of one of four kinds and derives the
// sparse feature fields the model embeds. This is a direct port of the Python
// tokenizer used for training -- the two must agree exactly, because a feature
// computed differently at inference is a feature the model was never trained on.
//
// Kinds: 0 word, 1 space, 2 newline, 3 symbol.

const TRANSITIONS = new Map([
  [(47 << 8) | 47, 1],   // //
  [(47 << 8) | 42, 2],   // /*
  [(42 << 8) | 47, 3],   // */
  [(45 << 8) | 45, 4],   // --
  [(61 << 8) | 62, 5],   // =>
  [(58 << 8) | 58, 6],   // ::
  [(60 << 8) | 47, 7],   // </
  [(123 << 8) | 123, 8], // {{
  [(36 << 8) | 123, 9],  // ${
  [(125 << 8) | 125, 10],// }}
  [(45 << 8) | 62, 11],  // ->
  [(63 << 8) | 63, 12],  // ??
  [(63 << 8) | 46, 13],  // ?.
  [(60 << 8) | 62, 14],  // <>
]);

const isSpace = (c) => c === 9 || c === 11 || c === 12 || c === 32;
const isWord = (c) =>
  c > 127 || c === 95 || (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
const charBucket = (c) => (c > 127 ? 95 : c & 127);

function lenBucket(n) {
  if (n <= 1) return 0;
  return Math.min(7, 31 - Math.clz32(n));
}

function symbolHash(a, b) {
  const v = ((Math.imul(a + 1, 131) >>> 0) ^ b) >>> 0;
  return 1 + (v % 31);
}

/**
 * @param {string} text
 * @returns {{count:number, starts:Int32Array, ends:Int32Array, kinds:Uint8Array, packed:Uint32Array}}
 *   `packed` holds two u32 per token in exactly the layout the shader unpacks.
 */
export function tokenize(text) {
  const n = text.length;
  // Upper bound: every character its own token.
  const starts = new Int32Array(n);
  const ends = new Int32Array(n);
  const kinds = new Uint8Array(n);
  const firstC = new Uint8Array(n);
  const lastC = new Uint8Array(n);
  const lenB = new Uint8Array(n);
  const h1 = new Uint16Array(n);
  const h2 = new Uint8Array(n);
  const flags = new Uint8Array(n);
  const transPrev = new Uint8Array(n);
  const transNext = new Uint8Array(n);
  const symPrev = new Uint8Array(n);
  const symNext = new Uint8Array(n);

  let count = 0;
  let pos = 0;
  let lineStart = true;

  while (pos < n) {
    const start = pos;
    const c = text.charCodeAt(pos);
    let kind;
    if (c === 10 || c === 13) kind = 2;
    else if (isSpace(c)) kind = 1;
    else if (isWord(c)) kind = 0;
    else kind = 3;

    let f = lineStart ? 16 : 0;
    const first = charBucket(c);
    let last = first;

    if (kind === 0) {
      let a = 2166136261 >>> 0;
      let b = 2654435769 >>> 0;
      while (pos < n && isWord(text.charCodeAt(pos))) {
        const cb = charBucket(text.charCodeAt(pos));
        last = cb;
        a = Math.imul(a ^ cb, 16777619) >>> 0;
        b = Math.imul(b ^ cb, 2246822519) >>> 0;
        if (cb >= 97 && cb <= 122) f |= 1;
        else if (cb >= 65 && cb <= 90) f |= 2;
        else if (cb >= 48 && cb <= 57) f |= 4;
        else if (cb === 95) f |= 8;
        pos++;
      }
      h1[count] = (a ^ (a >>> 16)) & 511;
      h2[count] = (b ^ (b >>> 16)) & 127;
      if ((f & 2) && !(f & 1)) f |= 128;
    } else if (kind === 1) {
      while (pos < n && isSpace(text.charCodeAt(pos))) {
        const cc = text.charCodeAt(pos);
        last = charBucket(cc);
        if (cc === 9) f |= 32;
        pos++;
      }
    } else if (kind === 2) {
      pos++;
      if (c === 13 && pos < n && text.charCodeAt(pos) === 10) {
        last = 10;
        pos++;
      }
    } else {
      if (c === 92) f |= 64;
      pos++;
    }

    starts[count] = start;
    ends[count] = pos;
    kinds[count] = kind;
    firstC[count] = first;
    lastC[count] = last;
    lenB[count] = lenBucket(pos - start);
    flags[count] = f;
    count++;

    if (kind === 2) lineStart = true;
    else if (kind !== 1) lineStart = false;
  }

  for (let i = 1; i < count; i++) {
    const t = TRANSITIONS.get((lastC[i - 1] << 8) | firstC[i]) || 0;
    transPrev[i] = t;
    transNext[i - 1] = t;
    if (kinds[i - 1] === 3 || kinds[i] === 3) {
      const s = symbolHash(lastC[i - 1], firstC[i]);
      symPrev[i] = s;
      symNext[i - 1] = s;
    }
  }

  // Pack into the two-word layout the shader reads. Keeping the packing here
  // means the GPU never does field arithmetic it can avoid.
  const packed = new Uint32Array(count * 2);
  for (let i = 0; i < count; i++) {
    packed[i * 2] =
      (kinds[i] & 3) |
      ((lenB[i] & 7) << 2) |
      ((firstC[i] & 127) << 5) |
      ((lastC[i] & 127) << 12) |
      ((flags[i] & 255) << 19) |
      ((symNext[i] & 31) << 27);
    packed[i * 2 + 1] =
      (h1[i] & 1023) |
      ((h2[i] & 127) << 10) |
      ((transPrev[i] & 15) << 17) |
      ((transNext[i] & 15) << 21) |
      ((symPrev[i] & 31) << 25);
  }

  return {
    count,
    starts: starts.subarray(0, count),
    ends: ends.subarray(0, count),
    kinds: kinds.subarray(0, count),
    packed,
  };
}

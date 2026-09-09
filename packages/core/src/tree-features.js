import {
  TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, TREE_PART_WORD,
} from "./constants.js";

export {
  TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, TREE_PART_WORD,
};

export const TREE_LEGACY_FEATURE_VERSION = 2;
export const TREE_FEATURE_VERSION = 3;
export const TREE_HASH_BUCKETS = 256;
export const TREE_SECONDARY_HASH_BUCKETS = 128;
export const TREE_SYMBOL_PAIR_BUCKETS = 32;
export const TREE_LEGACY_SCALE_BUCKETS = 8;
export const TREE_SCALE_BUCKETS = 12;

export function isTreeFeatureVersion(version) {
  return version === TREE_LEGACY_FEATURE_VERSION || version === TREE_FEATURE_VERSION;
}

export function treeScaleBuckets(version = TREE_FEATURE_VERSION) {
  if (!isTreeFeatureVersion(version)) throw new Error(`unsupported tree feature version ${version}`);
  return version === TREE_LEGACY_FEATURE_VERSION ? TREE_LEGACY_SCALE_BUCKETS : TREE_SCALE_BUCKETS;
}

export function treeFeatureLayout(hashBuckets = TREE_HASH_BUCKETS, version = TREE_FEATURE_VERSION) {
  if (!isTreeFeatureVersion(version)) throw new Error(`unsupported tree feature version ${version}`);
  if (!Number.isInteger(hashBuckets) || hashBuckets < 0 || hashBuckets > 256 ||
      (hashBuckets > 0 && (hashBuckets & (hashBuckets - 1)) !== 0)) {
    throw new RangeError("tree hash buckets must be zero or a power of two up to 256");
  }
  let offset = 268;
  const primaryHash = offset;
  offset += hashBuckets;
  const secondaryHashBuckets = version >= 3 && hashBuckets ? TREE_SECONDARY_HASH_BUCKETS : 0;
  const secondaryHash = offset;
  offset += secondaryHashBuckets;
  const flags = offset;
  offset += 7;
  const previousPair = offset;
  offset += 16;
  const nextPair = offset;
  offset += 16;
  const previousSymbolPair = offset;
  if (version >= 3) offset += TREE_SYMBOL_PAIR_BUCKETS;
  const nextSymbolPair = offset;
  if (version >= 3) offset += TREE_SYMBOL_PAIR_BUCKETS;
  return {
    inputSize: offset, primaryHash, secondaryHash, secondaryHashBuckets,
    flags, previousPair, nextPair, previousSymbolPair, nextSymbolPair,
  };
}

// The pair encoders reserve zero for "no feature". pairCode also never emits
// 15, so those rows can be omitted from the runtime embedding without changing
// any input the model can observe.
export function runtimeTreeFeatureLayout(hashBuckets = TREE_HASH_BUCKETS, version = TREE_FEATURE_VERSION) {
  const layout = treeFeatureLayout(hashBuckets, version);
  const previousPair = layout.previousPair;
  const nextPair = previousPair + 14;
  const previousSymbolPair = nextPair + 14;
  const symbolBuckets = version >= 3 ? TREE_SYMBOL_PAIR_BUCKETS - 1 : 0;
  const nextSymbolPair = previousSymbolPair + symbolBuckets;
  return {
    ...layout,
    inputSize: nextSymbolPair + symbolBuckets,
    previousPair,
    nextPair,
    previousSymbolPair,
    nextSymbolPair,
  };
}

export function unreachableTreeFeatureRows(hashBuckets = TREE_HASH_BUCKETS, version = TREE_FEATURE_VERSION) {
  const layout = treeFeatureLayout(hashBuckets, version);
  return [
    layout.previousPair,
    layout.previousPair + 15,
    layout.nextPair,
    layout.nextPair + 15,
    ...(version >= 3 ? [layout.previousSymbolPair, layout.nextSymbolPair] : []),
  ];
}

export function treeInputSize(hashBuckets = TREE_HASH_BUCKETS, version = TREE_FEATURE_VERSION) {
  return treeFeatureLayout(hashBuckets, version).inputSize;
}

/** Expand one two-word simple-part record to sparse model features. */
export function treeFeatureIndices(data, partIndex, hashBuckets = TREE_HASH_BUCKETS, version = TREE_FEATURE_VERSION) {
  const base = partIndex * TREE_FEATURE_STRIDE;
  const packed = data[base];
  const context = data[base + 1];
  const kind = packed & 3;
  const layout = treeFeatureLayout(hashBuckets, version);
  const indices = [
    kind,
    4 + ((packed >>> 2) & 7),
    12 + ((packed >>> 5) & 127),
    140 + ((packed >>> 12) & 127),
  ];
  if (kind === TREE_PART_WORD && hashBuckets) {
    indices.push(layout.primaryHash + (((packed >>> 19) & 255) & (hashBuckets - 1)));
    if (layout.secondaryHashBuckets) {
      indices.push(layout.secondaryHash + ((context >>> 15) & (layout.secondaryHashBuckets - 1)));
    }
  }
  const flags = context & 127;
  for (let bit = 0; bit < 7; bit++) if ((flags & (1 << bit)) !== 0) indices.push(layout.flags + bit);
  const previousPair = (context >>> 7) & 15;
  const nextPair = (context >>> 11) & 15;
  if (previousPair) indices.push(layout.previousPair + previousPair);
  if (nextPair) indices.push(layout.nextPair + nextPair);
  if (version >= 3) {
    const previousSymbolPair = (context >>> 22) & 31;
    const nextSymbolPair = (context >>> 27) & 31;
    if (previousSymbolPair) indices.push(layout.previousSymbolPair + previousSymbolPair);
    if (nextSymbolPair) indices.push(layout.nextSymbolPair + nextSymbolPair);
  }
  return indices;
}

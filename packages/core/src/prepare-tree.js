import {
  TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, TREE_PART_WORD,
} from "./constants.js";

const INITIAL_CAPACITY = 512;
const MAX_POOLED_WORKSPACES = 4;
const pool = [];
const WORKSPACE_CAPACITY = 0;
const WORKSPACE_DATA = 1;
const WORKSPACE_RANGES = 2;
const PREPARED_WORKSPACE = 4;

function createWorkspace() {
  return [
    INITIAL_CAPACITY,
    new Uint32Array(INITIAL_CAPACITY * TREE_FEATURE_STRIDE),
    new Uint32Array(INITIAL_CAPACITY * 2),
  ];
}

function growWorkspace(workspace) {
  const capacity = workspace[WORKSPACE_CAPACITY] * 2;
  const data = new Uint32Array(capacity * TREE_FEATURE_STRIDE);
  const ranges = new Uint32Array(capacity * 2);
  data.set(workspace[WORKSPACE_DATA]);
  ranges.set(workspace[WORKSPACE_RANGES]);
  workspace[WORKSPACE_CAPACITY] = capacity;
  workspace[WORKSPACE_DATA] = data;
  workspace[WORKSPACE_RANGES] = ranges;
}

/** Split source into words, horizontal-space runs, individual newlines, and symbols. */
export function prepareTreeSource(source) {
  const workspace = pool.pop() ?? createWorkspace();
  let count = 0;
  let offset = 0;
  let lineStart = true;
  let previousLast = -1;
  let previousKind = -1;
  while (offset < source.length) {
    const from = offset;
    const firstRaw = source.charCodeAt(offset);
    let kind;
    if (firstRaw === 10 || firstRaw === 13) {
      kind = TREE_PART_NEWLINE;
    } else if (isHorizontalSpace(firstRaw)) {
      kind = TREE_PART_SPACE;
    } else if (isWord(firstRaw)) {
      kind = TREE_PART_WORD;
    } else {
      kind = TREE_PART_SYMBOL;
    }

    let flags = lineStart ? 16 : 0;
    const first = normalize(firstRaw);
    let last = first;
    let hash = 0;
    let secondaryHash = 0;
    if (kind === TREE_PART_WORD) {
      hash = 2166136261;
      secondaryHash = 0x9e3779b9;
      do {
        const normalized = normalize(source.charCodeAt(offset));
        last = normalized;
        hash = Math.imul(hash ^ normalized, 16777619);
        secondaryHash = Math.imul(secondaryHash ^ normalized, 2246822519);
        if (normalized >= 97 && normalized <= 122) flags |= 1;
        else if (normalized >= 65 && normalized <= 90) flags |= 2;
        else if (normalized >= 48 && normalized <= 57) flags |= 4;
        else if (normalized === 95) flags |= 8;
        offset += 1;
      } while (offset < source.length && isWord(source.charCodeAt(offset)));
    } else if (kind === TREE_PART_SPACE) {
      do {
        const raw = source.charCodeAt(offset);
        last = normalize(raw);
        if (raw === 9) flags |= 32;
        offset += 1;
      } while (offset < source.length && isHorizontalSpace(source.charCodeAt(offset)));
    } else if (kind === TREE_PART_NEWLINE) {
      offset += 1;
      if (firstRaw === 13 && source.charCodeAt(offset) === 10) {
        last = 10;
        offset += 1;
      }
    } else {
      if (firstRaw === 92) flags |= 64;
      offset += 1;
    }
    if (count === workspace[WORKSPACE_CAPACITY]) growWorkspace(workspace);
    const to = offset;
    const length = to - from;
    const base = count * TREE_FEATURE_STRIDE;
    workspace[WORKSPACE_DATA][base] = kind | (lengthBucket(length) << 2) | (first << 5) |
      (last << 12) | ((hash & 255) << 19);
    workspace[WORKSPACE_DATA][base + 1] = flags |
      (kind === TREE_PART_WORD ? (secondaryHash & 127) << 15 : 0);
    if (count > 0) {
      const pair = pairCode(previousLast, first);
      workspace[WORKSPACE_DATA][base + 1] |= pair << 7;
      workspace[WORKSPACE_DATA][base - 1] |= pair << 11;
      if (kind === TREE_PART_SYMBOL || previousKind === TREE_PART_SYMBOL) {
        const generic = symbolPairHash(previousLast, first);
        workspace[WORKSPACE_DATA][base + 1] |= generic << 22;
        workspace[WORKSPACE_DATA][base - 1] |= generic << 27;
      }
    }
    workspace[WORKSPACE_RANGES][count * 2] = from;
    workspace[WORKSPACE_RANGES][count * 2 + 1] = to;
    count += 1;
    previousLast = last;
    previousKind = kind;
    if (kind === TREE_PART_NEWLINE) lineStart = true;
    else if (kind !== TREE_PART_SPACE) lineStart = false;
  }
  // Compact internal tuple: data, ranges, streams, token count, workspace.
  return [
    workspace[WORKSPACE_DATA].subarray(0, count * TREE_FEATURE_STRIDE),
    workspace[WORKSPACE_RANGES].subarray(0, count * 2),
    Uint32Array.of(0, count), count, workspace,
  ];
}

function symbolPairHash(left, right) {
  // Zero remains an edge/no-neighbor sentinel; 31 buckets cover every unseen
  // neighboring boundary without a growing hand-authored operator table.
  return 1 + ((Math.imul(left + 1, 131) ^ right) >>> 0) % 31;
}

export function releaseTreePrepared(prepared) {
  if (!prepared?.[PREPARED_WORKSPACE]) return;
  if (pool.length < MAX_POOLED_WORKSPACES) pool.push(prepared[PREPARED_WORKSPACE]);
  prepared[PREPARED_WORKSPACE] = null;
}

function isHorizontalSpace(code) {
  return code === 9 || code === 11 || code === 12 || code === 32;
}

function isWord(code) {
  return code > 127 || code === 95 || (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
}

function normalize(code) {
  return code > 127 ? 95 : code;
}

function lengthBucket(length) {
  return Math.min(7, 31 - Math.clz32(length));
}

function pairCode(left, right) {
  if (left === 47 && right === 47) return 1; // //
  if (left === 47 && right === 42) return 2; // /*
  if (left === 42 && right === 47) return 3; // */
  if (left === 45 && right === 45) return 4; // --
  if (left === 61 && right === 62) return 5; // =>
  if (left === 58 && right === 58) return 6; // ::
  if (left === 60 && right === 47) return 7; // </
  if (left === 123 && right === 123) return 8; // {{
  if (left === 36 && right === 123) return 9; // ${
  if (left === 125 && right === 125) return 10; // }}
  if (left === 45 && right === 62) return 11; // ->
  if (left === 63 && right === 63) return 12; // ??
  if (left === 63 && right === 46) return 13; // ?.
  if (left === 60 && right === 62) return 14; // <>
  return 0;
}

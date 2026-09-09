const STORAGE = 128;
const COPY_SRC = 4;
const COPY_DST = 8;
const MAP_READ = 1;
const UNIFORM = 64;
const FEATURE_STRIDE = 3;
const LOCAL_PROFILE_RADIUS = 32;
const FEATURE_STATS = 10;
const rawPipelineCache = new WeakMap();
const BOUNDARY_PIPELINES = 0;
const FEATURE_PIPELINES = 1;

export const RAW_PREPASS_SHADER = /* wgsl */ `
struct Params { units: u32, bytes: u32, encoding: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn utf16_at(index: u32) -> u32 {
  let word = input[index >> 1u];
  return (word >> ((index & 1u) * 16u)) & 0xffffu;
}

fn byte_at(index: u32) -> u32 {
  return (input[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu;
}

fn normalized(code: u32) -> u32 { return select(code, 95u, code > 127u); }
fn ascii_word(code: u32) -> bool {
  return (code >= 65u && code <= 90u) || (code >= 97u && code <= 122u) || code == 95u;
}
fn is_operator(code: u32) -> bool {
  return code == 43u || code == 45u || code == 42u || code == 47u || code == 37u ||
    code == 61u || code == 33u || code == 60u || code == 62u || code == 38u ||
    code == 124u || code == 94u || code == 126u || code == 35u || code == 36u || code == 92u;
}
fn kind(code: u32) -> u32 {
  if (code == 32u || (code >= 9u && code <= 13u)) { return 0u; }
  if (ascii_word(code)) { return 1u; }
  if (code >= 48u && code <= 57u) { return 2u; }
  if (code == 39u || code == 34u || code == 96u) { return 3u; }
  if (is_operator(code)) { return 5u; }
  if (code == 40u || code == 41u || code == 91u || code == 93u || code == 123u || code == 125u) { return 6u; }
  if (code == 44u || code == 59u || code == 58u || code == 46u || code == 63u || code == 64u) { return 7u; }
  return 8u;
}
fn continues(token_kind: u32, code: u32) -> bool {
  if (token_kind == 1u) { return ascii_word(code) || (code >= 48u && code <= 57u); }
  if (token_kind == 2u) { return (code >= 48u && code <= 57u) || ascii_word(code) || code == 46u; }
  return token_kind == 5u && is_operator(code);
}
fn packed(code: u32, previous: u32, first: bool, width: u32) -> u32 {
  let token_kind = kind(code);
  let boundary = first || token_kind != kind(previous) || !continues(token_kind, code);
  return token_kind | (u32(boundary) << 8u) | (width << 16u);
}

@compute @workgroup_size(256)
fn utf16(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.units) { return; }
  let code = normalized(utf16_at(index));
  var previous = 0u;
  if (index > 0u) { previous = normalized(utf16_at(index - 1u)); }
  output[index] = packed(code, previous, index == 0u, 1u);
}

fn continuation(byte: u32) -> bool { return (byte & 0xc0u) == 0x80u; }
fn previous_lead(index: u32) -> u32 {
  var cursor = index - 1u;
  for (var step = 0u; step < 3u && cursor > 0u && continuation(byte_at(cursor)); step++) {
    cursor -= 1u;
  }
  return cursor;
}

@compute @workgroup_size(256)
fn utf8(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.bytes) { return; }
  let byte = byte_at(index);
  if (continuation(byte)) { output[index] = 0u; return; }
  let ascii = byte < 128u;
  let code = select(95u, byte, ascii);
  var previous = 0u;
  if (index > 0u) {
    let previous_byte = byte_at(previous_lead(index));
    previous = select(95u, previous_byte, previous_byte < 128u);
  }
  let width = select(1u, 2u, (byte & 0xf8u) == 0xf0u);
  output[index] = packed(code, previous, index == 0u, width);
}
`;

export const UTF8_FEATURE_SHADER = /* wgsl */ `
struct Params { tokens: u32, bytes: u32, streams: u32, _pad: u32 }
struct Stats {
  width: u32, brackets: u32, angles: u32, separators: u32, quotes: u32,
  sigils: u32, pairs: u32, layout_count: u32, identifiers: u32, hash: u32,
}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read_write> streams: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<Stats>;
@group(0) @binding(4) var<storage, read_write> features: array<u32>;
@group(0) @binding(5) var<storage, read_write> document: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 { return (input[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu; }
fn ascii_word(code: u32) -> bool {
  return (code >= 65u && code <= 90u) || (code >= 97u && code <= 122u) || code == 95u;
}
fn is_operator(code: u32) -> bool {
  return code == 43u || code == 45u || code == 42u || code == 47u || code == 37u ||
    code == 61u || code == 33u || code == 60u || code == 62u || code == 38u ||
    code == 124u || code == 94u || code == 126u || code == 35u || code == 36u || code == 92u;
}
fn kind(code: u32) -> u32 {
  if (ascii_word(code)) { return 1u; }
  if (code >= 48u && code <= 57u) { return 2u; }
  if (code == 39u || code == 34u || code == 96u) { return 3u; }
  if (is_operator(code)) { return 5u; }
  if (code == 40u || code == 41u || code == 91u || code == 93u || code == 123u || code == 125u) { return 6u; }
  if (code == 44u || code == 59u || code == 58u || code == 46u || code == 63u || code == 64u) { return 7u; }
  return 8u;
}
fn continues(token_kind: u32, code: u32) -> bool {
  if (token_kind == 1u) { return ascii_word(code) || (code >= 48u && code <= 57u); }
  if (token_kind == 2u) { return (code >= 48u && code <= 57u) || ascii_word(code) || code == 46u; }
  return token_kind == 5u && is_operator(code);
}
fn character_class(code: u32) -> u32 {
  if (code >= 97u && code <= 122u) { return 1u; }
  if (code >= 65u && code <= 90u) { return 2u; }
  if (code >= 48u && code <= 57u) { return 3u; }
  if (code == 95u || code == 36u) { return 4u; }
  if (code <= 32u || code == 127u) { return 5u; }
  return 7u;
}
fn profile_pair(previous: u32, current: u32) -> bool {
  return (previous == 47u && (current == 47u || current == 42u)) ||
    (previous == 45u && current == 45u) || (previous == 61u && current == 62u) ||
    (previous == 58u && current == 58u) || (previous == 60u && current == 47u) ||
    (previous == 123u && current == 123u);
}
fn hash2(first: u32, second: u32) -> u32 {
  return (((2166136261u ^ first) * 16777619u) ^ second) * 16777619u;
}
fn hash3(first: u32, second: u32, third: u32) -> u32 {
  return (hash2(first, second) ^ third) * 16777619u;
}
fn length_bucket(length: u32) -> u32 {
  if (length >= 128u) { return 7u; } if (length >= 64u) { return 6u; }
  if (length >= 32u) { return 5u; } if (length >= 16u) { return 4u; }
  if (length >= 8u) { return 3u; } if (length >= 4u) { return 2u; }
  return select(0u, 1u, length >= 2u);
}
fn density_bucket(count: u32, length: u32, ratio: bool) -> u32 {
  if (ratio) {
    if (count * 20u < length * 7u) { return 0u; }
    if (count * 20u < length * 11u) { return 1u; }
    if (count * 25u < length * 18u) { return 2u; }
    return 3u;
  }
  if (count == 0u) { return 0u; }
  if (count * 64u < length) { return 1u; }
  if (count * 64u < length * 3u) { return 2u; }
  return 3u;
}
fn stat_value(value: Stats, dimension: u32) -> u32 {
  switch dimension {
    case 0u: { return value.brackets; } case 1u: { return value.angles; }
    case 2u: { return value.separators; } case 3u: { return value.quotes; }
    case 4u: { return value.sigils; } case 5u: { return value.pairs; }
    case 6u: { return value.layout_count; } default: { return value.identifiers; }
  }
}
fn range_is(token: u32, first: u32, second: u32) -> bool {
  let token_from = ranges[token * 2u];
  let token_to = ranges[token * 2u + 1u];
  return token_to - token_from == 2u && byte_at(token_from) == first && byte_at(token_from + 1u) == second;
}
fn odd_trailing_backslashes(token: u32) -> bool {
  let token_from = ranges[token * 2u];
  var cursor = ranges[token * 2u + 1u];
  var count = 0u;
  while (cursor > token_from && byte_at(cursor - 1u) == 92u) { count += 1u; cursor -= 1u; }
  return (count & 1u) == 1u;
}

@compute @workgroup_size(256)
fn token_features(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= params.tokens) { return; }
  let token_from = ranges[token * 2u];
  let to = ranges[token * 2u + 1u];
  var previous_to = 0u;
  if (token > 0u) { previous_to = ranges[token * 2u - 1u]; }
  let length = to - token_from;
  var gap_layout = 0u;
  var gap_newline = false;
  var line_start = false;
  for (var index = previous_to; index < token_from; index++) {
    let code = byte_at(index);
    if (code == 10u || code == 13u) { gap_newline = true; }
    if (code == 10u) { gap_layout += 1u; line_start = true; }
    else if (line_start && (code == 32u || code == 9u)) { gap_layout += 1u; line_start = false; }
    else if (code != 13u) { line_start = false; }
  }
  var hash = 2166136261u;
  var token_flags = select(0u, 128u, token_from > previous_to) | select(0u, 256u, gap_newline);
  var has_lower = false;
  var first0 = 0u; var first1 = 0u; var first2 = 0u;
  var last0 = 0u; var last1 = 0u; var last2 = 0u;
  var brackets = 0u; var angles = 0u; var separators = 0u; var quotes = 0u;
  var sigils = 0u; var pairs = 0u; var identifiers = 0u;
  var previous_code = 0u;
  if (token_from > 0u) { previous_code = byte_at(token_from - 1u); }
  for (var index = token_from; index < to; index++) {
    let local = index - token_from;
    let code = byte_at(index);
    if (local == 0u) { first0 = code; } else if (local == 1u) { first1 = code; } else if (local == 2u) { first2 = code; }
    last0 = last1; last1 = last2; last2 = code;
    hash = (hash ^ code) * 16777619u;
    if (code == 92u) { token_flags |= 2u; }
    else if (code >= 65u && code <= 90u) { token_flags |= 4u; }
    else if (code >= 97u && code <= 122u) { token_flags |= 8u; has_lower = true; }
    else if (code >= 48u && code <= 57u) { token_flags |= 16u; }
    brackets += u32(code == 40u || code == 41u || code == 91u || code == 93u || code == 123u || code == 125u);
    angles += u32(code == 60u || code == 62u);
    separators += u32(code == 59u || code == 58u || code == 44u || code == 46u);
    sigils += u32(code == 35u || code == 36u || code == 64u);
    quotes += u32(code == 39u || code == 34u || code == 96u);
    pairs += u32(code == 92u) + 2u * u32(profile_pair(previous_code, code));
    identifiers += u32((code >= 48u && code <= 57u) || (code >= 65u && code <= 90u) ||
      (code >= 97u && code <= 122u) || code == 95u);
    previous_code = code;
  }
  if (length > 1u && !has_lower) { token_flags |= 32u; }
  let token_kind = kind(first0);
  var continuation = false;
  if (token > 0u && previous_to == token_from) {
    let previous_kind = kind(byte_at(ranges[(token - 1u) * 2u]));
    continuation = previous_kind == token_kind && continues(token_kind, first0);
  }
  token_flags |= u32(continuation);
  let ngram0 = select(0u, hash2(first0, first1) & 63u, length >= 2u);
  let ngram1 = select(0u, hash2(last1, last2) & 63u, length >= 2u);
  let ngram3 = select(0u, hash3(last0, last1, last2) & 63u, length >= 3u);
  let base = token * ${FEATURE_STRIDE}u;
  features[base] = token_kind | (length_bucket(length) << 4u) | (character_class(first0) << 7u) |
    (character_class(last2) << 10u) | (token_flags << 13u) | ((hash & 128u) << 17u);
  features[base + 1u] = ((hash & 127u) << 3u) | (ngram0 << 10u) | (ngram1 << 16u) |
    (u32(length >= 2u) << 28u) | (u32(length >= 3u) << 29u);
  features[base + 2u] = ngram3;
  stats[token] = Stats(token_from - previous_to + length, brackets, angles, separators, quotes,
    sigils, pairs, gap_layout, identifiers, hash);
}

@compute @workgroup_size(1)
fn document_profile() {
  var totals: array<u32, 8>;
  var width = 0u;
  var votes: array<i32, 8>;
  for (var token = 0u; token < params.tokens; token++) {
    let value = stats[token];
    width += value.width;
    for (var dimension = 0u; dimension < 8u; dimension++) { totals[dimension] += stat_value(value, dimension); }
    if ((features[token * ${FEATURE_STRIDE}u] & 15u) == 1u) {
      for (var bit = 0u; bit < 8u; bit++) {
        votes[bit] += select(-1, 1, ((value.hash >> (bit * 4u)) & 1u) == 1u);
      }
    }
  }
  width = max(1u, width);
  var packed = 0u;
  for (var dimension = 0u; dimension < 8u; dimension++) {
    packed |= density_bucket(totals[dimension], width, dimension == 7u) << (dimension * 2u);
  }
  var sketch = 0u;
  for (var bit = 0u; bit < 8u; bit++) { sketch |= u32(votes[bit] > 0) << bit; }
  document[0] = (packed & 255u) | (sketch << 8u);
}

@compute @workgroup_size(1)
fn segment_tokens() {
  var stream_start = 0u;
  var stream_words = 0u;
  var depth = 0u;
  var quote = 0u;
  var block_comment = false;
  var line_comment = false;
  var previous_from = 0u;
  var previous_to = 0u;
  var blank_candidate = 0u;
  var line_candidate = 0u;
  var fallback_line = 0u;
  for (var token = 0u; token < params.tokens; token++) {
    let token_from = ranges[token * 2u];
    let token_to = ranges[token * 2u + 1u];
    var gap_newline = false;
    var gap_blank = false;
    var seen_line_feed = false;
    var only_indent = false;
    var pending_carriage = false;
    for (var index = previous_to; index < token_from; index++) {
      let code = byte_at(index);
      if (code == 10u || code == 13u) { gap_newline = true; }
      if (code == 10u) {
        if (seen_line_feed && (only_indent || pending_carriage)) { gap_blank = true; }
        seen_line_feed = true; only_indent = true; pending_carriage = false;
      } else if (code == 13u) {
        pending_carriage = seen_line_feed && only_indent;
      } else {
        pending_carriage = false;
        if (code != 32u && code != 9u) { only_indent = false; }
      }
    }
    if (gap_newline) {
      line_comment = false;
      fallback_line = token;
      if (quote == 0u && !block_comment && depth == 0u) {
        line_candidate = token;
        if (gap_blank) { blank_candidate = token; }
      }
    }
    if (!line_comment) {
      let length = token_to - token_from;
      let first = byte_at(token_from);
      if (block_comment) {
        if (range_is(token, 42u, 47u)) { block_comment = false; }
      } else if (quote != 0u) {
        if (length == 1u && first == quote && (token == 0u || !odd_trailing_backslashes(token - 1u))) { quote = 0u; }
      } else if (range_is(token, 47u, 42u)) { block_comment = true; }
      else if (range_is(token, 47u, 47u)) { line_comment = true; }
      else if (length == 1u && (first == 39u || first == 34u || first == 96u)) { quote = first; }
      else if (length == 1u && (first == 40u || first == 91u || first == 123u)) { depth += 1u; }
      else if (length == 1u && (first == 41u || first == 93u || first == 125u) && depth > 0u) { depth -= 1u; }
    }
    previous_from = token_from;
    previous_to = token_to;
    let next = token + 1u;
    if (next - stream_start >= 384u) {
      var preferred = 0u;
      if (blank_candidate >= stream_start + 192u) { preferred = blank_candidate; }
      else if (line_candidate >= stream_start + 192u) { preferred = line_candidate; }
      if (preferred > stream_start) {
        streams[stream_words] = stream_start; streams[stream_words + 1u] = preferred - stream_start;
        stream_words += 2u; stream_start = preferred;
        blank_candidate = 0u; line_candidate = 0u; fallback_line = 0u;
      }
    }
    if (next - stream_start >= 768u) {
      let cut = select(next, fallback_line, fallback_line > stream_start);
      streams[stream_words] = stream_start; streams[stream_words + 1u] = cut - stream_start;
      stream_words += 2u; stream_start = cut;
      blank_candidate = 0u; line_candidate = 0u; fallback_line = 0u;
    }
  }
  if (params.tokens > stream_start) {
    streams[stream_words] = stream_start; streams[stream_words + 1u] = params.tokens - stream_start;
    stream_words += 2u;
  }
  if (stream_words == 0u) { streams[0] = 0u; streams[1] = 0u; stream_words = 2u; }
  document[1] = stream_words;
}

@compute @workgroup_size(256)
fn finish_features(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= params.tokens) { return; }
  let local_from = select(0u, token - ${LOCAL_PROFILE_RADIUS}u, token > ${LOCAL_PROFILE_RADIUS}u);
  let local_to = min(params.tokens, token + ${LOCAL_PROFILE_RADIUS + 1}u);
  var totals: array<u32, 8>;
  var width = 0u;
  for (var index = local_from; index < local_to; index++) {
    let value = stats[index];
    width += value.width;
    for (var dimension = 0u; dimension < 8u; dimension++) { totals[dimension] += stat_value(value, dimension); }
  }
  width = max(1u, width);
  var local_profile = 0u;
  for (var dimension = 0u; dimension < 8u; dimension++) {
    local_profile |= density_bucket(totals[dimension], width, dimension == 7u) << (dimension * 2u);
  }
  var stream_start = token == 0u;
  for (var stream = 1u; stream < document[1] / 2u; stream++) {
    stream_start = stream_start || streams[stream * 2u] == token;
  }
  let base = token * ${FEATURE_STRIDE}u;
  let attributes = features[base];
  let has_newline = (attributes & (256u << 13u)) != 0u;
  if (stream_start || has_newline) { features[base] |= 512u << 13u; }
  if (!stream_start && token > 0u) {
    let previous_base = (token - 1u) * ${FEATURE_STRIDE}u;
    let previous_attributes = features[previous_base];
    let previous_from = ranges[(token - 1u) * 2u];
    let previous_to = ranges[(token - 1u) * 2u + 1u];
    if (previous_to - previous_from == 1u && byte_at(previous_from) == 46u) { features[base] |= 1024u << 13u; }
    features[base] |= (previous_attributes & 15u) << 25u;
    features[base] |= ((previous_attributes >> 7u) & 7u) << 29u;
    features[base + 1u] |= (previous_attributes >> 4u) & 7u;
  }
  features[base + 1u] |= (local_profile >> 10u) << 22u;
  features[base + 2u] |= (document[0] << 6u) | ((local_profile & 1023u) << 22u);
}
`;

export class RawInputWorkspace {
  constructor() {
    this.utf16 = new Uint32Array(0);
    this.utf8 = new Uint8Array(0);
    this.encoder = new TextEncoder();
  }
}

export function packUtf16(source, workspace = new RawInputWorkspace()) {
  const words = Math.ceil(source.length / 2);
  workspace.utf16 = grow(workspace.utf16, words);
  const output = workspace.utf16.subarray(0, words);
  output.fill(0);
  for (let index = 0; index < source.length; index++) {
    output[index >> 1] |= source.charCodeAt(index) << ((index & 1) * 16);
  }
  return output;
}

export function packUtf8(source, workspace = new RawInputWorkspace()) {
  const maximum = source.length * 3;
  workspace.utf8 = grow(workspace.utf8, maximum, Uint8Array);
  const result = workspace.encoder.encodeInto(source, workspace.utf8);
  if (result.read !== source.length) throw new Error("UTF-8 workspace was too small");
  return {
    bytes: workspace.utf8.subarray(0, result.written),
    hasLoneSurrogate: hasLoneSurrogate(source),
  };
}

export async function compareRawGpuPrepasses(device, source, expectedRanges, expectedFeatures, expectedStreams) {
  const workspace = new RawInputWorkspace();
  let started = performance.now();
  const utf16 = packUtf16(source, workspace);
  const utf16PackMs = performance.now() - started;
  started = performance.now();
  const utf8 = packUtf8(source, workspace);
  const utf8PackMs = performance.now() - started;
  const [utf16Tape, utf8Tape] = await Promise.all([
    runPrepass(device, utf16, source.length, "utf16"),
    runPrepass(device, asWords(utf8.bytes), utf8.bytes.length, "utf8"),
  ]);
  const expected = [...expectedRanges];
  const utf16Ranges = rangesFromTape(utf16Tape.tape, false);
  const utf8Ranges = rangesFromTape(utf8Tape.tape, true);
  const ascii = isAsciiSource(source);
  const fullFeatures = ascii && expectedFeatures && expectedStreams
    ? await runUtf8Features(device, asWords(utf8.bytes), utf8.bytes.length, expectedRanges, expectedStreams)
    : null;
  const featureMismatch = fullFeatures ? firstMismatch(expectedFeatures, fullFeatures.features) : -1;
  const streamMismatch = fullFeatures ? firstMismatch(expectedStreams, fullFeatures.streams) : -1;
  return {
    utf16: {
      bytes: utf16.byteLength,
      packMs: utf16PackMs,
      prepass: utf16Tape.timings,
      rangeParity: arraysEqual(expected, utf16Ranges),
      firstRangeMismatch: firstMismatch(expected, utf16Ranges),
      featureParityReady: false,
    },
    utf8: {
      bytes: utf8.bytes.byteLength,
      packMs: utf8PackMs,
      prepass: utf8Tape.timings,
      rangeParity: arraysEqual(expected, utf8Ranges),
      firstRangeMismatch: firstMismatch(expected, utf8Ranges),
      hasLoneSurrogate: utf8.hasLoneSurrogate,
      normalizedUnitParity: !hasAstralCodePoint(source),
      featureParityReady: Boolean(fullFeatures),
      featureParity: fullFeatures ? featureMismatch === -1 : null,
      firstFeatureMismatch: fullFeatures ? featureMismatch : -1,
      streamParity: fullFeatures ? streamMismatch === -1 : null,
      firstStreamMismatch: fullFeatures ? streamMismatch : -1,
      featurePrepass: fullFeatures?.timings ?? null,
      featureParityScope: fullFeatures
        ? "ASCII UTF-8 feature encoding from CPU ranges; GPU token-range and safe-segment parity are checked independently"
        : null,
    },
  };
}

async function runPrepass(device, input, units, entryPoint) {
  const totalStarted = performance.now();
  const outputLength = units;
  const inputBuffer = device.createBuffer({ size: alignedSize(input.byteLength), usage: STORAGE | COPY_DST });
  const outputBuffer = device.createBuffer({ size: alignedSize(outputLength * 4), usage: STORAGE | COPY_SRC });
  const paramsBuffer = device.createBuffer({ size: 16, usage: 64 | COPY_DST });
  const readBuffer = device.createBuffer({ size: alignedSize(outputLength * 4), usage: MAP_READ | COPY_DST });
  try {
    const initialized = performance.now();
    const pipeline = await cachedPipeline(device, BOUNDARY_PIPELINES, RAW_PREPASS_SHADER, entryPoint);
    const initializationMs = performance.now() - initialized;
    const uploadStarted = performance.now();
    device.queue.writeBuffer(inputBuffer, 0, input);
    device.queue.writeBuffer(paramsBuffer, 0, Uint32Array.of(units, units, Number(entryPoint === "utf8"), 0));
    const uploadMs = performance.now() - uploadStarted;
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputLength / 256));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputLength * 4);
    const gpuStarted = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - gpuStarted;
    const mapStarted = performance.now();
    await readBuffer.mapAsync(1);
    const tape = new Uint32Array(readBuffer.getMappedRange(), 0, outputLength).slice();
    const mapMs = performance.now() - mapStarted;
    return { tape, timings: { initializationMs, uploadMs, gpuMs, mapMs, totalMs: performance.now() - totalStarted } };
  } finally {
    if (readBuffer.mapState === "mapped") readBuffer.unmap();
    inputBuffer.destroy(); outputBuffer.destroy(); paramsBuffer.destroy(); readBuffer.destroy();
  }
}

async function runUtf8Features(device, input, byteLength, ranges, streams) {
  const totalStarted = performance.now();
  const tokenCount = ranges.length / 2;
  const buffers = {
    input: device.createBuffer({ size: alignedSize(input.byteLength), usage: STORAGE | COPY_DST }),
    ranges: device.createBuffer({ size: ranges.byteLength, usage: STORAGE | COPY_DST }),
    streams: device.createBuffer({ size: Math.max(8, ranges.length * 4), usage: STORAGE | COPY_SRC }),
    stats: device.createBuffer({ size: tokenCount * FEATURE_STATS * 4, usage: STORAGE }),
    features: device.createBuffer({ size: tokenCount * FEATURE_STRIDE * 4, usage: STORAGE | COPY_SRC }),
    document: device.createBuffer({ size: 8, usage: STORAGE | COPY_SRC }),
    params: device.createBuffer({ size: 16, usage: UNIFORM | COPY_DST }),
    read: device.createBuffer({ size: tokenCount * FEATURE_STRIDE * 4, usage: MAP_READ | COPY_DST }),
    streamRead: device.createBuffer({ size: Math.max(8, ranges.length * 4), usage: MAP_READ | COPY_DST }),
    documentRead: device.createBuffer({ size: 8, usage: MAP_READ | COPY_DST }),
  };
  try {
    const initialized = performance.now();
    const [tokenPipeline, documentPipeline, segmentPipeline, finishPipeline] = await Promise.all([
      cachedPipeline(device, FEATURE_PIPELINES, UTF8_FEATURE_SHADER, "token_features"),
      cachedPipeline(device, FEATURE_PIPELINES, UTF8_FEATURE_SHADER, "document_profile"),
      cachedPipeline(device, FEATURE_PIPELINES, UTF8_FEATURE_SHADER, "segment_tokens"),
      cachedPipeline(device, FEATURE_PIPELINES, UTF8_FEATURE_SHADER, "finish_features"),
    ]);
    const initializationMs = performance.now() - initialized;
    const uploadStarted = performance.now();
    device.queue.writeBuffer(buffers.input, 0, input);
    device.queue.writeBuffer(buffers.ranges, 0, ranges);
    device.queue.writeBuffer(buffers.params, 0, Uint32Array.of(tokenCount, byteLength, streams.length / 2, 0));
    const uploadMs = performance.now() - uploadStarted;
    const encoder = device.createCommandEncoder();
    dispatch(encoder, tokenPipeline, featureBindings(device, tokenPipeline, buffers, [0, 1, 3, 4, 6]), Math.ceil(tokenCount / 256));
    dispatch(encoder, documentPipeline, featureBindings(device, documentPipeline, buffers, [3, 4, 5, 6]), 1);
    dispatch(encoder, segmentPipeline, featureBindings(device, segmentPipeline, buffers, [0, 1, 2, 5, 6]), 1);
    dispatch(encoder, finishPipeline, featureBindings(device, finishPipeline, buffers, [0, 1, 2, 3, 4, 5, 6]), Math.ceil(tokenCount / 256));
    encoder.copyBufferToBuffer(buffers.features, 0, buffers.read, 0, tokenCount * FEATURE_STRIDE * 4);
    encoder.copyBufferToBuffer(buffers.streams, 0, buffers.streamRead, 0, Math.max(8, ranges.length * 4));
    encoder.copyBufferToBuffer(buffers.document, 0, buffers.documentRead, 0, 8);
    const gpuStarted = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - gpuStarted;
    const mapStarted = performance.now();
    await Promise.all([buffers.read.mapAsync(1), buffers.streamRead.mapAsync(1), buffers.documentRead.mapAsync(1)]);
    const features = new Uint32Array(buffers.read.getMappedRange(), 0, tokenCount * FEATURE_STRIDE).slice();
    const document = new Uint32Array(buffers.documentRead.getMappedRange(), 0, 2);
    const gpuStreams = new Uint32Array(buffers.streamRead.getMappedRange(), 0, document[1]).slice();
    const mapMs = performance.now() - mapStarted;
    return { features, streams: gpuStreams, timings: { initializationMs, uploadMs, gpuMs, mapMs, totalMs: performance.now() - totalStarted } };
  } finally {
    for (const name of ["read", "streamRead", "documentRead"]) {
      if (buffers[name].mapState === "mapped") buffers[name].unmap();
    }
    for (const buffer of Object.values(buffers)) buffer.destroy();
  }
}

function featureBindings(device, pipeline, buffers, indices) {
  const names = ["input", "ranges", "streams", "stats", "features", "document", "params"];
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: indices.map((index) => ({ binding: index, resource: { buffer: buffers[names[index]] } })),
  });
}

function dispatch(encoder, pipeline, bindGroup, workgroups) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
}

function cachedPipeline(device, namespace, code, entryPoint) {
  let cache = rawPipelineCache.get(device);
  if (!cache) { cache = []; rawPipelineCache.set(device, cache); }
  let group = cache[namespace];
  if (!group) group = cache[namespace] = [device.createShaderModule({ code }), new Map()];
  const pipelines = group[1];
  if (!pipelines.has(entryPoint)) {
    const descriptor = { layout: "auto", compute: { module: group[0], entryPoint } };
    pipelines.set(entryPoint, device.createComputePipelineAsync
      ? device.createComputePipelineAsync(descriptor)
      : device.createComputePipeline(descriptor));
  }
  return pipelines.get(entryPoint);
}

export function rangesFromTape(tape, sparse) {
  const ranges = [];
  let from = 0;
  let offset = 0;
  let opened = false;
  let openedKind = 0;
  for (let index = 0; index < tape.length; index++) {
    const item = tape[index];
    if (sparse && item === 0) continue;
    const boundary = (item & 256) !== 0;
    if (boundary && opened) {
      if (openedKind !== 0) ranges.push(from, offset);
      from = offset;
    }
    opened = true;
    if (boundary) openedKind = item & 255;
    offset += item >>> 16;
  }
  if (opened && openedKind !== 0) ranges.push(from, offset);
  return ranges;
}

function asWords(bytes) {
  const result = new Uint32Array(Math.ceil(bytes.length / 4));
  new Uint8Array(result.buffer).set(bytes);
  return result;
}

function grow(current, minimum, Type = Uint32Array) {
  if (current.length >= minimum) return current;
  let capacity = 256;
  while (capacity < minimum) capacity *= 2;
  return new Type(capacity);
}

function alignedSize(bytes) { return Math.max(4, Math.ceil(bytes / 4) * 4); }
function arraysEqual(left, right) { return left.length === right.length && left.every((value, index) => value === right[index]); }
function firstMismatch(left, right) {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index++) if (left[index] !== right[index]) return index;
  return -1;
}
function hasAstralCodePoint(source) { return /[\uD800-\uDBFF][\uDC00-\uDFFF]/.test(source); }
function isAsciiSource(source) {
  for (let index = 0; index < source.length; index++) if (source.charCodeAt(index) > 127) return false;
  return true;
}
function hasLoneSurrogate(source) {
  for (let index = 0; index < source.length; index++) {
    const code = source.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = source.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true;
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) return true;
  }
  return false;
}

export const __testing = Object.freeze({ hasLoneSurrogate, hasAstralCodePoint, cachedPipeline });

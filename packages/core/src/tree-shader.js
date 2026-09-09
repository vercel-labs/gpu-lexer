import { runtimeTensorLayout, validTreeContext } from "./model-layout.js";
import {
  TREE_SECONDARY_HASH_BUCKETS, TREE_SYMBOL_PAIR_BUCKETS,
  isTreeFeatureVersion, runtimeTreeFeatureLayout, treeInputSize, treeScaleBuckets,
} from "./tree-features.js";

const BLOCK = 32;
const HIDDEN = 32;
const HEAP = BLOCK * 2 - 1;
const CLASSIFIER_TOKENS = 8;
const CLASSIFIER_LANES = 8;

export function createTreeShader(model, { f16 = false } = {}) {
  const hidden = model.hiddenSize;
  const classifier = model.architecture.classifierDimensions;
  const auxiliary = model.architecture.auxiliaryStates.length;
  const outputs = model.outputSize;
  const hashBuckets = model.architecture.lexemeHashBuckets ?? 128;
  const featureVersion = model.featureVersion;
  const scaleBuckets = model.architecture.scaleBuckets;
  const featureLayout = runtimeTreeFeatureLayout(hashBuckets, featureVersion);
  if (!validTreeContext(model) || model.model !== "hierarchical-tree" ||
      !isTreeFeatureVersion(featureVersion) || model.inputSize !== treeInputSize(hashBuckets, featureVersion) ||
      hidden !== HIDDEN || classifier < 1 || classifier > 256 ||
      model.architecture.tree !== "scale-aware-butterfly-binary" ||
      model.architecture.blockParts !== BLOCK || scaleBuckets !== treeScaleBuckets(featureVersion) ||
      (featureVersion >= 3 &&
        (model.architecture.secondaryLexemeHashBuckets !== TREE_SECONDARY_HASH_BUCKETS ||
         model.architecture.neighborSymbolHashBuckets !== TREE_SYMBOL_PAIR_BUCKETS))) {
    throw new Error("promoted hierarchical-tree model is incompatible with this runtime");
  }
  const tensors = Object.fromEntries(runtimeTensorLayout(model).map((value) => [value.name, value]));
  const offset = (name) => {
    if (!tensors[name]) throw new Error(`promoted hierarchical-tree model is missing ${name}`);
    return tensors[name].offset;
  };
  const contextType = f16 ? "f16" : "f32";
  const cast = (value) => f16 ? `f16(${value})` : value;
  const read = (value) => f16 ? `f32(${value})` : value;
  const quantize = (value) => f16 ? `f32(f16(${value}))` : value;
  const enable = f16 ? "enable f16;\n" : "";
  const partnerDepth = Math.log2(hidden) - 1;
  const hashFeature = hashBuckets ? `
  if (kind == 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.primaryHash}u + (((packed >> 19u) & 255u) & ${hashBuckets - 1}u)) * ${hidden}u + h];${featureVersion >= 3 ? `
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.secondaryHash}u + ((context >> 15u) & 127u)) * ${hidden}u + h];` : ""}
  }` : "";
  const symbolPairFeatures = featureVersion >= 3 ? `
  let previous_symbol_pair = (context >> 22u) & 31u;
  let next_symbol_pair = (context >> 27u) & 31u;
  if (previous_symbol_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.previousSymbolPair}u + previous_symbol_pair - 1u) * ${hidden}u + h];
  }
  if (next_symbol_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.nextSymbolPair}u + next_symbol_pair - 1u) * ${hidden}u + h];
  }` : "";

  return /* wgsl */ `${enable}
struct Params { stream_count: u32, token_count: u32, block_count: u32, tree_nodes: u32, }
struct TreeStream { start: u32, count: u32, block_start: u32, block_count: u32, tree_offset: u32, tree_power: u32, }
struct TreeBlock { start: u32, count: u32, stream: u32, local_index: u32, }
@group(0) @binding(0) var<storage, read> features: array<u32>;
@group(0) @binding(1) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> streams: array<TreeStream>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> leaf_states: array<${contextType}>;
@group(0) @binding(6) var<storage, read_write> tree_up: array<${contextType}>;
@group(0) @binding(7) var<storage, read> reserved_7: array<u32>;
@group(0) @binding(8) var<storage, read> reserved_8: array<u32>;
@group(0) @binding(9) var<storage, read> blocks: array<TreeBlock>;
@group(0) @binding(10) var<storage, read_write> hybrid_local: array<f32>;
@group(0) @binding(11) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(12) var<storage, read_write> hybrid_neighbors: array<vec2<u32>>;
var<workgroup> hybrid_work: array<vec4<f32>, ${BLOCK * hidden}>;
var<workgroup> local_tree: array<f32, ${HEAP * hidden}>;
var<workgroup> classifier_auxiliary: array<f32, ${CLASSIFIER_TOKENS * auxiliary}>;
var<workgroup> classifier_projected: array<f32, ${CLASSIFIER_TOKENS * classifier}>;
var<workgroup> classifier_scores: array<f32, ${CLASSIFIER_TOKENS * outputs}>;

fn flat_group(id: vec3<u32>) -> u32 { return id.x + id.y * 65535u; }
fn flat_id(id: vec3<u32>) -> u32 { return id.x + id.y * 4194240u; }
fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn raw_state(token: u32, start: u32, end: u32, h: u32) -> f32 {
  if (token < start || token >= end) { return 0.0; }
  let packed = features[token * 2u];
  let context = features[token * 2u + 1u];
  let kind = packed & 3u;
  var sum = weights[${offset("featureEmbedding")}u + kind * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (4u + ((packed >> 2u) & 7u)) * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (12u + ((packed >> 5u) & 127u)) * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (140u + ((packed >> 12u) & 127u)) * ${hidden}u + h];${hashFeature}
  let flag_offset = ${featureLayout.flags}u;
  let flags = context & 127u;
  for (var bit = 0u; bit < 7u; bit++) {
    if ((flags & (1u << bit)) != 0u) {
      sum += weights[${offset("featureEmbedding")}u + (flag_offset + bit) * ${hidden}u + h];
    }
  }
  let previous_pair = (context >> 7u) & 15u;
  let next_pair = (context >> 11u) & 15u;
  if (previous_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.previousPair}u + previous_pair - 1u) * ${hidden}u + h];
  }
  if (next_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.nextPair}u + next_pair - 1u) * ${hidden}u + h];
  }${symbolPairFeatures}
  return sum;
}

fn local_state(token: u32, h: u32, start: u32, end: u32, neighbors: vec2<f32>) -> f32 {
  var sum = weights[${offset("leafBias")}u + h];
  for (var d = 0u; d < 5u; d++) {
    if (token + d >= start + 2u && token + d < end + 2u) {
      sum += raw_state(token + d - 2u, start, end, h) *
        weights[${offset("localOffsetScale")}u + d * ${hidden}u + h];
    }
  }
  sum += raw_state(bitcast<u32>(neighbors.x), start, end, h) * weights[${offset("localNonspaceScale")}u + h];
  sum += raw_state(bitcast<u32>(neighbors.y), start, end, h) * weights[${offset("localNonspaceScale") + hidden}u + h];
  return tanh(sum);
}

fn affine_at(part: u32, h: u32) -> vec2<f32> {
  var input = weights[${offset("stateInputBias")}u + h];
  var gate = weights[${offset("stateGateBias")}u + h];
  for (var c = 0u; c < ${hidden}u; c++) {
    let value = hybrid_work[part * ${hidden}u + c].x;
    input += value * weights[${offset("stateInput")}u + h * ${hidden}u + c];
    gate += value * weights[${offset("stateGate")}u + h * ${hidden}u + c];
  }
  let a = sigmoid(gate);
  return vec2<f32>(a, (1.0 - a) * tanh(input));
}

fn merge_state(left: f32, right: f32, left_cross: f32, right_cross: f32, h: u32, depth: u32) -> f32 {
  let scale = min(depth, ${scaleBuckets - 1}u) * ${hidden}u + h;
  let mixed = tanh(
    left * weights[${offset("mergeOwnLeft")}u + scale] +
    right * weights[${offset("mergeOwnRight")}u + scale] +
    left_cross * weights[${offset("mergeCrossLeft")}u + scale] +
    right_cross * weights[${offset("mergeCrossRight")}u + scale] +
    weights[${offset("mergeBias")}u + scale]
  );
  let boundary = select(right, left, h < ${hidden / 2}u);
  return (mixed + boundary) * 0.5;
}

fn descend_state(
  parent: f32, own: f32, sibling: f32,
  parent_cross: f32, own_cross: f32, sibling_cross: f32,
  h: u32, depth: u32, right: bool,
) -> f32 {
  let scale = min(depth, ${scaleBuckets - 1}u) * ${hidden}u + h;
  let bias = select(weights[${offset("downLeftBias")}u + scale], weights[${offset("downRightBias")}u + scale], right);
  let mixed = tanh(
    parent * weights[${offset("downOwnParent")}u + scale] +
    own * weights[${offset("downOwnSelf")}u + scale] +
    sibling * weights[${offset("downOwnSibling")}u + scale] +
    parent_cross * weights[${offset("downCrossParent")}u + scale] +
    own_cross * weights[${offset("downCrossSelf")}u + scale] +
    sibling_cross * weights[${offset("downCrossSibling")}u + scale] + bias
  );
  let skip = sigmoid(weights[${offset("downSkip")}u + scale]);
  return parent * skip + mixed * (1.0 - skip);
}

@compute @workgroup_size(64)
fn hybrid_neighbor_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = flat_id(id);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  var first = 0xffffffffu;
  var last = 0xffffffffu;
  for (var i = 0u; i < block.count; i++) {
    let token = block.start + i;
    let kind = features[token * 2u] & 3u;
    if (kind != 1u && kind != 2u) {
      if (first == 0xffffffffu) { first = token; }
      last = token;
    }
  }
  hybrid_neighbors[index] = vec2<u32>(first, last);
}

@compute @workgroup_size(64)
fn hybrid_neighbor_prefixes(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = flat_id(id);
  if (index >= params.stream_count) { return; }
  let stream = streams[index];
  var previous = 0xffffffffu;
  for (var i = 0u; i < stream.block_count; i++) {
    let b = stream.block_start + i;
    let own = hybrid_neighbors[b];
    hybrid_neighbors[b].y = previous;
    if (own.y != 0xffffffffu) { previous = own.y; }
  }
  var next = 0xffffffffu;
  for (var i = stream.block_count; i > 0u; i--) {
    let b = stream.block_start + i - 1u;
    let own = hybrid_neighbors[b].x;
    hybrid_neighbors[b].x = next;
    if (own != 0xffffffffu) { next = own; }
  }
}

@compute @workgroup_size(${hidden})
fn hybrid_scan_blocks(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  let stream = streams[block.stream];
  let h = local.x;
  if (h < block.count) {
    let token = block.start + h;
    var previous = hybrid_neighbors[index].y;
    var next = hybrid_neighbors[index].x;
    for (var i = token; i > block.start; i--) {
      let kind = features[(i - 1u) * 2u] & 3u;
      if (kind != 1u && kind != 2u) { previous = i - 1u; break; }
    }
    for (var i = token + 1u; i < block.start + block.count; i++) {
      let kind = features[i * 2u] & 3u;
      if (kind != 1u && kind != 2u) { next = i; break; }
    }
    hybrid_work[h * ${hidden}u].y = bitcast<f32>(previous);
    hybrid_work[h * ${hidden}u].z = bitcast<f32>(next);
  }
  workgroupBarrier();
  for (var i = 0u; i < block.count; i++) {
    let value = local_state(block.start + i, h, stream.start, stream.start + stream.count,
      hybrid_work[i * ${hidden}u].yz);
    hybrid_local[(block.start + i) * ${hidden}u + h] = value;
    hybrid_work[i * ${hidden}u + h].x = value;
  }
  workgroupBarrier();
  for (var i = 0u; i < block.count; i++) {
    let pair = affine_at(i, h);
    hybrid_work[i * ${hidden}u + h].y = pair.x;
    hybrid_work[i * ${hidden}u + h].z = pair.y;
  }
  var forward = vec2<f32>(1.0, 0.0);
  for (var i = 0u; i < block.count; i++) {
    let value = hybrid_work[i * ${hidden}u + h];
    forward = vec2<f32>(value.y * forward.x, value.y * forward.y + value.z);
  }
  var reverse = vec2<f32>(1.0, 0.0);
  for (var i = block.count; i > 0u; i--) {
    let value = hybrid_work[(i - 1u) * ${hidden}u + h];
    reverse = vec2<f32>(value.y * reverse.x, value.y * reverse.y + value.z);
  }
  let at = (index * ${hidden}u + h) * 4u;
  scratch[at] = forward.x;
  scratch[at + 1u] = forward.y;
  scratch[at + 2u] = reverse.x;
  scratch[at + 3u] = reverse.y;
}

@compute @workgroup_size(${hidden * 8})
fn hybrid_scan_prefixes(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  let stream_index = index >> 2u;
  if (stream_index >= params.stream_count) { return; }
  let stream = streams[stream_index];
  let lane = local.x & 31u;
  let h = (index & 3u) * 8u + (local.x >> 5u);
  var forward_carry = 0.0;
  var reverse_carry = 0.0;
  for (var chunk = 0u; chunk < stream.block_count; chunk += 32u) {
    let forward_index = chunk + lane;
    let reverse_index = stream.block_count - 1u - forward_index;
    let valid = forward_index < stream.block_count;
    let forward_at = ((stream.block_start + forward_index) * ${hidden}u + h) * 4u;
    let reverse_at = ((stream.block_start + reverse_index) * ${hidden}u + h) * 4u;
    var value = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    if (valid) {
      value = vec4<f32>(scratch[forward_at], scratch[forward_at + 1u],
        scratch[reverse_at + 2u], scratch[reverse_at + 3u]);
    }
    hybrid_work[local.x] = value;
    for (var step = 1u; step < 32u; step = step << 1u) {
      workgroupBarrier();
      var previous = vec4<f32>(1.0, 0.0, 1.0, 0.0);
      if (lane >= step) { previous = hybrid_work[local.x - step]; }
      workgroupBarrier();
      value = vec4<f32>(value.x * previous.x, value.x * previous.y + value.y,
        value.z * previous.z, value.z * previous.w + value.w);
      hybrid_work[local.x] = value;
    }
    workgroupBarrier();
    var before = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    if (lane > 0u) { before = hybrid_work[local.x - 1u]; }
    if (valid) {
      scratch[forward_at + 1u] = before.x * forward_carry + before.y;
      scratch[reverse_at + 3u] = before.z * reverse_carry + before.w;
    }
    let total = hybrid_work[local.x | 31u];
    forward_carry = total.x * forward_carry + total.y;
    reverse_carry = total.z * reverse_carry + total.w;
    workgroupBarrier();
  }
}

@compute @workgroup_size(${hidden})
fn hybrid_mix_tree_up(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  let stream = streams[block.stream];
  let h = local.x;
  for (var i = 0u; i < block.count; i++) {
    hybrid_work[i * ${hidden}u + h].x = hybrid_local[(block.start + i) * ${hidden}u + h];
  }
  workgroupBarrier();
  for (var i = 0u; i < block.count; i++) {
    let pair = affine_at(i, h);
    hybrid_work[i * ${hidden}u + h].y = pair.x;
    hybrid_work[i * ${hidden}u + h].z = pair.y;
  }
  let summary = (index * ${hidden}u + h) * 4u;
  var state = scratch[summary + 1u];
  for (var i = 0u; i < block.count; i++) {
    let at = i * ${hidden}u + h;
    let value = hybrid_work[at];
    state = value.y * state + value.z;
    hybrid_work[at].w = state;
  }
  state = scratch[summary + 3u];
  for (var i = block.count; i > 0u; i--) {
    let at = (i - 1u) * ${hidden}u + h;
    let value = hybrid_work[at];
    state = value.y * state + value.z;
    hybrid_work[at].z = state;
  }
  workgroupBarrier();
  for (var i = 0u; i < block.count; i++) {
    let token = block.start + i;
    var sum = hybrid_local[token * ${hidden}u + h] + weights[${offset("stateMixBias")}u + h];
    let row = ${offset("stateMix")}u + h * ${hidden * 2}u;
    for (var c = 0u; c < ${hidden}u; c++) {
      let context = hybrid_work[i * ${hidden}u + c];
      sum += context.w * weights[row + c] + context.z * weights[row + ${hidden}u + c];
    }
    let leaf = ${quantize("tanh(sum)")};
    hybrid_work[i * ${hidden}u + h].y = leaf;
    leaf_states[token * ${hidden}u + h] = ${cast("leaf")};
  }
  workgroupBarrier();
  var width = ${BLOCK}u;
  var valid = block.count;
  var depth = 0u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    for (var parent = 0u; parent < parent_width; parent++) {
      let child = parent * 2u;
      var merged = 0.0;
      if (child < valid) {
        let left_at = child * ${hidden}u;
        let input_y = (depth & 1u) == 0u;
        let left = select(hybrid_work[left_at + h].x, hybrid_work[left_at + h].y, input_y);
        merged = left;
        if (child + 1u < valid) {
          let right_at = (child + 1u) * ${hidden}u;
          let partner = h ^ (1u << depth);
          merged = merge_state(left,
            select(hybrid_work[right_at + h].x, hybrid_work[right_at + h].y, input_y),
            select(hybrid_work[left_at + partner].x, hybrid_work[left_at + partner].y, input_y),
            select(hybrid_work[right_at + partner].x, hybrid_work[right_at + partner].y, input_y), h, depth);
        }
      }
      if ((depth & 1u) == 0u) { hybrid_work[parent * ${hidden}u + h].x = merged; }
      else { hybrid_work[parent * ${hidden}u + h].y = merged; }
    }
    workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  let node = stream.tree_offset + stream.tree_power - 1u + block.local_index;
  tree_up[node * ${hidden}u + h] = ${cast("hybrid_work[h].x")};
}

@compute @workgroup_size(${hidden * 8})
fn tree_global(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let stream_index = flat_group(group);
  let h = local.x & ${hidden - 1}u;
  let node_lane = local.x >> 5u;
  if (stream_index >= params.stream_count) { return; }
  let stream = streams[stream_index];
  let leaf_base = stream.tree_offset + stream.tree_power - 1u;
  for (var index = stream.block_count + node_lane; index < stream.tree_power; index += 8u) {
    tree_up[(leaf_base + index) * ${hidden}u + h] = ${cast("0.0")};
  }
  storageBarrier(); workgroupBarrier();
  var width = stream.tree_power;
  var valid = stream.block_count;
  var depth = 5u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    let parent_base = stream.tree_offset + parent_width - 1u;
    let child_base = stream.tree_offset + width - 1u;
    let partner = h ^ (1u << min(depth, ${partnerDepth}u));
    for (var index = node_lane; index < parent_width; index += 8u) {
      let child = index * 2u;
      var value = 0.0;
      if (child < valid) {
        let left = ${read(`tree_up[(child_base + child) * ${hidden}u + h]`)};
        value = left;
        if (child + 1u < valid) {
          value = merge_state(left, ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + h]`)},
            ${read(`tree_up[(child_base + child) * ${hidden}u + partner]`)},
            ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + partner]`)}, h, depth);
        }
      }
      tree_up[(parent_base + index) * ${hidden}u + h] = ${cast("value")};
    }
    storageBarrier(); workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  if (node_lane == 0u) {
    scratch[stream.tree_offset * ${hidden}u + h] = ${quantize(read(`tree_up[stream.tree_offset * ${hidden}u + h]`))};
  }
  storageBarrier(); workgroupBarrier();
  width = 1u;
  var down_depth = 4u + (31u - countLeadingZeros(stream.tree_power));
  loop {
    if (width >= stream.tree_power) { break; }
    let parent_base = stream.tree_offset + width - 1u;
    let child_base = stream.tree_offset + width * 2u - 1u;
    let child_span = stream.tree_power / (width * 2u);
    let child_valid = (stream.block_count + child_span - 1u) / child_span;
    for (var index = node_lane; index < width; index += 8u) {
      let child = index * 2u;
      if (child >= child_valid) { continue; }
      let parent = scratch[(parent_base + index) * ${hidden}u + h];
      let left = ${read(`tree_up[(child_base + child) * ${hidden}u + h]`)};
      if (child + 1u < child_valid) {
        let right = ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + h]`)};
        let partner = h ^ (1u << min(down_depth, ${partnerDepth}u));
        let parent_cross = scratch[(parent_base + index) * ${hidden}u + partner];
        let left_cross = ${read(`tree_up[(child_base + child) * ${hidden}u + partner]`)};
        let right_cross = ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + partner]`)};
        scratch[(child_base + child) * ${hidden}u + h] = ${quantize("descend_state(parent, left, right, parent_cross, left_cross, right_cross, h, down_depth, false)")};
        scratch[(child_base + child + 1u) * ${hidden}u + h] = ${quantize("descend_state(parent, right, left, parent_cross, right_cross, left_cross, h, down_depth, true)")};
      } else {
        scratch[(child_base + child) * ${hidden}u + h] = ${quantize("parent")};
      }
    }
    storageBarrier(); workgroupBarrier();
    width *= 2u;
    down_depth = select(0u, down_depth - 1u, down_depth > 0u);
  }
}

@compute @workgroup_size(64)
fn tree_down_classify(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let block_index = flat_group(group);
  if (block_index >= params.block_count) { return; }
  let block = blocks[block_index];
  let stream = streams[block.stream];
  let tree_lane = local.x < ${hidden}u;
  let tree_channel = local.x;
  if (tree_lane) {
    for (var part = 0u; part < block.count; part++) {
      local_tree[(${BLOCK - 1}u + part) * ${hidden}u + tree_channel] =
        ${read(`leaf_states[(block.start + part) * ${hidden}u + tree_channel]`)};
    }
  }
  workgroupBarrier();
  var width = ${BLOCK}u;
  var valid = block.count;
  var depth = 0u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    let parent_base = parent_width - 1u;
    let child_base = width - 1u;
    if (tree_lane) {
      for (var parent = 0u; parent < parent_width; parent++) {
        let child = parent * 2u;
        var merged = 0.0;
        if (child < valid) {
          let left = local_tree[(child_base + child) * ${hidden}u + tree_channel];
          merged = left;
          if (child + 1u < valid) {
            let partner = tree_channel ^ (1u << depth);
            merged = merge_state(left, local_tree[(child_base + child + 1u) * ${hidden}u + tree_channel],
              local_tree[(child_base + child) * ${hidden}u + partner],
              local_tree[(child_base + child + 1u) * ${hidden}u + partner], tree_channel, depth);
          }
        }
        local_tree[(parent_base + parent) * ${hidden}u + tree_channel] = merged;
      }
    }
    workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  if (tree_lane) {
    let node = stream.tree_offset + stream.tree_power - 1u + block.local_index;
    local_tree[tree_channel] = scratch[node * ${hidden}u + tree_channel];
  }
  workgroupBarrier();
  width = 1u;
  depth = 4u;
  loop {
    if (width >= ${BLOCK}u) { break; }
    let parent_base = width - 1u;
    let child_base = width * 2u - 1u;
    let child_span = ${BLOCK}u / (width * 2u);
    let child_valid = (block.count + child_span - 1u) / child_span;
    if (tree_lane) {
      for (var parent_index = 0u; parent_index < width; parent_index++) {
        let child = parent_index * 2u;
        if (child >= child_valid) { continue; }
        let parent = local_tree[(parent_base + parent_index) * ${hidden}u + tree_channel];
        let left = local_tree[(child_base + child) * ${hidden}u + tree_channel];
        if (child + 1u < child_valid) {
          let right = local_tree[(child_base + child + 1u) * ${hidden}u + tree_channel];
          let partner = tree_channel ^ (1u << depth);
          let parent_cross = local_tree[(parent_base + parent_index) * ${hidden}u + partner];
          let left_cross = local_tree[(child_base + child) * ${hidden}u + partner];
          let right_cross = local_tree[(child_base + child + 1u) * ${hidden}u + partner];
          local_tree[(child_base + child) * ${hidden}u + tree_channel] = descend_state(
            parent, left, right, parent_cross, left_cross, right_cross, tree_channel, depth, false);
          local_tree[(child_base + child + 1u) * ${hidden}u + tree_channel] = descend_state(
            parent, right, left, parent_cross, right_cross, left_cross, tree_channel, depth, true);
        } else {
          local_tree[(child_base + child) * ${hidden}u + tree_channel] = parent;
        }
      }
    }
    workgroupBarrier();
    width *= 2u;
    depth = select(0u, depth - 1u, depth > 0u);
  }

  let slot = local.x / ${CLASSIFIER_LANES}u;
  let lane = local.x % ${CLASSIFIER_LANES}u;
  for (var batch = 0u; batch < ${BLOCK / CLASSIFIER_TOKENS}u; batch++) {
    let part = batch * ${CLASSIFIER_TOKENS}u + slot;
    let token = block.start + part;
    let token_valid = part < block.count;
    let kind = select(1u, features[token * 2u] & 3u, token_valid);
    let styled = token_valid && kind != 1u && kind != 2u;
    for (var a = lane; styled && a < ${auxiliary}u; a += ${CLASSIFIER_LANES}u) {
      var value = weights[${offset("auxiliaryBias")}u + a];
      for (var feature_channel = 0u; feature_channel < ${hidden}u; feature_channel++) {
        let leaf = ${read(`leaf_states[token * ${hidden}u + feature_channel]`)};
        let context = ${quantize(`local_tree[(${BLOCK - 1}u + part) * ${hidden}u + feature_channel]`)};
        let row = ${offset("auxiliaryOutput")}u + a * ${hidden * 2}u;
        value += weights[row + feature_channel] * leaf + weights[row + ${hidden}u + feature_channel] * context;
      }
      classifier_auxiliary[slot * ${auxiliary}u + a] = sigmoid(value);
    }
    workgroupBarrier();
    for (var c = lane; styled && c < ${classifier}u; c += ${CLASSIFIER_LANES}u) {
      var value = weights[${offset("classifierBias")}u + c];
      for (var feature_channel = 0u; feature_channel < ${hidden}u; feature_channel++) {
        let leaf = ${read(`leaf_states[token * ${hidden}u + feature_channel]`)};
        let context = ${quantize(`local_tree[(${BLOCK - 1}u + part) * ${hidden}u + feature_channel]`)};
        let row = ${offset("classifierInput")}u + c * ${hidden * 2 + auxiliary}u;
        value += weights[row + feature_channel] * leaf + weights[row + ${hidden}u + feature_channel] * context;
      }
      let columns = ${offset("classifierInput")}u + c * ${hidden * 2 + auxiliary}u + ${hidden * 2}u;
      for (var a = 0u; a < ${auxiliary}u; a++) {
        value += weights[columns + a] * classifier_auxiliary[slot * ${auxiliary}u + a];
      }
      classifier_projected[slot * ${classifier}u + c] = tanh(value);
    }
    workgroupBarrier();
    for (var output = lane; styled && output < ${outputs}u; output += ${CLASSIFIER_LANES}u) {
      var score = weights[${offset("outputBias")}u + output];
      for (var c = 0u; c < ${classifier}u; c++) {
        score += weights[${offset("output")}u + output * ${classifier}u + c] *
          classifier_projected[slot * ${classifier}u + c];
      }
      classifier_scores[slot * ${outputs}u + output] = score;
    }
    workgroupBarrier();
    if (lane == 0u && token_valid) {
      var selected = 0u;
      if (styled) {
        var selected_score = classifier_scores[slot * ${outputs}u];
        for (var output = 1u; output < ${outputs}u; output++) {
          let score = classifier_scores[slot * ${outputs}u + output];
          if (score > selected_score) { selected_score = score; selected = output; }
        }
      }
      let ignored = atomicOr(&labels[token / 4u], selected << ((token & 3u) * 8u));
    }
    workgroupBarrier();
  }
}
`;
}

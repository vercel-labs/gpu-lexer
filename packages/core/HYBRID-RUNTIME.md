# Tree runtime

The runtime requires format9, `model: "hierarchical-tree"`, and architecture
metadata `context: "local-affine-tree"`, `localRadius: 2`. The
`tree: "scale-aware-butterfly-binary"`, 32-part blocks, the feature-version
scale count, H=32, and packed int4/int5/int6 tensors remain supported.
The published core contains only this architecture and its generated shader.

## Weight contract

All tree matrices are row-major (output dimension first); `stateInput` is not
transposed during GPU weight preparation.
The additional tensors are:

- `localOffsetScale[5,H]`, in offset order -2, -1, 0, +1, +2
- `localNonspaceScale[2,H]`, previous then next
- `stateInput[H,H]`, `stateInputBias[H]`
- `stateGate[H,H]`, `stateGateBias[H]`
- `stateMix[H,2H]`, `stateMixBias[H]`, forward channels then reverse channels

`neighborScale[3,H]` remains required in the training checkpoint schema but is
not read by hybrid WGSL or included in the browser weight payload. All offsets
outside a stream contribute zero. Nearest nonspace excludes
part kinds 1 and 2, excludes the current part, and never crosses a stream
boundary. It can cross any number of blocks or whitespace/newline parts.

Local states are tanh of the weighted raw embedding neighborhood plus leafBias.
The two state projections produce `u=tanh(Winput*local+binput)` and
`a=sigmoid(Wgate*local+bgate)`. Both directional scans are inclusive, initialized
with zero at their respective stream boundary, and apply
`h=a*hprev+(1-a)*u`. Enhanced leaves are
`tanh(local+Wmix*concat(forward,reverse)+bmix)`. The existing upward tree,
downward tree, and classifier then consume those enhanced leaves unchanged.

## Passes and cost

The runtime uses eight passes:

1. Find first/last nonspace part in each 32-part block, in parallel across blocks.
2. Compute exclusive previous/next nonspace block prefixes per stream.
3. Compute local states, one invocation per channel per block. Neighbor
   searching is bounded to the current block; the prefixes supply cross-block
   neighbors.
4. Project u/a and compute inclusive affine-pair scans within each block, one
   invocation per channel per block. Each invocation visits at most 32 parts;
   the reverse scan reuses the original projected affine pairs.
5. Compute exclusive forward/reverse block-prefix states, one invocation per
   channel per stream.
6. Recompute each affine pair once, apply both prefixes, mix forward/reverse
   states, write enhanced leaves, and build each 32-part tree root in the same
   workgroup.
7. Merge block roots and propagate whole-stream context downward.
8. Propagate context through each block and classify eight parts at a time.
   Four byte-sized labels share each readback word.

This is intentionally simple rather than a fully parallel hierarchical prefix
implementation. No invocation scans all parts of a long stream. Prefix work is
still serial over its B=ceil(N/32) block summaries (O(B) depth); a hierarchical
scan could replace that stage for very long inputs. The existing tree's global
pass is unchanged.

For N parts, H channels, B blocks, T tree nodes, and state width S (two bytes
with shader-f16, otherwise four), the main working storage is `4*N*H` bytes for
local states, `S*N*H` for enhanced leaves, `S*T*H` for upward tree states,
`max(16*B*H, 4*T*H)` for shared summary/downward scratch, and `8*B` for block
neighbors. The former token-sized affine-pair, downward-tree, and final-context
buffers no longer exist. Grow-only capacities can exceed these used sizes.
Every entry point fits the baseline eight-storage-buffer-per-stage limit.

## Verification

`node --test test/hybrid-shader.test.js` covers WGSL validation for H=32 and
f32/f16, metadata/tensor rejection, row-major packed int4/int5/int6 weights,
active-format pass selection, binding counts, and buffer reuse/growth.
The runtime tests use a fake WebGPU device; they do not establish numerical
GPU/reference parity. Real-device comparison with the parent's JS reference
is still needed, particularly for partial blocks, long whitespace runs,
multiple streams, and non-symmetric state matrices. No training or promotion
is part of this change.

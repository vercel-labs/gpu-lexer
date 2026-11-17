import { createRuntime } from "./gpu.js";
import { preparePromotedWeights } from "./weights.js";

// Readable access to runtime internals for tests and the benchmark profiler.
// The public build never imports this module.
export function createInspectableRuntime(options = {}) {
  const runtime = createRuntime(
    preparePromotedWeights,
    options.gpu === undefined ? globalThis.navigator?.gpu : options.gpu,
    options.device,
    options.shader,
    options.run,
  );
  return {
    parse: runtime.h,
    initialize: runtime.i,
    run: runtime.r,
    ensureBuffer: runtime.e,
    createTreeBindGroups: runtime.g,
    buffers: runtime.b,
    scanParams: runtime.p,
    get device() { return runtime.d; },
    get useF16() { return runtime.f; },
    get pipelines() { return runtime.c; },
    get treeBindGroups() { return runtime.q; },
    set treeBindGroups(value) { runtime.q = value; },
    get treeLayout() { return runtime.l; },
    set treeLayout(value) { runtime.l = value; },
  };
}

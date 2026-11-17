import { createRuntime } from "./gpu.js";
// Imported after the runtime so that the weights end the bundle; see gpu.js.
import { preparePromotedWeights } from "./weights.js";

let runtime;

export async function parse(code) {
  return (runtime ??= createRuntime(preparePromotedWeights)).h(code);
}

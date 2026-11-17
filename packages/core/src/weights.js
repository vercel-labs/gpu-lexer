import { WEIGHT_RANGES, WEIGHTS } from "./model.runtime.generated.js";

/** Expand the promoted int6 weight symbols into the float values the shader reads. */
export function preparePromotedWeights() {
  const result = new Float32Array(WEIGHTS.length);
  let start = 0;
  for (let range = 0; range < WEIGHT_RANGES.length; range += 2) {
    const end = start + WEIGHT_RANGES[range];
    const scale = WEIGHT_RANGES[range + 1];
    for (let index = start; index < end; index++) {
      const code = WEIGHTS.charCodeAt(index);
      const value = code >= 97 ? code - 71 : code >= 65 ? code - 65 : code >= 48 ? code + 4 : code === 45 ? 62 : 63;
      result[index] = (value & 1 ? -(value + 1) / 2 : value / 2) * scale;
    }
    start = end;
  }
  return result;
}

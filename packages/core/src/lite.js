import { createParse } from "./lex/runtime.js";
import { tokenize } from "./lex/lite/tokenizer.js";
import { META, PIPELINE, WEIGHTS_F16_B85, WEIGHTS_SYM } from "./lex/lite/weights.js";
import shader from "./lex/lite/shader.wgsl";

export const parse = /* @__PURE__ */ createParse({
  tokenize, shader, meta: META, pipeline: PIPELINE, sym: WEIGHTS_SYM, f16: WEIGHTS_F16_B85,
});

import { createRuntime } from "./gpu.js";

let runtime;

export async function parse(code) {
  return (runtime ??= createRuntime()).h(code);
}

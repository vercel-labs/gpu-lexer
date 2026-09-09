import { parse } from "gpu-lexer";

self.onmessage = async ({ data: source }) => {
  try {
    await parse(source);
    const started = performance.now();
    await parse(source);
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};

globalThis.Prism = { disableWorkerMessageHandler: true };
const { default: Prism } = await import("prismjs");

self.addEventListener("message", ({ data: source }) => {
  try {
    Prism.highlight(source, Prism.languages.javascript, "javascript");
    const started = performance.now();
    Prism.highlight(source, Prism.languages.javascript, "javascript");
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
});

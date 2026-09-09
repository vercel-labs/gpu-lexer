import hljs from "highlight.js";

self.onmessage = ({ data: source }) => {
  try {
    hljs.highlight(source, { language: "javascript", ignoreIllegals: true });
    const started = performance.now();
    hljs.highlight(source, { language: "javascript", ignoreIllegals: true });
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};

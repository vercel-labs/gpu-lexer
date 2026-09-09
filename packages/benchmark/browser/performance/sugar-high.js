import { highlight } from "sugar-high";

self.onmessage = ({ data: source }) => {
  try {
    highlight(source, { lang: "javascript" });
    const started = performance.now();
    highlight(source, { lang: "javascript" });
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};

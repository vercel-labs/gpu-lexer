import sourceJs from "@wooorm/starry-night/source.js";
import { createStarryNight } from "@wooorm/starry-night";

const highlighter = createStarryNight([sourceJs]);

self.onmessage = async ({ data: source }) => {
  try {
    const loaded = await highlighter;
    loaded.highlight(source, "source.js");
    const started = performance.now();
    loaded.highlight(source, "source.js");
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};

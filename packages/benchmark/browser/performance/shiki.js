import { createHighlighter } from "shiki";

const highlighter = createHighlighter({
  langs: ["javascript"],
  themes: ["github-light"],
});

self.onmessage = async ({ data: source }) => {
  try {
    const loaded = await highlighter;
    loaded.codeToTokens(source, { lang: "javascript", theme: "github-light" });
    const started = performance.now();
    loaded.codeToTokens(source, { lang: "javascript", theme: "github-light" });
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};

import { readFile, writeFile } from "node:fs/promises";
import { extname } from "node:path";

import { createLabeler } from "./label.js";

const extensionLanguages = new Map([
  [".js", "javascript"],
  [".jsx", "jsx"],
  [".ts", "typescript"],
  [".tsx", "tsx"],
  [".py", "python"],
  [".rs", "rust"],
  [".go", "go"],
  [".css", "css"],
  [".html", "html"],
]);

const [input, output, explicitLanguage] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: pnpm label <input-file> <output-json> [language]");
  process.exitCode = 1;
} else {
  const language = explicitLanguage ?? extensionLanguages.get(extname(input));
  if (!language) throw new Error("language is required for this file extension");
  const code = await readFile(input, "utf8");
  const labeler = await createLabeler({ langs: [language] });
  try {
    const sourceLabels = labeler.labelSource(code, language);
    await writeFile(output, `${JSON.stringify({ language, source: input, sourceLabelsVersion: 1, sourceLabels })}\n`);
  } finally {
    labeler.dispose();
  }
}

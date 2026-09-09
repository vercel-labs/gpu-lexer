import { createHighlighter } from "shiki";

import { auxiliaryFromScopes, classFromScopes, confidenceFromScopes } from "./classes.js";
import { prepareTreeSource, releaseTreePrepared } from "../../core/src/prepare-tree.js";
import { TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE } from "../../core/src/constants.js";
import { alignTreeLabels } from "./tree-label-alignment.js";

export async function createLabeler({ langs, themes = ["github-dark-default"] }) {
  const highlighter = await createHighlighter({ langs, themes });

  function labelSource(code, language) {
    const themed = highlighter.codeToTokens(code, {
      lang: language,
      theme: themes[0],
      includeExplanation: true,
    });
    return sourceLabelsFromTokens(code, themed.tokens);
  }

  return {
    dispose() {
      highlighter.dispose();
    },
    labelSource,
  };
}

/** The corpus quota unit: one non-whitespace runtime part with its direct
 * Shiki projection. Runtime whitespace parts are restored from source text.
 */
export function sourceParts(code, sourceLabels) {
  const prepared = prepareTreeSource(code);
  try {
    const [data, ranges, , count] = prepared;
    const labels = alignTreeLabels(sourceLabels, ranges, { sourceLength: code.length });
    const parts = [];
    for (let index = 0; index < count; index++) {
      const kind = data[index * TREE_FEATURE_STRIDE] & 3;
      if (kind === TREE_PART_SPACE || kind === TREE_PART_NEWLINE) continue;
      const label = labels[index];
      parts.push({
        from: label.from, to: label.to, kind, value: code.slice(label.from, label.to),
        class: label.class, auxiliary: label.auxiliary, confidence: label.confidence,
      });
    }
    return parts;
  } finally {
    releaseTreePrepared(prepared);
  }
}

/** Original-source UTF-16, half-open spans; confidence is a float in [0, 1].
 * Shiki omits line separators. Leave these uncovered rather than inventing scopes.
 * Reject malformed explanations instead of silently shifting all later labels.
 */
export function sourceLabelsFromTokens(code, lines) {
  if (typeof code !== "string" || !Array.isArray(lines) || lines.some((line) => !Array.isArray(line))) {
    throw new TypeError("Shiki tokens must be lines of tokens for source text");
  }
  const labels = [];
  let previousTo = 0;
  for (const line of lines) for (const token of line) {
    if (!token || typeof token.content !== "string") throw new Error("malformed Shiki token content");
    let from = token.offset;
    if (!Number.isInteger(from) || from < previousTo || from + token.content.length > code.length ||
        code.slice(from, from + token.content.length) !== token.content) {
      throw new Error(`Shiki token/source offset mismatch at ${from}`);
    }
    if (!Array.isArray(token.explanation) || (token.content.length && !token.explanation.length) ||
        token.explanation.some((item) => !item || typeof item.content !== "string" ||
          !Array.isArray(item.scopes) || item.scopes.some((scope) => typeof scope?.scopeName !== "string"))) {
      throw new Error(`missing or malformed Shiki explanation at ${from}`);
    }
    const explanation = token.explanation;
    if (explanation.map((item) => item.content).join("") !== token.content) {
      throw new Error(`Shiki explanation does not partition token at ${from}`);
    }
    for (const item of explanation) {
      const to = from + item.content.length;
      const scopes = (item.scopes ?? []).map((scope) => scope.scopeName);
      if (to > from) labels.push({
        from, to, class: classFromScopes(scopes),
        auxiliary: auxiliaryFromScopes(scopes), confidence: confidenceFromScopes(scopes),
      });
      from = to;
    }
    previousTo = from;
  }
  return labels;
}

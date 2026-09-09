export const requiredSyntaxConstructs = Object.freeze([
  "multiline-comment",
  "multiline-string",
  "preprocessor-directive",
  "embedded-language",
  "markup-comment",
  "interpolation",
  "escape-sequence",
  "heredoc",
  "css-selector",
  "css-declaration",
  "diff-hunk",
  "line-continuation",
]);

export function detectSyntaxConstructs(source, language = "", family = language) {
  const constructs = [];
  const add = (name, present) => { if (present) constructs.push(name); };
  add("multiline-comment", /\/\*[\s\S]*?\n[\s\S]*?\*\/|<!--[\s\S]*?\n[\s\S]*?-->/.test(source));
  add("multiline-string", /(?:"""|''')[\s\S]*?\n[\s\S]*?(?:"""|''')|`[^`]*\n[^`]*`/.test(source));
  add("preprocessor-directive", /^[\t ]*#(?:include|define|undef|if|ifdef|ifndef|elif|else|endif|pragma|error|warning|line)\b/m.test(source));
  const markdown = ["markdown", "mdx"].includes(language) || family === "markdown";
  const shader = language === "shaderlab" || family === "shaderlab";
  const lisp = ["common-lisp", "emacs-lisp"].includes(language) || family === "lisp";
  const batch = language === "bat" || family === "batchfile";
  add("embedded-language", /<(?:script|style)(?:\s|>)/i.test(source) ||
    (markdown && /^[\t ]{0,3}(?:`{3,}|~{3,})[\t ]*[A-Za-z][\w+-]*/m.test(source)) ||
    (shader && /\b(?:CGPROGRAM|CGINCLUDE|HLSLPROGRAM|HLSLINCLUDE)\b[\s\S]*?\b(?:ENDCG|ENDHLSL)\b/.test(source)));
  if (lisp && /#\|[\s\S]*?\n[\s\S]*?\|#/.test(source) && !constructs.includes("multiline-comment")) constructs.push("multiline-comment");
  add("markup-comment", /<!--[\s\S]*?-->/.test(source));
  add("interpolation", /\$\{|\{\{|\{%|<%|#\{/.test(source));
  add("escape-sequence", /\\(?:["'`\\nrtbfv0]|x[\da-f]{2}|u[\da-f]{4})/i.test(source));
  add("heredoc", /<<[-~]?[\t ]*["']?[A-Za-z_][\w]*["']?/.test(source));
  const css = family === "css" || ["css", "scss", "sass", "less"].includes(language);
  add("css-selector", css && /(?:^|})[^@{}\n][^{]*\{/m.test(source));
  add("css-declaration", css && /(?:^|[;{])[\t ]*(?:--)?[A-Za-z_-][\w-]*[\t ]*:/m.test(source));
  add("diff-hunk", family === "diff" || /^(?:diff --git|@@[\t ]+-\d)/m.test(source));
  add("line-continuation", /\\\r?\n/.test(source) || (batch && /\^\r?\n/.test(source)));
  return constructs;
}

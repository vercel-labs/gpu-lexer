import assert from "node:assert/strict";
import test from "node:test";

import { detectSyntaxConstructs, requiredSyntaxConstructs } from "../src/syntax-constructs.js";

test("construct detector covers difficult language-agnostic boundaries", () => {
  const source = `#include "server.h"
/* first
 * second */
const view = \`<style>
\${theme}</style>\`;
\\
`;
  const constructs = detectSyntaxConstructs(source, "cpp", "cpp");
  assert.ok(constructs.includes("preprocessor-directive"));
  assert.ok(constructs.includes("multiline-comment"));
  assert.ok(constructs.includes("multiline-string"));
  assert.ok(constructs.includes("embedded-language"));
  assert.ok(constructs.includes("interpolation"));
  assert.ok(constructs.includes("line-continuation"));
});

test("construct detector recognizes Python, markup, CSS, shell, and diff cases", () => {
  assert.ok(detectSyntaxConstructs('"""one\ntwo"""', "python", "python").includes("multiline-string"));
  assert.ok(detectSyntaxConstructs("<!-- one\ntwo -->", "html", "html").includes("markup-comment"));
  assert.deepEqual(
    detectSyntaxConstructs(".button { color: red; }", "css", "css").filter((name) => name.startsWith("css-")),
    ["css-selector", "css-declaration"],
  );
  assert.ok(detectSyntaxConstructs("cat <<EOF\ntext\nEOF", "shellscript", "shell").includes("heredoc"));
  assert.ok(detectSyntaxConstructs("@@ -1 +1 @@\n-old\n+new", "diff", "diff").includes("diff-hunk"));
  assert.equal(new Set(requiredSyntaxConstructs).size, requiredSyntaxConstructs.length);
});

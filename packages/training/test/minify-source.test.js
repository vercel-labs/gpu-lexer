import assert from "node:assert/strict";
import test from "node:test";

import { minifySource } from "../src/minify-source.js";

test("minifies JavaScript into dense valid source", async () => {
  const result = await minifySource("function greet(name) { return name + '!'; }", "javascript", "greet.js");
  assert.ok(result.length < 48);
  assert.match(result, /function greet\(./);
});

test("minifies CSS", async () => {
  assert.equal(await minifySource("main { color: red; }", "css", "site.css"), "main{color:red}");
});

test("minifies HTML and its embedded CSS and JavaScript", async () => {
  const result = await minifySource(`<!doctype html>
    <html><style>main { color: red; }</style>
    <script>const answer = 1 + 2;</script></html>`, "html", "index.html");
  assert.equal(result, "<!doctype html><html><style>main{color:red}</style><script>const answer=3</script></html>");
});


import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { promisify } from "node:util";

const execute = promisify(execFile);
const cache = await mkdtemp(resolve(tmpdir(), "gpu-lexer-npm-cache-"));
try {
  const { stdout } = await execute("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], {
    cwd: new URL("../", import.meta.url),
    env: { ...process.env, npm_config_cache: cache },
  });
  const [manifest] = JSON.parse(stdout);
  const files = manifest.files.map(({ path }) => path).sort();
  const allowed = ["LICENSE", "README.md", "dist/index.d.ts", "dist/index.js", "package.json"];
  assert.deepEqual(files, allowed, `unexpected npm package contents:\n${files.join("\n")}`);
  console.log(`npm package contents: ${files.join(", ")}`);
} finally {
  await rm(cache, { recursive: true, force: true });
}

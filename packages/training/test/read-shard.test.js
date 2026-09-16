import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { readShard } from "../src/read-shard.js";
import { loadTreeShard } from "../src/tree-model.js";

test("JSONL preserves Unicode separators, UTF-8 chunk boundaries, CRLF and final records without LF", async () => {
  const directory = await mkdtemp(join(tmpdir(), "read-shard-"));
  const path = join(directory, "fixture.jsonl.gz");
  const item = (source) => ({ source, language: "markdown", family: "markdown",
    sourceName: "fixture", path: "CHANGELOG.md", sourceLabelsVersion: 1,
    sourceLabels: [{ from: 0, to: source.length, class: "plain", confidence: 1 }],
  });
  const items = [item("documentation\u2029 for IE\u2028 on Windows"),
    item("x".repeat(16372) + "😀\u2029" + "y".repeat(17000))];
  try {
    await writeFile(path, gzipSync(items.map(JSON.stringify).join("\r\n\n")));
    assert.deepEqual(await Array.fromAsync(readShard(path)), items);
    const loaded = await loadTreeShard(path, Infinity, 0, { retainSource: true });
    assert.equal(loaded.fileCount, 2);
    assert.deepEqual(loaded.records.map(({ source }) => source), items.map(({ source }) => source));
    assert.equal((await loadTreeShard(path, Infinity, 0, { maxFiles: 1 })).fileCount, 1);
    await writeFile(path, gzipSync(JSON.stringify(items[0]) + '\n{"incomplete":'));
    assert.equal((await loadTreeShard(path, Infinity, 0, { maxFiles: 1 })).fileCount, 1);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("shard errors retain path, physical line number and I/O failures", async () => {
  const directory = await mkdtemp(join(tmpdir(), "read-shard-errors-"));
  const path = join(directory, "fixture.jsonl.gz");
  try {
    await writeFile(path, gzipSync('{"source":"a\u2029b"}\n{"source":"unterminated\n'));
    await assert.rejects(Array.fromAsync(readShard(path)), (error) =>
      error instanceof SyntaxError && error.message.includes(`${path}:2:`));
    await assert.rejects(Array.fromAsync(readShard(join(directory, "missing.gz"))), { code: "ENOENT" });
    await writeFile(path, "invalid gzip");
    await assert.rejects(Array.fromAsync(readShard(path)), { code: "Z_DATA_ERROR" });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

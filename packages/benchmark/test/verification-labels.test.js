import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

import { classNames } from "../../training/src/classes.js";
import { createTreeRecord } from "../../training/src/tree-model.js";
import { relabelVerification } from "../src/verification-labels.js";

test("benchmark targets are regenerated from pinned source instead of cached labels", async () => {
  const directory = await mkdtemp(join(tmpdir(), "verification-labels-"));
  const path = join(directory, "verification.jsonl.gz");
  const items = [
    { source: 'const object = { "my-key": 1 };\r\n', language: "javascript", family: "javascript" },
    { source: '<div title="😀">hello</div>', language: "html", family: "html" },
    { source: "const value: number = 1;", language: "angular-ts", family: "typescript" },
  ].map((item, index) => ({
    ...item, path: `fixture-${index}`, sourceName: "held-out", revision: "pinned",
    sourceLabelsVersion: 1,
    sourceLabels: [{ from: 0, to: item.source.length, class: "plain", confidence: 1 }],
  }));
  const excluded = { source: "ignored", language: "missing-grammar", family: "excluded" };
  const original = gzipSync([...items, excluded].map((item) => JSON.stringify(item)).join("\n") + "\n");
  try {
    await writeFile(path, original);
    const relabeled = [];
    for await (const item of relabelVerification(path, new Set(["javascript", "html", "typescript"]))) {
      relabeled.push(item);
    }
    assert.equal(relabeled.length, items.length);
    for (let index = 0; index < items.length; index++) {
      const { sourceLabels: cached, ...identity } = items[index];
      const { sourceLabels: fresh, ...actual } = relabeled[index];
      assert.deepEqual(actual, identity);
      assert.notDeepEqual(fresh, cached);
    }
    for (const [index, token, expected] of [[0, ";", "operator"], [0, "my-key", "string"], [1, "div", "keyword"], [2, ";", "operator"]]) {
      const item = relabeled[index];
      const from = item.source.indexOf(token);
      const { record } = createTreeRecord(item, undefined, { retainSource: true });
      let matchedParts = 0;
      for (let part = 0; part < record.targets.length; part++) {
        if (record.ranges[part * 2] >= from && record.ranges[part * 2] < from + token.length) {
          matchedParts++;
          assert.equal(classNames[record.targets[part]], expected, `${item.language}: ${token}`);
          assert.ok(record.supervisionWeights[part] > 0);
        }
      }
      assert.ok(matchedParts > 0, `${item.language}: ${token} must be scored`);
    }
    assert.deepEqual(await readFile(path), original, "checkpoint corpus remains byte-for-byte pinned");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

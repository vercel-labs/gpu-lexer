import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  activeCheckpointPath, compactCheckpointMetadata, loadDeployedCheckpoint, loadFloatCheckpoint,
} from "../src/checkpoint.js";
import { promotedRunId, promotedRunPath } from "../src/promoted-run.generated.js";

test("tracked active checkpoint is a complete float and deployed continuation baseline", async () => {
  assert.equal(promotedRunPath, activeCheckpointPath);
  const [floating, deployed] = await Promise.all([
    loadFloatCheckpoint(promotedRunPath), loadDeployedCheckpoint(promotedRunPath),
  ]);
  assert.equal(floating.metadata.runId, promotedRunId);
  assert.equal(deployed.metadata.runId, promotedRunId);
  assert.deepEqual(Object.keys(floating.model), Object.keys(deployed.model));
  assert.equal(floating.metadata.checkpointRole, "promoted-continuation-baseline");
  assert.equal(floating.metadata.history, undefined);
  assert.match(await readFile(new URL("../active/model-active.json", import.meta.url), "utf8"), /"verification"/);
});

test("compact checkpoint metadata removes history and machine-local run paths", () => {
  const result = compactCheckpointMetadata({
    runId: "run", history: [{ large: true }], selection: { status: "eligible" },
    promotionComparison: { candidate: { path: "/tmp/candidate" }, baseline: { path: "/tmp/baseline" } },
    config: {
      treeContext: "hybrid", languageObjective: { familyWeights: { javascript: 1 } },
      output: "/tmp/runs", trainShard: "/tmp/train", verificationShard: "/tmp/verify",
      miningShard: "/tmp/mine", initialRun: "/tmp/old", teacherRun: "/tmp/teacher",
      fixedBaseline: { run: "/tmp/base" }, fixedBaselineMetrics: { accuracy: 1 },
    },
  });
  assert.deepEqual(result.config, {
    treeContext: "hybrid", languageObjective: { familyWeights: { javascript: 1 } },
  });
  assert.equal(result.history, undefined);
  assert.equal(result.promotionComparison, undefined);
  assert.equal(result.selection, undefined);
});

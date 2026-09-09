import assert from "node:assert/strict";
import test from "node:test";

import { classificationMetrics } from "../src/metrics.js";

test("classification metrics exclude plain from the styled macro score", () => {
  const metrics = classificationMetrics(
    ["plain", "keyword", "keyword", "string"],
    ["plain", "keyword", "plain", "string"],
    ["plain", "keyword", "string"],
    { exclude: ["plain"] },
  );
  assert.equal(metrics.accuracy, 0.75);
  assert.ok(Math.abs(metrics.macroF1 - 5 / 6) < 1e-12);
  assert.equal(metrics.confusion.keyword.plain, 1);
});

import assert from "node:assert/strict";
import test from "node:test";

import { promotionRejection, validateModel, websiteStatsContents } from "../src/promote.js";

test("promotion generates human-facing website quality stats", () => {
  const contents = websiteStatsContents({
    runId: "run-123",
    verification: { accuracy: 0.875, macroF1: 0.75, perLanguage: {
      javascript: { accuracy: 0.97, support: 100 },
      typescript: { accuracy: 0.93, support: 100 },
      css: { accuracy: 0.84, support: 100 },
      markdown: { accuracy: 0.68, support: 100 },
      liquid: { accuracy: 0.55, support: 100 },
      jinja: { accuracy: 0.42, support: 100 },
    } },
    corpus: { train: { tokens: 1500, scoredTokens: 1234 } },
    history: [{ trainingTokens: 1510 }],
    runtimeParameterCount: 2048,
    runtimeWeightBytes: 1536,
    quantization: { bits: 6 },
  });

  assert.match(contents, /"runId": "run-123"/);
  assert.match(contents, /"accuracy": 0\.875/);
  assert.match(contents, /"shikiDisagreementRate": 0\.125/);
  assert.match(contents, /"trainingTokensPerEpoch": 1510/);
  assert.match(contents, /"modelParameters": 2048/);
  assert.match(contents, /"modelWeightBytes": 1536/);
  assert.match(contents, /"range": "95–100%",\n      "languages": \[\n        "javascript"/);
  assert.match(contents, /"range": "<50%",\n      "languages": \[\n        "jinja"/);
});

test("only the current tree architecture is promotable", () => {
  assert.throws(() => validateModel({
    model: "affine-scan", formatVersion: 5, acceptance: { accepted: true },
  }, new Uint8Array()), /unsupported model/);
  assert.match(promotionRejection({ precision: "float32" }), /teacher\/float32/);
  assert.match(promotionRejection({
    acceptance: { accepted: false, criterion: "obsolete-quality-target" },
    verification: { accuracy: 0.9, macroF1: 0.8 },
  }), /obsolete-quality-target/);
});

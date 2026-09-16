import assert from "node:assert/strict";
import test from "node:test";
import { initialize, validate } from "wgslender";

import { promotedModel } from "../src/model.generated.js";
import { createPromotedShader, TREE_ENTRY_POINTS } from "../src/shader.min.generated.js";
import { createTreeShader } from "../src/tree-shader.js";

const entries = [
  "hybrid_neighbor_blocks", "hybrid_neighbor_prefixes", "hybrid_scan_blocks",
  "hybrid_scan_prefixes", "hybrid_mix_tree_up", "tree_global", "tree_down_classify",
];

test("the active tree architecture and generated shaders compile", async () => {
  await initialize();
  for (const f16 of [false, true]) {
    const source = createTreeShader(promotedModel, { f16 });
    for (const entry of entries) assert.match(source, new RegExp(`fn ${entry}\\(`));
    assert.equal(validate(source).valid, true);
    const generated = createPromotedShader(f16);
    for (const entry of TREE_ENTRY_POINTS) assert.match(generated, new RegExp(`fn ${entry}\\(`));
    assert.equal(generated.startsWith("enable f16;"), f16);
  }
});

test("obsolete tree formats are rejected", () => {
  assert.throws(() => createTreeShader({ ...promotedModel, formatVersion: 8 }), /incompatible/);
});

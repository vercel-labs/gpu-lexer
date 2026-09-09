import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createTreeModel, treeTensorNamesFor } from "./tree-model.js";
import { capacityExperiments } from "./tree-training-policy.js";
import { createLanguageObjective } from "./language-objective.js";

// Read-only planner: never launches a trainer or allocates corpus features.
export function experimentPlan() {
  return { languageObjective: createLanguageObjective(), context: "hybrid", bits: 6,
    comparisons: capacityExperiments.map((shape) => {
      const model = createTreeModel({ ...shape, treeContext: "hybrid", random: () => 0.5 });
      const parameters = treeTensorNamesFor(model).reduce((sum, name) => sum + model[name].length, 0);
      return { ...shape, parameters, packedBytes: Math.ceil(parameters * 6 / 8),
        arguments: `--context hybrid --bits 6 --hidden ${shape.hiddenSize} --classifier ${shape.classifierSize} --hash ${shape.hashBuckets}` };
    }),
    lossComparisons: [
      { name: "original", arguments: "--class-weight-power 0.5 --calibration-epochs 0" },
      { name: "gentle", arguments: "--class-weight-power 0.25 --calibration-epochs 0" },
      { name: "calibrated", arguments: "--class-weight-power 0.5 --calibration-epochs 2" },
    ],
    notes: ["Change one axis at a time; keep shards, seed, objective and fixed baseline identical.",
      "Save languageObjective as JSON and pass --language-objective <file> to pin or customize weights.",
      "Pass --baseline-run <direct-label-run> for promotion eligibility; no baseline means diagnostic only.",
      "Calibration uses the natural corpus distribution and class weights, freezes the encoder, and retains confidence masks.",
      "Use evaluate-tree.js for bounded float-vs-packed diagnostics before lower-bit experiments."],
  };
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  console.log(JSON.stringify(experimentPlan(), null, 2));
}

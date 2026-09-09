import { classNames } from "./classes.js";
import { treeProbabilities } from "./tree-model.js";

export const TREE_REPLAY_FAMILIES = Object.freeze([
  "markdown", "plpgsql", "vue", "css", "shell", "ruby", "lua",
]);
export const TREE_REPLAY_FRACTION = 0.01;
export const TREE_MAX_REPLAY_FRACTION = 0.25;
export const TREE_AUTO_REPLAY_FRACTION = 0.01;
export const TREE_AUTO_REPLAY_REPEATS = 2;

/** Resolve the tree curriculum's token mix. Automatic mining is deliberately
 * capped at 1%; tiny failure banks must not be repeated into a new majority.
 */
export function treeReplayFraction(value, { autoMining = false } = {}) {
  const fraction = value ?? (autoMining ? TREE_REPLAY_FRACTION : 0);
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > TREE_MAX_REPLAY_FRACTION) {
    throw new RangeError(`tree replay fraction must be between 0 and ${TREE_MAX_REPLAY_FRACTION}`);
  }
  if (autoMining && fraction > TREE_AUTO_REPLAY_FRACTION) {
    throw new RangeError(`automatic tree mining is capped at ${TREE_AUTO_REPLAY_FRACTION}`);
  }
  return fraction;
}

/** Rank independent mining windows by the deployed model's actual errors.
 * The returned records contain full contextual labels, with only failing parts
 * receiving the extra hard-target weight.
 */
export function mineTreeReplay(records, model, config, {
  families = TREE_REPLAY_FAMILIES,
  windowParts = 128,
  strideParts = Math.floor(windowParts / 2),
  contextParts = Math.floor(windowParts / 4),
  maxParts = 250_000,
  maxWindowsPerSource = 12,
  failureWeight = 2,
  predict = (record) => treeProbabilities(model, record, config),
} = {}) {
  for (const [name, value] of Object.entries({ windowParts, strideParts, contextParts, maxParts, maxWindowsPerSource, failureWeight })) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`);
  }
  if (failureWeight > 255) throw new RangeError("failureWeight must fit in the uint8 training dataset");
  const focused = new Set(families);
  const candidates = [];
  const totals = Object.fromEntries(families.map((family) => [family, {
    files: 0, supervised: 0, errors: 0, candidateWindows: 0, selectedWindows: 0, selectedParts: 0,
  }]));

  records.forEach((record, recordIndex) => {
    const family = record.family ?? record.language ?? "unknown";
    if (!focused.has(family) || !record.targets.length) return;
    const probabilities = predict(record);
    if (probabilities.length !== record.targets.length) throw new Error("tree mining prediction length mismatch");
    const failures = [];
    let supervised = 0;
    for (let part = 0; part < record.targets.length; part++) {
      if (record.supervisionWeights?.[part] === 0) continue;
      if (probabilities[part]?.length !== classNames.length) throw new Error("tree mining class count mismatch");
      supervised += 1;
      const predicted = argmax(probabilities[part]);
      if (predicted !== record.targets[part]) failures.push(part);
    }
    const familyTotal = totals[family];
    familyTotal.files += 1;
    familyTotal.supervised += supervised;
    familyTotal.errors += failures.length;
    if (!failures.length) return;

    const starts = new Set();
    const maximumStart = Math.max(0, record.targets.length - windowParts);
    for (const part of failures) {
      const anchored = Math.floor(part / strideParts) * strideParts - contextParts;
      starts.add(Math.max(0, Math.min(maximumStart, anchored)));
    }
    for (const from of starts) {
      const to = Math.min(record.targets.length, from + windowParts);
      const windowFailures = failures.filter((part) => part >= from && part < to);
      let windowSupervised = 0;
      let errorContribution = 0;
      for (let part = from; part < to; part++) {
        if (record.supervisionWeights?.[part] === 0) continue;
        windowSupervised += 1;
      }
      for (const part of windowFailures) {
        const expectedProbability = probabilities[part]?.[record.targets[part]] ?? 0;
        errorContribution += -Math.log(Math.max(1e-6, expectedProbability));
      }
      candidates.push({
        record, recordIndex, family, from, to, failures: windowFailures,
        errors: windowFailures.length, density: windowFailures.length / Math.max(1, windowSupervised),
        errorContribution,
        sourceKey: record.sourceName ?? record.origin ?? record.path ?? `${family}:${recordIndex}`,
      });
      familyTotal.candidateWindows += 1;
    }
  });

  const groups = new Map(families.map((family) => [family, []]));
  for (const candidate of candidates) groups.get(candidate.family).push(candidate);
  for (const group of groups.values()) group.sort(compareCandidates);
  const selected = [];
  const perSource = new Map();
  const intervals = new Map();
  let selectedParts = 0;
  let changed = true;
  while (changed && selectedParts < maxParts) {
    changed = false;
    for (const family of families) {
      const group = groups.get(family);
      let candidate;
      while ((candidate = group.shift())) {
        if ((perSource.get(candidate.sourceKey) ?? 0) >= maxWindowsPerSource) continue;
        const accepted = intervals.get(candidate.recordIndex) ?? [];
        if (accepted.some(([from, to]) => overlapFraction(from, to, candidate.from, candidate.to) > 0.5)) continue;
        break;
      }
      if (!candidate) continue;
      const length = candidate.to - candidate.from;
      if (selectedParts + length > maxParts) continue;
      selected.push(sliceReplayWindow(candidate, failureWeight));
      selectedParts += length;
      perSource.set(candidate.sourceKey, (perSource.get(candidate.sourceKey) ?? 0) + 1);
      const accepted = intervals.get(candidate.recordIndex) ?? [];
      accepted.push([candidate.from, candidate.to]);
      intervals.set(candidate.recordIndex, accepted);
      totals[family].selectedWindows += 1;
      totals[family].selectedParts += length;
      changed = true;
      if (selectedParts >= maxParts) break;
    }
  }

  return {
    records: selected,
    stats: {
      focusedFamilies: [...families], sourceRecords: records.length,
      candidateWindows: candidates.length, selectedWindows: selected.length, selectedParts,
      failureWeight, windowParts, maxWindowsPerSource, perFamily: totals,
    },
  };
}

function sliceReplayWindow(candidate, failureWeight) {
  const { record, from, to } = candidate;
  const lossWeights = record.lossWeights?.slice(from, to) ?? new Uint8Array(to - from).fill(1);
  for (const part of candidate.failures) lossWeights[part - from] = failureWeight;
  return {
    ...record,
    features: record.features.slice(from, to),
    targets: record.targets.slice(from, to),
    auxiliary: record.auxiliary?.slice(from, to),
    supervisionWeights: record.supervisionWeights?.slice(from, to),
    lossWeights,
    replayWeight: Math.max(1, candidate.errorContribution * (1 + candidate.density)),
    miningWindow: {
      from, to, errors: candidate.errors, density: candidate.density,
      errorContribution: candidate.errorContribution,
    },
  };
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < classNames.length; index++) if (values[index] > values[best]) best = index;
  return best;
}

function compareCandidates(left, right) {
  return right.errorContribution - left.errorContribution || right.errors - left.errors ||
    right.density - left.density || left.recordIndex - right.recordIndex || left.from - right.from;
}

function overlapFraction(aFrom, aTo, bFrom, bTo) {
  return Math.max(0, Math.min(aTo, bTo) - Math.max(aFrom, bFrom)) / Math.min(aTo - aFrom, bTo - bFrom);
}

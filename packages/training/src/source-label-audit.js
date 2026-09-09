import { classNames } from "./classes.js";
import { corpusSourceLabels, validateSourceLabels } from "./tree-label-alignment.js";

/** Tokenizer-independent diagnostics on the same original-source UTF-16 grid.
 * Inputs are sorted disjoint {from,to,class} spans (confidence is ignored).
 * Unlabeled teacher positions are excluded; missing predictions count as plain
 * AND reduce predictionCoverage. Whitespace is excluded by default using the
 * runtime ASCII whitespace set. Exact styled spans are maximal contiguous runs
 * on the scored grid; whitespace/unlabeled positions break runs. Boundaries are
 * class transitions between adjacent scored units, excluding file edges/gaps.
 * This is an in-memory diagnostic only: no model, corpus I/O, or teacher calls.
 */
export function auditSourceLabels(source, expectedSpans, predictedSpans, { includeWhitespace = false } = {}) {
  const expected = rasterize(source.length, expectedSpans);
  const predicted = rasterize(source.length, predictedSpans);
  const confusion = classNames.map(() => classNames.map(() => 0));
  const expectedRuns = [], predictedRuns = [];
  const expectedBoundaries = new Set(), predictedBoundaries = new Set();
  let total = 0, correct = 0, predictedCoverage = 0;
  let eligible = 0, unlabeled = 0;
  const uncoveredRanges = [];
  let uncovered = null;
  let previousExpected = -1, previousPredicted = -1;
  let expectedRun = null, predictedRun = null;
  for (let offset = 0; offset < source.length; offset++) {
    const code = source.charCodeAt(offset);
    const whitespace = code === 32 || (code >= 9 && code <= 13);
    if (includeWhitespace || !whitespace) {
      eligible++;
      if (expected[offset] < 0) {
        unlabeled++;
        uncovered = appendRun(uncoveredRanges, uncovered, offset, -1);
      } else uncovered = null;
    } else uncovered = null;
    if (expected[offset] < 0 || (!includeWhitespace && whitespace)) {
      previousExpected = previousPredicted = -1;
      expectedRun = predictedRun = null;
      continue;
    }
    const target = expected[offset];
    const guess = Math.max(0, predicted[offset]);
    confusion[target][guess]++;
    total++;
    correct += Number(target === guess);
    predictedCoverage += Number(predicted[offset] >= 0);
    if (previousExpected >= 0 && previousExpected !== target) expectedBoundaries.add(offset);
    if (previousPredicted >= 0 && previousPredicted !== guess) predictedBoundaries.add(offset);
    expectedRun = appendRun(expectedRuns, expectedRun, offset, target);
    predictedRun = appendRun(predictedRuns, predictedRun, offset, guess);
    previousExpected = target;
    previousPredicted = guess;
  }
  const perClass = classNames.map((name, index) => {
    const support = confusion[index].reduce((sum, count) => sum + count, 0);
    const predictedCount = confusion.reduce((sum, row) => sum + row[index], 0);
    return { class: name, support, ...prf(confusion[index][index], predictedCount, support) };
  });
  const majority = perClass.reduce((best, value) => value.support > best.support ? value : best, perClass[0]);
  const expectedStyled = new Set(expectedRuns.filter((run) => run.class !== 0).map(runKey));
  const predictedStyled = new Set(predictedRuns.filter((run) => run.class !== 0).map(runKey));
  const byClass = classNames.slice(1).map((name, index) => {
    const expected = new Set(expectedRuns.filter((run) => run.class === index + 1).map(runKey));
    const predicted = new Set(predictedRuns.filter((run) => run.class === index + 1).map(runKey));
    return { class: name, ...setMetrics(expected, predicted) };
  });
  return {
    coordinateUnit: "utf16-code-unit", includeWhitespace,
    teacherCoverage: {
      eligible, labeled: total, unlabeled, coverage: total / Math.max(1, eligible),
      uncoveredRanges: uncoveredRanges.map(({ from, to }) => ({ from, to })),
    },
    characters: {
      total, correct, accuracy: correct / Math.max(1, total),
      predictionCoverage: predictedCoverage / Math.max(1, total),
      macroF1: perClass.slice(1).reduce((sum, value) => sum + value.f1, 0) / (classNames.length - 1),
      falseColorRate: (perClass[0].support - confusion[0][0]) / Math.max(1, perClass[0].support),
      majorityBaseline: { class: majority.class, accuracy: majority.support / Math.max(1, total) },
      confusion, perClass,
    },
    spans: { ...setMetrics(expectedStyled, predictedStyled), perClass: byClass },
    boundary: setMetrics(expectedBoundaries, predictedBoundaries),
  };
}

/** Audits old shards only with explicit opt-in, retaining fallback provenance. */
export function auditCorpusLabels(item, predictedSpans, options = {}) {
  const { sourceLabels, ...provenance } = corpusSourceLabels(item);
  return { ...provenance, ...auditSourceLabels(item.source, sourceLabels, predictedSpans, options) };
}

function rasterize(length, spans) {
  const labels = spans.map(({ from, to, class: name }) => ({ from, to, class: name }));
  validateSourceLabels(labels, length);
  const result = new Int16Array(length).fill(-1);
  for (const span of labels) result.fill(classNames.indexOf(span.class), span.from, span.to);
  return result;
}

function appendRun(runs, previous, offset, name) {
  if (previous?.class === name && previous.to === offset) {
    previous.to++;
    return previous;
  }
  const run = { from: offset, to: offset + 1, class: name };
  runs.push(run);
  return run;
}

function runKey(run) { return `${run.from}:${run.to}:${run.class}`; }
function setMetrics(expected, predicted) {
  let matched = 0;
  for (const value of predicted) matched += Number(expected.has(value));
  return { matched, expected: expected.size, predicted: predicted.size, ...prf(matched, predicted.size, expected.size) };
}
function prf(matched, predicted, expected) {
  return {
    precision: matched / Math.max(1, predicted), recall: matched / Math.max(1, expected),
    f1: 2 * matched / Math.max(1, predicted + expected),
  };
}

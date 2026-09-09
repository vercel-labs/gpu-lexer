import { classNames } from "./classes.js";

/** Validate disjoint, sorted, half-open UTF-16 source spans. Overlapping teachers
 * are ambiguous and rejected, never double-counted. Gaps are permitted (newlines).
 */
export function validateSourceLabels(labels, sourceLength = Infinity) {
  if (!Array.isArray(labels)) throw new TypeError("sourceLabels must be an array");
  let end = 0;
  for (const label of labels) {
    if (!label || !Number.isInteger(label.from) || !Number.isInteger(label.to) ||
        label.from < end || label.to <= label.from || label.to > sourceLength) {
      throw new Error("sourceLabels must have sorted, non-overlapping, nonempty UTF-16 ranges within the source");
    }
    if (!classNames.includes(label.class)) throw new Error(`unknown source label class ${label.class}`);
    if (label.confidence != null && (!Number.isFinite(label.confidence) || label.confidence < 0 || label.confidence > 1)) {
      throw new Error("sourceLabels confidence must be in [0, 1], not tokenizer byte confidence");
    }
    if (label.auxiliary != null && (!Number.isInteger(label.auxiliary) || label.auxiliary < 0 || label.auxiliary > 65535)) {
      throw new Error("sourceLabels auxiliary must be a uint16 bit mask");
    }
    end = label.to;
  }
}

/** Corpus selection can truncate source at a tokenizer boundary, inside a Shiki
 * span. Clip only at serialization; do not mutate the original teacher array.
 */
export function clipSourceLabels(sourceLabels, sourceLength) {
  if (!Number.isInteger(sourceLength) || sourceLength < 0) throw new Error("invalid source length");
  validateSourceLabels(sourceLabels);
  const clipped = [];
  for (const span of sourceLabels) {
    if (span.from >= sourceLength) break;
    clipped.push({ ...span, to: Math.min(span.to, sourceLength) });
  }
  return clipped;
}

/** Align direct Shiki spans to flat [from,to,...] tree ranges (including typed
 * arrays), or an array of {from,to}. Tree ranges may overlap or be unordered.
 * Votes aggregate by class, not by largest individual span; ties choose the
 * earliest source class. Auxiliary bits union all overlaps. Output confidence
 * is a byte, attenuated by dominance, scope confidence and coverage. Uncovered
 * ranges have confidence 0. Callers still mask whitespace and add nesting bits.
 */
export function alignTreeLabels(sourceLabels, ranges, { sourceLength = Infinity } = {}) {
  validateSourceLabels(sourceLabels, sourceLength);
  const objects = ranges.length > 0 && typeof ranges[0] === "object";
  if (!objects && ranges.length % 2) throw new Error("tree ranges must contain from/to pairs");
  const count = objects ? ranges.length : ranges.length / 2;
  const result = new Array(count);
  for (let index = 0; index < count; index++) {
    const from = objects ? ranges[index].from : ranges[index * 2];
    const to = objects ? ranges[index].to : ranges[index * 2 + 1];
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to <= from || to > sourceLength) {
      throw new Error(`invalid tree UTF-16 range at ${index}`);
    }
    let low = 0, high = sourceLabels.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (sourceLabels[middle].to <= from) low = middle + 1;
      else high = middle;
    }
    const votes = new Map();
    let coveredWidth = 0, auxiliary = 0;
    for (let cursor = low; cursor < sourceLabels.length && sourceLabels[cursor].from < to; cursor++) {
      const span = sourceLabels[cursor];
      const width = Math.min(to, span.to) - Math.max(from, span.from);
      const vote = votes.get(span.class) ?? { width: 0, confidence: 0 };
      vote.width += width;
      vote.confidence += width * (span.confidence ?? 1);
      votes.set(span.class, vote);
      coveredWidth += width;
      auxiliary |= span.auxiliary ?? 0;
    }
    let selected = "plain", best = { width: 0, confidence: 0 };
    for (const [name, vote] of votes) if (vote.width > best.width) { selected = name; best = vote; }
    const coverage = coveredWidth / (to - from);
    const dominance = coveredWidth ? best.width / coveredWidth : 0;
    const scopeConfidence = best.width ? best.confidence / best.width : 0;
    result[index] = {
      from, to, class: selected, auxiliary,
      confidence: coveredWidth ? Math.round(255 * Math.max(0.25, dominance * scopeConfidence * coverage)) : 0,
      coverage, dominance,
    };
  }
  return result;
}

/** Missing source labels are never silently presented as direct supervision. */
export function corpusSourceLabels(item) {
  if (typeof item.source !== "string") throw new Error("corpus item is missing source text");
  if (item.sourceLabels == null) {
    throw new Error(`corpus item ${item.path ?? "<unknown>"} is missing sourceLabels; direct labels require a corpus rebuild`);
  }
  if (item.sourceLabelsVersion != null && item.sourceLabelsVersion !== 1) {
    throw new Error(`unsupported sourceLabelsVersion ${item.sourceLabelsVersion}`);
  }
  validateSourceLabels(item.sourceLabels, item.source.length);
  return { sourceLabels: item.sourceLabels, labelSource: "shiki-source" };
}

export function alignCorpusTreeLabels(item, ranges) {
  const { sourceLabels, ...provenance } = corpusSourceLabels(item);
  return { ...provenance, labels: alignTreeLabels(sourceLabels, ranges, { sourceLength: item.source.length }) };
}

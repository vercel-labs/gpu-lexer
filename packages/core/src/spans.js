import { syntaxClassNames } from "./constants.js";

// Each label byte holds the class in its low four bits and a confidence level
// in its high four. A span reports the lowest confidence among its parts.
//
// Parts [from, to) are appended to spans, continuing its last span, so a large
// source can be rebuilt in pieces as its labels arrive.
export function spansFromRanges(ranges, labels, labelOffset = 0, from = 0, to = ranges.length >> 1, spans = []) {
  // The open span's class and confidence level stay in integers; names and
  // fractions are only produced when a span starts or its confidence drops.
  let previous = spans[spans.length - 1];
  let type = previous ? syntaxClassNames.indexOf(previous.type) : -1;
  let level = previous ? previous.confidence * 16 - 1 : 0;
  for (let index = from; index < to; index++) {
    const start = ranges[index * 2];
    const labelIndex = labelOffset + index;
    const label = labels[labelIndex >> 2] >> ((labelIndex & 3) * 8) & 255;
    // Class codes without a name are plain.
    const next = (label & 15) < syntaxClassNames.length ? label & 15 : 0;
    if (next === type && previous.end === start) {
      previous.end = ranges[index * 2 + 1];
      if (label >> 4 < level) {
        level = label >> 4;
        previous.confidence = (level + 1) / 16;
      }
    } else {
      type = next;
      level = label >> 4;
      previous = { type: syntaxClassNames[type], start, end: ranges[index * 2 + 1], confidence: (level + 1) / 16 };
      spans.push(previous);
    }
  }
  return spans;
}

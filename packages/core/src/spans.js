import { syntaxClassNames } from "./constants.js";

export function spansFromRanges(ranges, labels, labelOffset = 0) {
  const spans = [];
  for (let index = 0; index < ranges.length >> 1; index++) {
    const from = ranges[index * 2];
    const to = ranges[index * 2 + 1];
    const labelIndex = labelOffset + index;
    const label = labels[labelIndex >> 2] >> ((labelIndex & 3) * 8) & 255;
    append(from, to, syntaxClassNames[label] ?? "plain");
  }
  return spans;

  function append(start, end, type) {
    const previous = spans.at(-1);
    if (previous && previous.type === type && previous.end === start) {
      previous.end = end;
    } else {
      spans.push({ type, start, end });
    }
  }
}

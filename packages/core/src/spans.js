import { syntaxClassNames } from "./constants.js";

export function spansFromRanges(ranges, labels, labelOffset = 0) {
  const spans = [];
  let previous;
  for (let index = 0; index < ranges.length >> 1; index++) {
    const start = ranges[index * 2];
    const end = ranges[index * 2 + 1];
    const labelIndex = labelOffset + index;
    const label = labels[labelIndex >> 2] >> ((labelIndex & 3) * 8) & 255;
    const type = syntaxClassNames[label] ?? "plain";
    if (previous && previous.type === type && previous.end === start) {
      previous.end = end;
    } else {
      previous = { type, start, end };
      spans.push(previous);
    }
  }
  return spans;
}

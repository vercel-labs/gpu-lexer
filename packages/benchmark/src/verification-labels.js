import { teacherLanguages } from "../../training/src/corpus.js";
import { createLabeler } from "../../training/src/label.js";
import { readShard } from "../../training/src/read-shard.js";

/** Keep the pinned source selection, but regenerate targets with today's taxonomy. */
export async function* relabelVerification(path, families) {
  const items = [];
  for await (const item of readShard(path)) {
    if (!families.has(item.family)) continue;
    delete item.sourceLabels;
    items.push(item);
  }

  // Include the actual shard languages (including website examples) and embedded grammars.
  const labeler = await createLabeler({
    langs: [...new Set([...teacherLanguages, ...items.map(({ language }) => language)])],
  });
  try {
    for (const item of items) {
      yield {
        ...item,
        sourceLabelsVersion: 1,
        sourceLabels: labeler.labelSource(item.source, item.language),
      };
    }
  } finally {
    labeler.dispose();
  }
}

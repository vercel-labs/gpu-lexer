import { languageFamily } from "./corpus.js";

/** Duplicate short families until each reaches its requested part floor. */
export function balanceLanguageRecords(records, balance = 10_000) {
  const policy = typeof balance === "number"
    ? { defaultMinimumTokens: balance, minimumTokens: {}, aliases: {} }
    : { ...balance, minimumTokens: balance.minimumTokens ?? {}, aliases: balance.aliases ?? {} };
  const groups = new Map();
  for (const record of records) {
    const family = balancedFamily(record, policy.aliases);
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push(record);
  }
  const result = records.slice();
  for (const [family, group] of groups) {
    const minimum = policy.minimumTokens[family] ?? policy.defaultMinimumTokens;
    let parts = group.reduce((sum, record) => sum + record.targets.length, 0);
    for (let index = 0; parts < minimum; index++) {
      const copy = sliceRecord(group[index % group.length], minimum - parts);
      result.push(copy);
      parts += copy.targets.length;
    }
  }
  return result;
}

export function upsampleMixedRecords(records, multiplier) {
  const result = records.slice();
  for (let copy = 1; copy < multiplier; copy++) {
    for (const record of records) if (record.mixedLanguage) result.push(record);
  }
  return result;
}

export function balancedAuxiliaryPositiveWeights(records, cap = 8, auxiliaryCount = 16) {
  const positives = new Uint32Array(auxiliaryCount);
  let total = 0;
  for (const record of records) for (const bits of record.auxiliary ?? []) {
    for (let index = 0; index < auxiliaryCount; index++) positives[index] += (bits >> index) & 1;
    total += 1;
  }
  return Float32Array.from(positives, (positive) => {
    if (!positive || positive >= total) return 1;
    return Math.min(cap, Math.max(1, Math.sqrt((total - positive) / positive)));
  });
}

function balancedFamily(record, aliases) {
  let family = record.family ?? languageFamily(record.language ?? "unknown");
  const seen = new Set();
  while (aliases[family]) {
    if (seen.has(family)) throw new Error(`cyclic family balance alias at ${family}`);
    seen.add(family);
    family = aliases[family];
  }
  return family;
}

function sliceRecord(record, maximum) {
  const length = Math.min(record.targets.length, maximum);
  return {
    ...record,
    features: record.features?.slice(0, length),
    targets: record.targets.slice(0, length),
    auxiliary: record.auxiliary?.slice(0, length),
    supervisionWeights: record.supervisionWeights?.slice(0, length),
    lossWeights: record.lossWeights?.slice(0, length),
  };
}

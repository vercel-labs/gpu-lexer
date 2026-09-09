import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { websiteExampleFamilies } from "./website-examples.js";

export const LANGUAGE_OBJECTIVE_VERSION = 1;
export const stronglyGuardedLanguages = Object.freeze([
  "css", "html", "javascript", "python", "tsx", "typescript",
]);
export const supplementalLanguageWeights = Object.freeze({
  html: 1, css: 1, markdown: 1, vue: 1, svelte: 1, astro: 1, diff: 1,
});

// Raw weights use the snapshot's percentage-point units, not token frequencies.
// Persist this complete object and pass it unchanged to JS and Python evaluation.
export function createLanguageObjective({
  popularityPath = new URL("../data/language-popularity.json", import.meta.url),
  supplementalWeights = supplementalLanguageWeights,
  protectedFamilies,
  minSupport = 100, minPlainSupport = 100,
  maxErrorIncrease = 0.01, maxFalseColorIncrease = 0.01, z = 1.96,
  strictLanguages = stronglyGuardedLanguages,
  maxStrictErrorIncrease = 0.002, maxStrictFalseColorIncrease = 0.002,
  matureLanguages = [], maxMatureErrorIncrease = 0.01, maxMatureFalseColorIncrease = 0.01,
  minWeightedImprovement = 0,
} = {}) {
  const bytes = readFileSync(popularityPath);
  const popularity = JSON.parse(bytes);
  const raw = {};
  const supplemental = new Set(popularity.languages.filter((row) => row.supplemental).map((row) => row.family));
  for (const family of supplemental) {
    if (!Object.hasOwn(supplementalWeights, family)) throw new Error(`missing explicit supplemental weight: ${family}`);
  }
  for (const family of Object.keys(supplementalWeights)) {
    if (!supplemental.has(family)) throw new Error(`unknown supplemental family: ${family}`);
  }
  for (const row of popularity.languages) {
    const weight = row.supplemental ? supplementalWeights[row.family] : row.percent;
    if (!Number.isFinite(weight) || weight < 0) throw new Error(`invalid family weight: ${row.family}`);
    raw[row.family] = (raw[row.family] ?? 0) + weight;
  }
  // Website-only verification families participate in aggregate accuracy and
  // diagnostics without changing the popularity-weighted objective.
  for (const family of websiteExampleFamilies()) raw[family] ??= 0;
  const total = Object.values(raw).reduce((sum, weight) => sum + weight, 0);
  if (!(total > 0)) throw new Error("language objective needs positive weights");
  const objective = {
    schemaVersion: LANGUAGE_OBJECTIVE_VERSION,
    popularitySha256: createHash("sha256").update(bytes).digest("hex"),
    supplementalWeights: { ...supplementalWeights },
    familyWeights: Object.fromEntries(Object.keys(raw).sort().map((family) => [family, raw[family] / total])),
    protectedFamilies: [...(protectedFamilies ?? Object.keys(raw).filter((family) => raw[family] >= 1))].sort(),
    strictLanguages: [...strictLanguages].sort(),
    matureLanguages: [...matureLanguages].sort(),
    minSupport, minPlainSupport, maxErrorIncrease, maxFalseColorIncrease,
    maxStrictErrorIncrease, maxStrictFalseColorIncrease, z, minWeightedImprovement,
    maxMatureErrorIncrease, maxMatureFalseColorIncrease,
  };
  validateLanguageObjective(objective);
  return objective;
}

export function validateLanguageObjective(objective) {
  if (objective?.schemaVersion !== LANGUAGE_OBJECTIVE_VERSION ||
      !/^[a-f0-9]{64}$/.test(objective.popularitySha256 ?? "")) throw new Error("invalid language objective provenance");
  const weights = Object.entries(objective.familyWeights ?? {});
  if (!weights.length || weights.some(([family, weight]) => !family || !Number.isFinite(weight) || weight < 0) ||
      Math.abs(weights.reduce((sum, [, weight]) => sum + weight, 0) - 1) > 1e-9) {
    throw new Error("familyWeights must be globally normalized, finite nonnegative weights");
  }
  if (!objective.supplementalWeights || Object.entries(objective.supplementalWeights).some(([family, weight]) =>
    !Object.hasOwn(objective.familyWeights, family) || !Number.isFinite(weight) || weight < 0)) {
    throw new Error("invalid explicit supplementalWeights");
  }
  if (!Array.isArray(objective.protectedFamilies) || new Set(objective.protectedFamilies).size !== objective.protectedFamilies.length ||
      objective.protectedFamilies.some((family) => !(objective.familyWeights[family] > 0))) {
    throw new Error("invalid protectedFamilies");
  }
  if (!Array.isArray(objective.strictLanguages) || !objective.strictLanguages.length ||
      new Set(objective.strictLanguages).size !== objective.strictLanguages.length ||
      objective.strictLanguages.some((language) => typeof language !== "string" || !language)) {
    throw new Error("invalid strictLanguages");
  }
  const matureLanguages = objective.matureLanguages ?? [];
  if (!Array.isArray(matureLanguages) || new Set(matureLanguages).size !== matureLanguages.length ||
      matureLanguages.some((language) => typeof language !== "string" || !language ||
        objective.strictLanguages.includes(language))) {
    throw new Error("invalid matureLanguages");
  }
  for (const key of ["minSupport", "minPlainSupport"]) {
    if (!Number.isSafeInteger(objective[key]) || objective[key] < 1) throw new Error(`invalid ${key}`);
  }
  for (const key of ["maxErrorIncrease", "maxFalseColorIncrease", "maxStrictErrorIncrease",
    "maxStrictFalseColorIncrease", "z", "minWeightedImprovement"]) {
    if (!Number.isFinite(objective[key]) || objective[key] < 0) throw new Error(`invalid ${key}`);
  }
  for (const key of ["maxMatureErrorIncrease", "maxMatureFalseColorIncrease"]) {
    const value = objective[key] ?? 0.01;
    if (!Number.isFinite(value) || value < 0) throw new Error(`invalid ${key}`);
  }
  return objective;
}

// Each supervised, non-whitespace part counts once regardless of confidence/loss
// weight. falseColorRate is plain -> styled / plain support (null without plain).
export function languageMetrics(counts, objective = null) {
  if (objective) validateLanguageObjective(objective);
  const families = [...new Set([...Object.keys(counts), ...Object.keys(objective?.familyWeights ?? {})])].sort();
  const perFamily = Object.fromEntries(families.map((family) => {
    const count = counts[family] ?? { support: 0, errors: 0, plainSupport: 0, falseColors: 0 };
    validateCounts(count);
    return [family, { ...count, error: count.support ? count.errors / count.support : null,
      falseColorRate: count.plainSupport ? count.falseColors / count.plainSupport : null }];
  }));
  const missingFamilies = objective ? families.filter((family) => objective.familyWeights[family] > 0 && !perFamily[family].support) : [];
  const unknownFamilies = objective ? families.filter((family) => !Object.hasOwn(objective.familyWeights, family) && perFamily[family].support) : [];
  const complete = Boolean(objective) && !missingFamilies.length && !unknownFamilies.length;
  return {
    weightedError: complete ? Object.entries(objective.familyWeights).reduce((sum, [family, weight]) =>
      sum + (weight ? weight * perFamily[family].error : 0), 0) : null,
    perFamily, missingFamilies, unknownFamilies, objectiveComplete: complete,
    languageObjective: objective,
  };
}

function validateCounts(count) {
  for (const key of ["support", "errors", "plainSupport", "falseColors"]) {
    if (!Number.isSafeInteger(count[key]) || count[key] < 0) throw new Error(`invalid family ${key}`);
  }
  if (count.errors > count.support || count.plainSupport > count.support ||
      count.falseColors > count.plainSupport || count.falseColors > count.errors) throw new Error("inconsistent family counts");
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return value;
}

export function assertSameProvenance(candidate, baseline) {
  for (const value of [candidate, baseline]) {
    if (!value || !/^[a-f0-9]{64}$/.test(value.shardSha256 ?? "") || value.labelSource !== "shiki-spans-v1" ||
        !Number.isInteger(value.featureVersion) || !Number.isInteger(value.tokenizerVersion) ||
        !(value.maxTokens === null || Number.isSafeInteger(value.maxTokens) && value.maxTokens > 0)) {
      throw new Error("missing or invalid direct-label evaluation provenance");
    }
  }
  if (JSON.stringify(canonical(candidate)) !== JSON.stringify(canonical(baseline))) {
    throw new Error("evaluation provenance mismatch");
  }
}

// Fixed-baseline, two-proportion tolerance using add-one smoothed rates:
// max(absolute allowance, z * sqrt(pC(1-pC)/nC + pB(1-pB)/nB)).
// Below-minimum support fails closed rather than waiving a protected guard.
export function compareLanguageMetrics(candidate, baseline, objective) {
  validateLanguageObjective(objective);
  assertSameProvenance(candidate.provenance, baseline.provenance);
  for (const metrics of [candidate, baseline]) {
    if (JSON.stringify(canonical(metrics.languageObjective)) !== JSON.stringify(canonical(objective))) {
      throw new Error("language objective provenance mismatch");
    }
    const recomputed = languageMetrics(metrics.perFamily ?? {}, objective);
    if (!recomputed.objectiveComplete || !metrics.objectiveComplete || !Number.isFinite(metrics.weightedError) ||
        Math.abs(metrics.weightedError - recomputed.weightedError) > 1e-12) throw new Error("incomplete or inconsistent language metrics");
    for (const [family, actual] of Object.entries(recomputed.perFamily)) {
      const supplied = metrics.perFamily[family];
      if (actual.error !== supplied.error || actual.falseColorRate !== supplied.falseColorRate) throw new Error(`inconsistent rates: ${family}`);
    }
  }
  const failures = [], warnings = [];
  for (const family of new Set([...Object.keys(candidate.perFamily), ...Object.keys(baseline.perFamily)])) {
    const c = candidate.perFamily[family], b = baseline.perFamily[family];
    if (!c || !b || c.support !== b.support || c.plainSupport !== b.plainSupport) {
      throw new Error(`evaluation support mismatch: ${family}`);
    }
  }
  const guards = [];
  for (const family of objective.protectedFamilies) {
    const c = candidate.perFamily[family], b = baseline.perFamily[family];
    for (const [metric, denominator, numerator, minimum, absolute] of [
      ["error", "support", "errors", objective.minSupport, objective.maxErrorIncrease],
      ["falseColorRate", "plainSupport", "falseColors", objective.minPlainSupport, objective.maxFalseColorIncrease],
    ]) {
      const n = c[denominator];
      const delta = n ? c[metric] - b[metric] : null;
      const pc = (c[numerator] + 1) / (n + 2), pb = (b[numerator] + 1) / (n + 2);
      const tolerance = n ? Math.max(absolute, objective.z * Math.sqrt((pc * (1 - pc) + pb * (1 - pb)) / n)) : null;
      const passed = n >= minimum && delta <= tolerance;
      guards.push({ family, metric, support: n, minimum, delta, tolerance, passed });
      if (!passed) {
        const message = `${family} ${metric}: ${n < minimum ? "insufficient support" : "regression"}`;
        warnings.push(message);
      }
    }
  }
  const strictGuards = compareStrictLanguages(candidate, baseline, objective);
  failures.push(...strictGuards.filter((guard) => !guard.passed).map(({ language, metric, reason }) =>
    `${language} ${metric}: ${reason}`));
  const improvement = baseline.weightedError - candidate.weightedError;
  if (!(improvement > objective.minWeightedImprovement)) failures.push("weighted error did not improve");
  return { accepted: !failures.length, criterion: "fixed-baseline-language-weighted-error-v2", improvement,
    candidateWeightedError: candidate.weightedError, baselineWeightedError: baseline.weightedError,
    guards, strictGuards, failures, warnings };
}

export function compareStrictLanguages(candidate, baseline, objective) {
  validateLanguageObjective(objective);
  const guards = [];
  const tiers = [
    [objective.strictLanguages, objective.maxStrictErrorIncrease, objective.maxStrictFalseColorIncrease, "strict"],
    [objective.matureLanguages ?? [], objective.maxMatureErrorIncrease ?? 0.01,
      objective.maxMatureFalseColorIncrease ?? 0.01, "mature"],
  ];
  for (const [languages, errorAllowance, falseColorAllowance, tier] of tiers) for (const language of languages) {
    const current = candidate.perLanguage?.[language], before = baseline.perLanguage?.[language];
    if (current) validateCounts(current);
    if (before) validateCounts(before);
    for (const [metric, denominator, numerator, allowance] of [
      ["error", "support", "errors", errorAllowance],
      ["falseColorRate", "plainSupport", "falseColors", falseColorAllowance],
    ]) {
      const support = current?.[denominator] ?? 0;
      let reason = null;
      if (!current || !before) reason = "missing language metrics";
      else if (support !== before[denominator]) reason = "support mismatch";
      else if (support < (metric === "error" ? objective.minSupport : objective.minPlainSupport)) reason = "insufficient support";
      const delta = reason || !support ? null : (current[numerator] - before[numerator]) / support;
      if (!reason && delta > allowance) reason = "regression";
      guards.push({ language, tier, metric, support, delta, tolerance: allowance, passed: reason === null, reason });
    }
  }
  return guards;
}

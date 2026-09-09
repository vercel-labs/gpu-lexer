export function classifyGitPath(path, language) {
  const lower = path.toLowerCase();
  if (isVendoredPath(lower)) return null;
  if (/(?:\/|^)(?:test|tests|testing|__tests__|spec|specs|fixtures?|examples?|samples?|demos?)(?:\/|$)/.test(lower)) return "tests";
  if (language === "markdown" || language === "mdx" || /(?:^|\/)(?:docs?|documentation)(?:\/|$)/.test(lower)) return "documentation";
  return "author";
}

export function isVendoredPath(path) {
  const lower = path.toLowerCase();
  return /(?:^|\/)(?:dist|build|built|out|output|generated|vendor|vendored|third_party|third-party|node_modules|bower_components|deps|external|extern|submodules|msys2|coverage)(?:\/|$)/.test(lower) ||
    /(?:^|\/)usr\/(?:include|lib|libexec|share)(?:\/|$)/.test(lower);
}

export function selectCorpus(records, policy, split, languageWeights, { minifiedFraction = 0 } = {}) {
  const total = policy.targetTokens[split];
  const selected = [];
  const used = new Set();
  const languageMinimums = new Map(Object.entries(policy.minimumLanguageTokens?.[split] ?? {}));
  const strata = Object.entries(policy.strata);
  for (let stratumIndex = 0; stratumIndex < strata.length; stratumIndex++) {
    const [stratum, ratio] = strata[stratumIndex];
    const quota = stratumIndex === strata.length - 1
      ? total - selected.reduce((sum, record) => sum + record.parts.length, 0)
      : Math.round(total * ratio);
    const pool = records.filter((record) => record.stratum === stratum && !used.has(record));
    let reserved = 0;
    for (const [language, remaining] of languageMinimums) {
      if (remaining <= 0 || reserved === quota) continue;
      const taken = takeMatchingRecords(pool, selected, used,
        (record) => record.language === language, Math.min(remaining, quota - reserved));
      languageMinimums.set(language, remaining - taken);
      reserved += taken;
    }
    const availableQuota = quota - reserved;
    const targets = policy.popularityWeightedStrata.includes(stratum)
      ? allocateLanguageTargets(
          availableQuota,
          languageWeights,
          stratum === "author" ? policy.minimumPopularityTokens : (policy.minimumCoverageTokens ?? 0),
        )
      : null;
    if (stratum === "compiled" && minifiedFraction > 0) {
      if (reserved) throw new Error("language minimums cannot share a split with minified compiled quotas");
      const minifiedQuota = Math.round(availableQuota * minifiedFraction);
      selected.push(...selectStratum(
        pool.filter(({ minified }) => minified),
        minifiedQuota,
        allocateLanguageTargets(minifiedQuota, languageWeights, policy.minimumCoverageTokens ?? 0),
        used,
      ));
      const regularQuota = availableQuota - minifiedQuota;
      selected.push(...selectStratum(
        pool.filter(({ minified }) => !minified),
        regularQuota,
        allocateLanguageTargets(regularQuota, languageWeights, policy.minimumCoverageTokens ?? 0),
        used,
      ));
    } else {
      selected.push(...selectStratum(pool, availableQuota, targets, used));
    }
  }
  const missing = [...languageMinimums].filter(([, remaining]) => remaining > 0);
  if (missing.length) throw new Error(`not enough held-out language tokens: ${missing.map(
    ([language, remaining]) => `${language} needs ${remaining} more`,
  ).join(", ")}`);
  return selected;
}

function allocateLanguageTargets(total, languageWeights, minimum) {
  const entries = [...languageWeights];
  const floor = Math.min(minimum, Math.floor(total / entries.length));
  const remainder = total - floor * entries.length;
  const weightTotal = entries.reduce((sum, [, weight]) => sum + weight, 0);
  const targets = new Map(entries.map(([language, weight]) => [language, floor + Math.round(remainder * weight / weightTotal)]));
  let delta = total - [...targets.values()].reduce((sum, value) => sum + value, 0);
  for (let index = 0; delta !== 0; index = (index + 1) % entries.length) {
    const language = entries[index][0];
    const step = Math.sign(delta);
    targets.set(language, targets.get(language) + step);
    delta -= step;
  }
  return targets;
}

function selectStratum(pool, quota, targets, used) {
  const selected = [];
  let remaining = quota;
  if (targets) {
    for (const [language, target] of targets) {
      remaining -= takeRecords(pool, selected, used, language, Math.min(target, remaining));
      if (remaining === 0) break;
    }
    while (remaining > 0) {
      const available = new Map([...targets].filter(([language]) => pool.some(
        (record) => !used.has(record) && record.family === language,
      )));
      if (!available.size) break;
      const weightTotal = [...available.values()].reduce((sum, weight) => sum + weight, 0);
      let changed = false;
      for (const [language, weight] of available) {
        const target = Math.max(1, Math.round(remaining * weight / weightTotal));
        const taken = takeRecords(pool, selected, used, language, Math.min(target, remaining));
        remaining -= taken;
        changed ||= taken > 0;
        if (remaining === 0) break;
      }
      if (!changed) break;
    }
  }
  while (remaining > 0) {
    const next = pool.find((record) => !used.has(record));
    if (!next) throw new Error(`not enough ${pool[0]?.stratum ?? "unknown"} tokens; need ${remaining} more`);
    remaining -= takeRecord(next, selected, used, remaining);
  }
  return selected;
}

function takeRecords(pool, selected, used, language, quota) {
  return takeMatchingRecords(pool, selected, used, (record) => record.family === language, quota);
}

function takeMatchingRecords(pool, selected, used, matches, quota) {
  const groups = new Map();
  for (const record of pool) {
    if (used.has(record) || !matches(record)) continue;
    if (!groups.has(record.sourceName)) groups.set(record.sourceName, []);
    groups.get(record.sourceName).push(record);
  }
  if (!groups.size) return 0;
  let taken = 0;
  const balancedShare = Math.ceil(quota / groups.size);
  const sourceTokens = new Map([...groups].map(([source]) => [source, 0]));
  let changed = true;
  while (taken < quota && changed) {
    changed = false;
    for (const [source, records] of groups) {
      if (sourceTokens.get(source) >= balancedShare) continue;
      const record = records.shift();
      if (!record) continue;
      const maximum = Math.min(quota - taken, balancedShare - sourceTokens.get(source));
      const count = takeRecord(record, selected, used, maximum);
      sourceTokens.set(source, sourceTokens.get(source) + count);
      taken += count;
      changed ||= count > 0;
      if (taken === quota) break;
    }
  }
  // Sparse families may not have enough data in every source. Redistribute only
  // after each available source had an equal opportunity to contribute.
  changed = true;
  while (taken < quota && changed) {
    changed = false;
    for (const records of groups.values()) {
      const record = records.shift();
      if (!record) continue;
      const count = takeRecord(record, selected, used, quota - taken);
      taken += count;
      changed ||= count > 0;
      if (taken === quota) break;
    }
  }
  return taken;
}

function takeRecord(record, selected, used, maximum) {
  const count = Math.min(maximum, record.parts.length);
  if (!count) return 0;
  used.add(record);
  if (count === record.parts.length) selected.push(record);
  else {
    const parts = record.parts.slice(0, count);
    const end = parts.at(-1).to;
    selected.push({ ...record, source: record.source.slice(0, end), parts });
  }
  return count;
}

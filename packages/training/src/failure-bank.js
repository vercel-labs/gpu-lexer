import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const failureBankPath = fileURLToPath(
  new URL("../data/generated/tree-failure-bank.json", import.meta.url),
);
const VERSION = 1;
const MAX_ENTRIES = 256;
const MAX_SOURCE_UNITS = 2_000_000;

export async function loadFailureBank(path = failureBankPath) {
  let value;
  try { value = JSON.parse(await readFile(path, "utf8")); }
  catch (error) {
    if (error?.code === "ENOENT") return { version: VERSION, entries: [] };
    throw new Error(`cannot read failure bank ${path}: ${error.message}`, { cause: error });
  }
  if (value?.version !== VERSION || !Array.isArray(value.entries)) {
    throw new Error(`unsupported failure bank ${path}`);
  }
  for (const entry of value.entries) validateEntry(entry);
  return value;
}

export async function addFailureToBank(entry, path = failureBankPath) {
  const normalized = normalizeEntry(entry);
  const current = await loadFailureBank(path);
  const key = entryKey(normalized);
  const entries = current.entries.filter((item) => entryKey(item) !== key);
  entries.push(normalized);
  const selected = selectDiverseFailures(entries, {
    maxEntries: MAX_ENTRIES, maxSourceUnits: MAX_SOURCE_UNITS,
  });
  const value = { version: VERSION, updatedAt: new Date().toISOString(), entries: selected };
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value)}\n`, { flag: "wx" });
  await rename(temporary, path);
  return value;
}

/** Round-robin families so a large pile of one language cannot evict every
 * example from smaller families. Newer unique examples win within a family.
 */
export function selectDiverseFailures(entries, {
  maxEntries = MAX_ENTRIES, maxSourceUnits = MAX_SOURCE_UNITS,
} = {}) {
  const unique = new Map();
  for (const entry of entries) {
    validateEntry(entry);
    const key = entryKey(entry);
    const previous = unique.get(key);
    if (!previous || entry.createdAt > previous.createdAt) unique.set(key, entry);
  }
  const groups = new Map();
  for (const entry of unique.values()) {
    const group = groups.get(entry.family) ?? [];
    group.push(entry);
    groups.set(entry.family, group);
  }
  for (const group of groups.values()) group.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const families = [...groups].sort((a, b) => b[1][0].createdAt.localeCompare(a[1][0].createdAt));
  const selected = [];
  let sourceUnits = 0, cursor = 0;
  while (selected.length < maxEntries && sourceUnits < maxSourceUnits) {
    let changed = false;
    for (const [, group] of families) {
      const entry = group[cursor];
      if (!entry || sourceUnits + entry.source.length > maxSourceUnits) continue;
      selected.push(entry);
      sourceUnits += entry.source.length;
      changed = true;
      if (selected.length >= maxEntries) break;
    }
    if (!changed) break;
    cursor += 1;
  }
  return selected;
}

function normalizeEntry(entry) {
  const value = {
    language: entry.language,
    family: entry.family,
    source: entry.source,
    sourceLabels: entry.sourceLabels,
    sourceFile: entry.sourceFile ?? null,
    createdAt: entry.createdAt ?? new Date().toISOString(),
    sourceSha256: createHash("sha256").update(entry.source).digest("hex"),
  };
  validateEntry(value);
  return value;
}

function validateEntry(entry) {
  if (!entry || typeof entry.language !== "string" || !entry.language ||
      typeof entry.family !== "string" || !entry.family || typeof entry.source !== "string" ||
      !entry.source.length || !Array.isArray(entry.sourceLabels) ||
      !/^[a-f0-9]{64}$/.test(entry.sourceSha256 ?? "") ||
      createHash("sha256").update(entry.source).digest("hex") !== entry.sourceSha256 ||
      typeof entry.createdAt !== "string" || !Number.isFinite(Date.parse(entry.createdAt))) {
    throw new Error("invalid tree failure-bank entry");
  }
}

function entryKey(entry) { return `${entry.language}:${entry.sourceSha256}`; }

export function resolveFailureBankPath(path) {
  return path ? resolve(path) : failureBankPath;
}

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

export const TORCH_DATA_VERSION = 2;

export async function writeTorchDataset(directory, name, records) {
  await mkdir(directory, { recursive: true });
  const recordOffsets = new Uint32Array(records.length + 1);
  let tokenCount = 0;
  let featureCount = 0;
  for (let index = 0; index < records.length; index++) {
    const record = records[index];
    tokenCount += record.targets.length;
    recordOffsets[index + 1] = tokenCount;
    for (const row of record.features) featureCount += row.length;
  }

  const featureOffsets = new Uint32Array(tokenCount + 1);
  const features = new Uint16Array(featureCount);
  const targets = new Uint8Array(tokenCount);
  const auxiliary = new Uint16Array(tokenCount);
  const supervisionWeights = new Uint8Array(tokenCount);
  const lossWeights = new Uint8Array(tokenCount);
  const metadata = new Array(records.length);
  let tokenOffset = 0;
  let featureOffset = 0;
  for (let recordIndex = 0; recordIndex < records.length; recordIndex++) {
    const record = records[recordIndex];
    metadata[recordIndex] = {
      language: record.language ?? "unknown",
      family: record.family ?? record.language ?? "unknown",
      mixedLanguage: Boolean(record.mixedLanguage),
      embeddedLanguage: Boolean(record.embeddedLanguage),
      stratum: record.stratum ?? null,
      replayWeight: record.replayWeight ?? 1,
    };
    for (let index = 0; index < record.targets.length; index++) {
      const row = record.features[index];
      features.set(row, featureOffset);
      featureOffset += row.length;
      featureOffsets[tokenOffset + 1] = featureOffset;
      targets[tokenOffset] = record.targets[index];
      auxiliary[tokenOffset] = record.auxiliary?.[index] ?? 0;
      supervisionWeights[tokenOffset] = record.supervisionWeights?.[index] ?? 255;
      lossWeights[tokenOffset] = record.lossWeights?.[index] ?? 1;
      tokenOffset += 1;
    }
  }

  const prefix = resolve(directory, name);
  const manifest = {
    format: "gpu-lexer-sparse-features",
    version: TORCH_DATA_VERSION,
    byteOrder: "little-endian",
    records: records.length,
    tokens: tokenCount,
    features: featureCount,
    recordMetadata: metadata,
  };
  await Promise.all([
    writeFile(`${prefix}.json`, `${JSON.stringify(manifest)}\n`),
    writeFile(`${prefix}.record-offsets.u32`, bytesOf(recordOffsets)),
    writeFile(`${prefix}.feature-offsets.u32`, bytesOf(featureOffsets)),
    writeFile(`${prefix}.features.u16`, bytesOf(features)),
    writeFile(`${prefix}.targets.u8`, targets),
    writeFile(`${prefix}.auxiliary.u16`, bytesOf(auxiliary)),
    writeFile(`${prefix}.supervision-weights.u8`, supervisionWeights),
    writeFile(`${prefix}.loss-weights.u8`, lossWeights),
  ]);
  return manifest;
}

export async function readTorchDatasetManifest(directory, name) {
  return JSON.parse(await readFile(resolve(directory, `${name}.json`), "utf8"));
}

function bytesOf(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

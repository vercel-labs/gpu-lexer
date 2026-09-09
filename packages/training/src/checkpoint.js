import { mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { dequantizeTensors } from "./quantization.js";

const runsRoot = new URL("../runs/", import.meta.url).pathname;
export const activeCheckpointPath = fileURLToPath(new URL("../active/", import.meta.url));

export async function loadFloatCheckpoint(argument) {
  const path = resolveRunPath(argument);
  let entries;
  try { entries = await readdir(path); }
  catch (error) {
    if (error.code === "ENOENT") throw new Error(`missing float checkpoint ${argument}; use --fresh for a new baseline`, { cause: error });
    throw error;
  }
  const metadataName = one(entries.filter((name) => /^model-.*\.json$/.test(name)), "model metadata");
  const weightsName = one(entries.filter((name) => /^weights-f32-.*\.bin$/.test(name)), "float32 weights");
  const [metadata, buffer] = await Promise.all([
    readFile(resolve(path, metadataName), "utf8").then(JSON.parse),
    readFile(resolve(path, weightsName)),
  ]);
  const bytes = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const values = new Float32Array(bytes);
  const model = {};
  for (const tensor of metadata.tensorLayout) {
    model[tensor.name] = values.slice(tensor.offset, tensor.offset + tensor.length);
  }
  return { path, metadata, model };
}

/** Load the weights users actually run, rather than the latent float values
 * that produced the promoted quantized artifact. Continuation training must
 * start here so epoch zero is the deployed model exactly.
 */
export async function loadDeployedCheckpoint(argument) {
  const path = resolveRunPath(argument);
  let entries;
  try { entries = await readdir(path); }
  catch (error) {
    if (error.code === "ENOENT") throw new Error(`missing deployed checkpoint ${argument}`, { cause: error });
    throw error;
  }
  const metadataName = one(entries.filter((name) => /^model-.*\.json$/.test(name)), "model metadata");
  const weightsName = one(entries.filter((name) => /^weights-int[456]-.*\.bin$/.test(name)), "deployed weights");
  const [metadata, buffer] = await Promise.all([
    readFile(resolve(path, metadataName), "utf8").then(JSON.parse),
    readFile(resolve(path, weightsName)),
  ]);
  if (!metadata.quantization) throw new Error(`checkpoint ${argument} has no deployed quantization metadata`);
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return { path, metadata, model: dequantizeTensors(bytes, metadata.quantization) };
}

export function resolveRunPath(argument) {
  return argument.includes("/") || argument.startsWith(".")
    ? resolve(argument)
    : resolve(runsRoot, argument);
}

/** Keep the promoted float and deployed checkpoints available in a clean clone.
 * Run histories stay ignored; this copy contains only continuation-critical
 * metadata and the two exact weight artifacts.
 */
export async function writeActiveCheckpoint(metadata, model, packedWeights) {
  await mkdir(activeCheckpointPath, { recursive: true });
  const nonce = `${process.pid}.${Date.now()}`;
  const files = {
    metadata: resolve(activeCheckpointPath, "model-active.json"),
    float: resolve(activeCheckpointPath, "weights-f32-active.bin"),
    packed: resolve(activeCheckpointPath, `weights-int${metadata.quantization.bits}-active.bin`),
  };
  const temporary = Object.fromEntries(Object.entries(files).map(([name, path]) => [name, `${path}.${nonce}.tmp`]));
  const values = new Float32Array(metadata.tensorLayout.reduce((sum, tensor) => sum + tensor.length, 0));
  for (const tensor of metadata.tensorLayout) values.set(model[tensor.name], tensor.offset);
  await Promise.all([
    writeFile(temporary.metadata, `${JSON.stringify(compactCheckpointMetadata(metadata), null, 2)}\n`, { flag: "wx" }),
    writeFile(temporary.float, new Uint8Array(values.buffer), { flag: "wx" }),
    writeFile(temporary.packed, packedWeights, { flag: "wx" }),
  ]);
  await Promise.all(Object.keys(files).filter((name) => name !== "packed").map(async (name) => {
    await rename(temporary[name], files[name]);
  }));
  for (const entry of await readdir(activeCheckpointPath)) {
    if (/^weights-int[456]-active\.bin$/.test(entry) && resolve(activeCheckpointPath, entry) !== files.packed) {
      await import("node:fs/promises").then(({ rm }) => rm(resolve(activeCheckpointPath, entry)));
    }
  }
  await rename(temporary.packed, files.packed);
}

export function compactCheckpointMetadata(metadata) {
  const {
    history: _history,
    promotionComparison: _promotionComparison,
    selection: _selection,
    config: originalConfig,
    ...compact
  } = metadata;
  const config = { ...originalConfig };
  for (const name of [
    "fixedBaseline", "fixedBaselineMetrics", "initialRun", "teacherRun", "replayMining",
    "trainShard", "verificationShard", "miningShard", "output",
    "feedbackOnly", "projectionSize", "localRefinement",
  ]) delete config[name];
  return { ...compact, config, checkpointRole: "promoted-continuation-baseline" };
}

function one(entries, label) {
  if (entries.length !== 1) throw new Error(`expected exactly one ${label}, found ${entries.length}`);
  return entries[0];
}

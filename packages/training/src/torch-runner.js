import { spawn } from "node:child_process";
import { access, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, resolve } from "node:path";

import { writeTorchDataset } from "./torch-data.js";
import { TREE_MODEL, treeTensorLayout } from "./tree-model.js";

const repositoryRoot = resolve(new URL("../../../", import.meta.url).pathname);
const bundledPython = resolve(repositoryRoot, ".venv/bin/python");
const trainerPath = new URL("../torch/train.py", import.meta.url).pathname;

export async function resolveTorchRuntime({ backend = "auto", device = "auto", python } = {}) {
  if (!new Set(["auto", "torch", "js"]).has(backend)) {
    throw new Error("backend must be auto, torch, or js");
  }
  if (backend === "js") return null;
  const candidates = python
    ? [python]
    : [process.env.GPU_LEXER_PYTHON, await exists(bundledPython) ? bundledPython : null, "python3"].filter(Boolean);
  const failures = [];
  for (const executable of [...new Set(candidates)]) {
    try {
      const result = await capture(executable, ["-c", probeSource(device)]);
      const probe = JSON.parse(result.trim());
      if (probe.device !== "cpu" || device === "cpu") return { executable, ...probe };
      failures.push(`PyTorch ${probe.torch} has no MPS or CUDA accelerator available`);
    } catch (error) {
      failures.push(error.message);
    }
  }
  if (backend === "torch") {
    throw new Error(`PyTorch backend unavailable: ${failures[0] ?? "PyTorch is not installed"}. Run pnpm train:gpu:setup once, then retry.`);
  }
  return null;
}

export async function runTorchTraining({
  runtime,
  config,
  model,
  pretrainRecords,
  fineTuneRecords,
  hardRecords,
  requiredRecords,
  verificationRecords,
  pretrainClassWeights,
  fineTuneClassWeights,
  pretrainAuxiliaryPositiveWeights,
  fineTuneAuxiliaryPositiveWeights,
  classNames,
  auxiliaryNames,
  sameVerification,
  teacherModel,
  teacherConfig,
  log = console.log,
}) {
  const stagingRoot = resolve(tmpdir(), "gpu-lexer-torch-");
  const staging = await mkdtemp(stagingRoot);
  log(`PyTorch recovery files: ${staging}`);
  try {
    const { usesNaturalTrain, hasFineDataset } = trainingDatasetPlan(config);
    const trainRecords = usesNaturalTrain ? pretrainRecords : fineTuneRecords;
    log(`exporting sparse training features for PyTorch (${trainRecords.length.toLocaleString("en-US")} pretrain streams` +
      `${hasFineDataset ? ` / ${fineTuneRecords.length.toLocaleString("en-US")} fine-tune streams` : ""})`);
    await writeTorchDataset(staging, "train", trainRecords);
    if (hasFineDataset) await writeTorchDataset(staging, "fine", fineTuneRecords);
    if (hardRecords?.length) await writeTorchDataset(staging, "hard", hardRecords);
    if (requiredRecords?.length) await writeTorchDataset(staging, "required", requiredRecords);
    await writeTorchDataset(staging, "verification", verificationRecords);

    const tensorLayout = createTensorLayout(model, config, classNames.length, auxiliaryNames.length);
    await writeFile(resolve(staging, "initial.f32"), concatenateModel(model, tensorLayout));
    const teacherTensorLayout = teacherModel
      ? createTensorLayout(teacherModel, teacherConfig, classNames.length, auxiliaryNames.length)
      : null;
    if (teacherModel) {
      await writeFile(resolve(staging, "teacher.f32"), concatenateModel(teacherModel, teacherTensorLayout));
    }
    await writeFile(resolve(staging, "config.json"), `${JSON.stringify({
      ...pick(config, [
        "model",
        "epochs", "fineTuneEpochs", "batchTokens", "learningRate", "finalLearningRate",
        "fineTuneLearningRate", "fineTuneFinalLearningRate", "gradientClip", "patience",
        "headTuneEpochs", "stagedFineTune", "scanFineTuneLearningRate", "scanFineTuneFinalLearningRate",
        "minDelta", "seed", "progressEvery", "replayFraction", "repeatHardReplay", "maxHardReplayRepeats",
        "mixedF1Tolerance", "qatEpochs",
        "emaDecay", "emaStartEpoch",
        "inputSize", "hiddenSize", "classifierSize",
        "weightBits", "treeContext", "localRadius", "teacherMode", "polishMode",
        "languageObjective", "classWeightPower", "calibrationEpochs", "selectionMetric", "fixedBaselineMetrics",
        "agreementEpochs", "agreementLearningRate", "agreementFinalLearningRate",
        "calibrationLearningRate", "calibrationFinalLearningRate",
        "trainingLanguageMultipliers",
        "focusedReplay", "focusedReplaySteps", "focusedReplayLearningRate",
        "focusedReplayFinalLearningRate", "initialIsBaseline",
        "boundaryLossMultiplier",
        "migrationExpectedAccuracy",
        "distillationWeight", "distillationTemperature", "distillationWarmupEpochs",
        "distillationRampEpochs",
      ]),
      device: runtime.device,
      sameVerification,
      hasInitial: Boolean(config.initialRun),
      classNames,
      auxiliaryNames,
      classWeights: [...fineTuneClassWeights],
      auxiliaryPositiveWeights: [...fineTuneAuxiliaryPositiveWeights],
      pretrainClassWeights: [...pretrainClassWeights],
      fineTuneClassWeights: [...fineTuneClassWeights],
      pretrainAuxiliaryPositiveWeights: [...pretrainAuxiliaryPositiveWeights],
      fineTuneAuxiliaryPositiveWeights: [...fineTuneAuxiliaryPositiveWeights],
      hasPretrain: hasFineDataset,
      hasFineDataset,
      auxiliaryLossWeight: 0.15,
      tensorLayout,
      hasTeacher: Boolean(teacherModel),
      teacherTensorLayout,
      teacherModel: teacherConfig?.model ?? TREE_MODEL,
      teacherWeightBits: teacherConfig?.weightBits ?? 6,
      teacherTreeContext: teacherConfig?.treeContext ?? "tree",
      teacherFloat: teacherConfig?.teacherFloat ?? false,
      hasHardRecords: Boolean(hardRecords?.length),
      hasRequiredRecords: Boolean(requiredRecords?.length),
    }, null, 2)}\n`);

    log(`starting PyTorch ${runtime.torch} training on ${runtime.device.toUpperCase()}`);
    await inherit(runtime.executable, [trainerPath, "--directory", staging]);
    const result = JSON.parse(await readFile(resolve(staging, "result.json"), "utf8"));
    const [best, bestOverall, bestMixed] = await Promise.all([
      readCandidate(staging, "selected", tensorLayout, result.selected),
      readCandidate(staging, "best-overall", tensorLayout, result.bestOverall),
      readCandidate(staging, "best-mixed", tensorLayout, result.bestMixed),
    ]);
    return {
      history: result.history, best, bestOverall, bestMixed, initialMetrics: result.initialMetrics,
      batchTokens: result.batchTokens, selection: result.selection,
      recoveryDirectory: staging,
    };
  } catch (error) {
    throw new Error(`${error.message}; recovery files retained at ${staging}`, { cause: error });
  }
}

export function trainingDatasetPlan(config) {
  return {
    usesNaturalTrain: Boolean(config.polishMode) || config.fineTuneEpochs < config.epochs,
    hasFineDataset: !config.polishMode && config.fineTuneEpochs < config.epochs,
  };
}

function createTensorLayout(model, config, classCount, auxiliaryCount) {
  if (config.model !== TREE_MODEL) throw new Error("only the hierarchical-tree trainer is supported");
  return treeTensorLayout(model, config, classCount, auxiliaryCount);
}

function concatenateModel(model, tensorLayout) {
  const length = tensorLayout.reduce((sum, tensor) => sum + tensor.length, 0);
  const values = new Float32Array(length);
  for (const tensor of tensorLayout) values.set(model[tensor.name], tensor.offset);
  return new Uint8Array(values.buffer);
}

async function readCandidate(directory, name, tensorLayout, summary) {
  const buffer = await readFile(resolve(directory, `${name}.f32`));
  const bytes = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const values = new Float32Array(bytes);
  const model = Object.fromEntries(tensorLayout.map((tensor) => [
    tensor.name,
    values.slice(tensor.offset, tensor.offset + tensor.length),
  ]));
  return { ...summary, model };
}

function pick(object, names) {
  return Object.fromEntries(names.map((name) => [name, object[name]]));
}

async function exists(path) {
  try { await access(path); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

function capture(command, arguments_) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, arguments_, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("exit", (code) => code === 0
      ? resolvePromise(stdout)
      : reject(new Error(stderr.trim() || `${command} exited with ${code}`)));
  });
}

function inherit(command, arguments_) {
  return new Promise((resolvePromise, reject) => {
    const path = [dirname(resolve(command)), process.env.PATH].filter(Boolean).join(delimiter);
    const child = spawn(command, arguments_, { cwd: repositoryRoot, stdio: "inherit", env: { ...process.env, PATH: path } });
    child.on("error", reject);
    child.on("exit", (code, signal) => code === 0
      ? resolvePromise()
      : reject(new Error(`PyTorch trainer failed${signal ? ` (${signal})` : ` with exit ${code}`}`)));
  });
}

function probeSource(device) {
  return [
    "import json, warnings",
    "warnings.filterwarnings('ignore', message='Failed to initialize NumPy')",
    "import torch",
    `requested = ${JSON.stringify(device)}`,
    "available = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')",
    "selected = available if requested == 'auto' else requested",
    "ok = selected == 'cpu' or (selected == 'mps' and torch.backends.mps.is_available()) or (selected == 'cuda' and torch.cuda.is_available())",
    "assert ok, f'requested accelerator {selected} is unavailable'",
    "print(json.dumps({'torch': torch.__version__, 'device': selected}))",
  ].join("; ");
}

export const __testing = Object.freeze({ createTensorLayout, concatenateModel });

import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { resolve } from "node:path";

const repositoryRoot = resolve(new URL("../../../", import.meta.url).pathname);
const environment = resolve(repositoryRoot, ".venv");
const environmentPython = resolve(environment, "bin/python");
const requirements = new URL("../requirements-torch.txt", import.meta.url).pathname;

const python = await findPython();
if (!python) {
  throw new Error("Python 3.11–3.13 is required by PyTorch. Install Python 3.12 (brew install python@3.12), then rerun this command.");
}

if (!(await exists(environmentPython))) {
  console.log(`Creating ${environment} with ${python}`);
  await run(python, ["-m", "venv", environment]);
}
const publicIndex = ["--index-url", "https://pypi.org/simple"];
await run(environmentPython, ["-m", "pip", "install", ...publicIndex, "--upgrade", "pip"]);
await run(environmentPython, ["-m", "pip", "install", ...publicIndex, "-r", requirements]);
await run(environmentPython, ["-c", [
  "import warnings",
  "warnings.filterwarnings('ignore', message='Failed to initialize NumPy')",
  "import torch",
  "device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')",
  "print(f'PyTorch {torch.__version__} ready on {device.upper()}')",
].join("; ")]);

async function findPython() {
  const candidates = [
    process.env.GPU_LEXER_BOOTSTRAP_PYTHON,
    "/opt/homebrew/opt/python@3.13/bin/python3.13",
    "/opt/homebrew/opt/python@3.12/bin/python3.12",
    "python3.13",
    "python3.12",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await run(candidate, ["-c", "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13)"], false);
      return candidate;
    } catch {}
  }
  return null;
}

function run(command, arguments_, inherit = true) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, arguments_, { cwd: repositoryRoot, stdio: inherit ? "inherit" : "ignore" });
    child.on("error", reject);
    child.on("exit", (code, signal) => code === 0
      ? resolvePromise()
      : reject(new Error(`${command} failed${signal ? ` (${signal})` : ` with exit ${code}`}`)));
  });
}

async function exists(path) {
  try { await access(path); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

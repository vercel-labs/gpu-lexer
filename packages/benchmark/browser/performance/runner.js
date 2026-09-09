const copies = 10;
const sourceUrl = "https://unpkg.com/three@0.97.0/build/three.min.js";
const engines = [
  ["gpu-lexer", "./gpu-lexer.js?run=10"],
  ["Prism.js", "./prism.js?run=10"],
  ["Highlight.js", "./highlight.js?run=10"],
  ["Sugar High", "./sugar-high.js?run=10"],
  ["Starry Night", "./starry-night.js?run=10"],
  ["Shiki", "./shiki.js?run=10"],
];
const status = document.querySelector("#status");
const result = document.querySelector("#result");

run().catch((error) => {
  status.textContent = "failed";
  result.textContent = error?.stack ?? String(error);
  const output = { status: "failed", error: error?.message ?? String(error) };
  globalThis.__benchmarkResult = output;
  void report(output);
});

async function run() {
  const response = await fetch(sourceUrl);
  if (!response.ok) throw new Error(`source download failed: HTTP ${response.status}`);
  const unit = await response.text();
  if (unit.length !== 555_650) throw new Error(`unexpected three.min.js length: ${unit.length}`);
  const source = unit.repeat(copies);
  const times = {};
  for (const [name, workerUrl] of engines) {
    status.textContent = `${name}: warming and measuring…`;
    await report({ status: status.textContent, copies, characters: source.length, times });
    times[name] = await runWorker(name, workerUrl, source);
    result.textContent = JSON.stringify(times, null, 2);
  }
  const output = {
    copies,
    characters: source.length,
    source: sourceUrl,
    warmups: 1,
    measuredRuns: 1,
    userAgent: navigator.userAgent,
    times,
  };
  status.textContent = "done";
  result.textContent = JSON.stringify(output, null, 2);
  globalThis.__benchmarkResult = output;
  await report({ status: "done", ...output });
}

function report(value) {
  return fetch("/result", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(value),
  });
}

function runWorker(name, url, source) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(url, { type: "module" });
    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(`${name} worker failed: ${data.error}`));
      else resolve(data.elapsed);
    };
    worker.onerror = ({ message, filename, lineno }) => {
      worker.terminate();
      const location = filename ? ` (${filename}:${lineno})` : "";
      reject(new Error(`${name} worker failed${location}: ${message}`));
    };
    worker.postMessage(source);
  });
}

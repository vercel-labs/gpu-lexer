const response = await fetch("https://unpkg.com/three@0.97.0/build/three.min.js");
const source = (await response.text()).repeat(10);
const worker = new Worker("./profile.js", { type: "module" });
worker.onmessage = async ({ data }) => {
  const output = data.error ? { status: "failed", error: data.error } : { status: "done", ...data };
  document.querySelector("pre").textContent = JSON.stringify(output, null, 2);
  await fetch("/result", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(output),
  });
  worker.terminate();
};
worker.postMessage(source);

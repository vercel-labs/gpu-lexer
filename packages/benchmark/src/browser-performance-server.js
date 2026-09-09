import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve } from "node:path";

const root = resolve(process.argv[2] ?? "/tmp/gpu-lexer-browser-performance");
const port = Number(process.argv[3] ?? 4174);
let latest = { status: "waiting for browser" };

createServer(async (request, response) => {
  const pathname = new URL(request.url, `http://${request.headers.host}`).pathname;
  if (pathname === "/result") {
    if (request.method === "POST") {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      latest = JSON.parse(Buffer.concat(chunks));
      console.log(JSON.stringify(latest));
    }
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify(latest));
    return;
  }
  const file = resolve(root, pathname === "/" ? "index.html" : `.${pathname}`);
  if (!file.startsWith(root)) {
    response.writeHead(403).end();
    return;
  }
  try {
    await stat(file);
    response.setHeader("cache-control", "no-store");
    response.setHeader("content-type", mime(extname(file)));
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
}).listen(port, () => console.log(`http://127.0.0.1:${port}`));

function mime(extension) {
  if (extension === ".html") return "text/html; charset=utf-8";
  if (extension === ".js") return "text/javascript; charset=utf-8";
  return "application/octet-stream";
}

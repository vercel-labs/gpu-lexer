import { createReadStream } from "node:fs";
import { createGunzip } from "node:zlib";

/** JSONL uses LF delimiters; U+2028 and U+2029 are valid characters inside JSON strings. */
export async function* readShard(path) {
  const source = createReadStream(path);
  const input = source.pipe(createGunzip());
  source.on("error", (error) => input.destroy(error));
  input.setEncoding("utf8");
  let pending = "", lineNumber = 0;
  try {
    for await (const chunk of input) {
      pending += chunk;
      let from = 0, end;
      while ((end = pending.indexOf("\n", from)) !== -1) {
        const line = pending.slice(from, end);
        from = end + 1;
        lineNumber++;
        if (line.trim()) yield parse(line);
      }
      pending = pending.slice(from);
    }
    if (pending.trim()) {
      lineNumber++;
      yield parse(pending);
    }
  } finally {
    source.destroy();
    input.destroy();
  }

  function parse(line) {
    try { return JSON.parse(line); }
    catch (error) {
      throw new SyntaxError(`${path}:${lineNumber}: ${error.message}`, { cause: error });
    }
  }
}

import { profileHighlight } from "../../src/browser-profile.js";
import { createInspectableRuntime as createRuntime } from "../../../core/src/gpu.js";

const runner = createRuntime();

self.onmessage = async ({ data: source }) => {
  try {
    await profileHighlight(source, { compareReadback: false, runner });
    self.postMessage(await profileHighlight(source, { compareReadback: false, runner }));
  } catch (error) {
    self.postMessage({ error: error?.stack ?? error?.message ?? String(error) });
  }
};

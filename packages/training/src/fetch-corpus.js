import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { promisify } from "node:util";

import {
  corpusRoot,
  gitDirectoryName,
  packageDirectoryName,
  readCorpusManifest,
  readLanguagePopularity,
  validateCorpusManifest,
} from "./corpus.js";
import { readWebsiteExamples, websiteExampleFileName } from "./website-examples.js";

const execute = promisify(execFile);
const downloads = new URL("downloads/", corpusRoot);
const repositories = new URL("repositories/", corpusRoot);
const packages = new URL("packages/", corpusRoot);
const websiteExamples = new URL("website-examples/", corpusRoot);
const [manifest, popularity] = await Promise.all([readCorpusManifest(), readLanguagePopularity()]);
validateCorpusManifest(manifest, popularity);
await Promise.all([downloads, repositories, packages, websiteExamples].map((url) => mkdir(url, { recursive: true })));
const requestedSplit = parseSplit(process.argv.slice(2));

for (const split of requestedSplit ? [requestedSplit] : ["train", "verification", "mining"]) {
  for (const entry of manifest[split].git) await fetchRepository(split, entry);
  for (const entry of manifest[split].npm) await fetchPackage(split, entry);
  if (split === "verification") {
    const examples = readWebsiteExamples();
    for (let index = 0; index < examples.length; index += 8) {
      await Promise.all(examples.slice(index, index + 8).map(fetchWebsiteExample));
    }
  }
}

function parseSplit(arguments_) {
  const index = arguments_.indexOf("--split");
  if (index < 0) return null;
  const split = arguments_[index + 1];
  if (!["train", "verification", "mining"].includes(split)) throw new Error("--split must be train, verification, or mining");
  return split;
}

async function fetchRepository(split, entry) {
  const destination = new URL(`${gitDirectoryName(entry)}/`, repositories);
  try {
    const provenance = JSON.parse(await readFile(new URL("provenance.json", destination), "utf8"));
    if (provenance.repo === entry.repo && provenance.commit === entry.commit) {
      console.log(`reuse ${split}/${entry.repo}@${entry.commit.slice(0, 12)}`);
      return;
    }
  } catch {}

  await mkdir(destination, { recursive: true });
  const archive = new URL(`${gitDirectoryName(entry)}.tar.gz`, downloads);
  await execute("curl", ["-sS", "-L", `https://github.com/${entry.repo}/archive/${entry.commit}.tar.gz`, "-o", archive.pathname]);
  await execute("tar", ["-xzf", archive.pathname, "-C", destination.pathname, "--strip-components=1"]);
  await writeFile(new URL("provenance.json", destination), `${JSON.stringify({
    kind: "git", split, repo: entry.repo, commit: entry.commit, declaredLicense: entry.license,
    archive: `https://github.com/${entry.repo}/archive/${entry.commit}.tar.gz`, fetchedAt: new Date().toISOString(),
  }, null, 2)}\n`);
  console.log(`fetch ${split}/${entry.repo}@${entry.commit.slice(0, 12)}`);
}

async function fetchPackage(split, entry) {
  const directoryName = packageDirectoryName(entry);
  const destination = new URL(`${directoryName}/`, packages);
  try {
    const packageJson = JSON.parse(await readFile(new URL("package/package.json", destination), "utf8"));
    if (packageJson.name === entry.name && packageJson.version === entry.version) {
      console.log(`reuse ${split}/${entry.name}@${entry.version}`);
      return;
    }
  } catch {}

  const { stdout } = await execute("npm", [
    "pack", `${entry.name}@${entry.version}`, "--pack-destination", downloads.pathname, "--json", "--ignore-scripts",
  ], { maxBuffer: 1024 * 1024 });
  const [{ filename, integrity }] = JSON.parse(stdout);
  await mkdir(destination, { recursive: true });
  await execute("tar", ["-xzf", new URL(filename, downloads).pathname, "-C", destination.pathname]);
  await writeFile(new URL("provenance.json", destination), `${JSON.stringify({
    kind: "npm", split, name: entry.name, version: entry.version, declaredLicense: entry.license, integrity,
    fetchedAt: new Date().toISOString(),
  }, null, 2)}\n`);
  console.log(`fetch ${split}/${entry.name}@${entry.version}`);
}

async function fetchWebsiteExample(entry) {
  const name = websiteExampleFileName(entry);
  const destination = new URL(name, websiteExamples);
  const provenanceUrl = new URL(`${name}.json`, websiteExamples);
  try {
    const [provenance, bytes] = await Promise.all([
      readFile(provenanceUrl, "utf8").then(JSON.parse),
      readFile(destination),
    ]);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    if (provenance.id === entry.id && provenance.url === (entry.url ?? null) && provenance.sha256 === sha256) {
      console.log(`reuse verification/website/${entry.id}`);
      return;
    }
  } catch {}

  if (entry.inlineSource !== undefined) await writeFile(destination, entry.inlineSource);
  else await execute("curl", ["-f", "-sS", "-L", entry.url, "-o", destination.pathname]);
  const bytes = await readFile(destination);
  if (!bytes.length) throw new Error(`empty website example: ${entry.id}`);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  await writeFile(provenanceUrl, `${JSON.stringify({
    kind: "website-example", split: "verification", id: entry.id, url: entry.url ?? null,
    fileName: entry.fileName, shiki: entry.shiki, family: entry.family, sha256,
    bytes: bytes.length, fetchedAt: new Date().toISOString(),
  }, null, 2)}\n`);
  console.log(`fetch verification/website/${entry.id}`);
}

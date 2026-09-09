import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  corpusConfigurationDigest, corpusRoot, languageFamily, languageForPath, popularityWeights, readCorpusManifest,
  readLanguagePopularity, validateCorpusManifest,
} from "../src/corpus.js";
import { classifyGitPath, isVendoredPath, selectCorpus } from "../src/corpus-selection.js";
import {
  readWebsiteExamples, WEBSITE_EXAMPLE_SOURCE_LIMIT, websiteExampleMatchesCandidate,
} from "../src/website-examples.js";

test("mixed corpus pins disjoint Git commits and npm versions", async () => {
  const [manifest, popularity] = await Promise.all([readCorpusManifest(), readLanguagePopularity()]);
  assert.doesNotThrow(() => validateCorpusManifest(manifest, popularity));
  assert.deepEqual(manifest.policy.strata, { author: 0.70, tests: 0.15, documentation: 0.10, compiled: 0.05 });
  assert.equal(manifest.policy.targetTokens.train, 3_000_000);
  assert.equal(manifest.policy.targetTokens.verification, 240_000);
  assert.equal(manifest.policy.minimumTrainingFamilyTokens, 10_000);
  assert.equal(manifest.policy.minimumLanguageTokens.verification.tsx, 2_000);
  assert.equal(manifest.train.minifiedCompiledFraction, 0.5);
  assert.equal(manifest.mining.minimumFamilyTokens, 2_000);
  assert.equal(manifest.policy.trainingSamplingWeights.plpgsql, 10);
  assert.equal(manifest.policy.supplementalSamplingWeights.plpgsql, 10);
  assert.ok(manifest.policy.maxFilesPerMiningRepository > manifest.policy.maxFilesPerRepository);
  assert.ok(manifest.train.git.every((entry) => /^[\da-f]{40}$/.test(entry.commit)));
  assert.ok(manifest.verification.git.every((entry) => /^[\da-f]{40}$/.test(entry.commit)));
  assert.ok(manifest.mining.git.every((entry) => /^[\da-f]{40}$/.test(entry.commit)));
  const repositories = ["train", "verification", "mining"].flatMap((split) => manifest[split].git.map((entry) => entry.repo));
  assert.equal(new Set(repositories).size, repositories.length);
});

test("fine-tuning uses tiered floors and shared quotas for related formats", async () => {
  const policy = JSON.parse(await readFile(new URL("../data/fine-tune-policy.json", import.meta.url), "utf8"));
  assert.equal(policy.defaultMinimumTokens, 5_000);
  assert.equal(policy.minimumTokens.javascript, 30_000);
  assert.equal(policy.minimumTokens.svelte, 15_000);
  assert.equal(policy.minimumTokens["vim-script"], 1_000);
  assert.equal(policy.aliases.procfile, "shell");
  assert.equal(policy.aliases.tsql, "plpgsql");
  assert.equal(policy.aliases.shaderlab, "shader");
  assert.equal(policy.aliases.smarty, "html");
});

test("sealed audit repositories are pinned and disjoint from all development splits", async () => {
  const [manifest, audit] = await Promise.all([
    readCorpusManifest(),
    readFile(new URL("../data/audit.json", import.meta.url), "utf8").then(JSON.parse),
  ]);
  assert.equal(audit.sealed, true);
  const development = new Set(["train", "verification", "mining"].flatMap(
    (split) => manifest[split].git.map(({ repo }) => repo),
  ));
  assert.equal(new Set(audit.repositories.map(({ repo }) => repo)).size, audit.repositories.length);
  assert.ok(audit.repositories.every(({ repo, commit }) =>
    !development.has(repo) && /^[\da-f]{40}$/.test(commit)));
});

test("corpus configuration digests cover policy, sources, and language weights", async () => {
  const [manifest, popularity] = await Promise.all([readCorpusManifest(), readLanguagePopularity()]);
  const digest = corpusConfigurationDigest(manifest, popularity, "train");
  assert.match(digest, /^[\da-f]{64}$/);
  assert.notEqual(digest, corpusConfigurationDigest({ ...manifest, train: manifest.verification }, popularity, "train"));
  const changedVerificationFloor = structuredClone(manifest);
  changedVerificationFloor.policy.minimumLanguageTokens.verification.tsx += 1;
  assert.equal(corpusConfigurationDigest(changedVerificationFloor, popularity, "train"), digest);
  assert.notEqual(corpusConfigurationDigest(changedVerificationFloor, popularity, "verification"),
    corpusConfigurationDigest(manifest, popularity, "verification"));
});

test("every website code example is reserved for verification", async () => {
  const examples = readWebsiteExamples();
  const [demoSource, probes] = await Promise.all([
    readFile(new URL("../../../apps/website/app/demo.tsx", import.meta.url), "utf8"),
    readFile(new URL("../data/language-probes.json", import.meta.url), "utf8").then(JSON.parse),
  ]);
  const demoBlock = demoSource.slice(demoSource.indexOf("const demoFolders"),
    demoSource.indexOf("] as const", demoSource.indexOf("const demoFolders")));
  assert.equal(examples.length, [...demoBlock.matchAll(/\burl:\s*'/g)].length + probes.length + 1);
  assert.equal(new Set(examples.map(({ id }) => id)).size, examples.length);
  const websiteLimit = Number(demoSource.match(/MAX_RENDERED_SOURCE_LENGTH = ([\d_]+)/)?.[1].replaceAll("_", ""));
  assert.equal(WEBSITE_EXAMPLE_SOURCE_LIMIT, websiteLimit);
  assert.ok(examples.some(({ id }) => id === "api-example"));
  assert.ok(examples.some(({ id }) => id === "demo-three"));
  assert.ok(examples.some(({ id }) => id === "probe-wgsl"));
  const react = examples.find(({ id }) => id === "demo-react");
  assert.equal(websiteExampleMatchesCandidate(react, {
    origin: "npm", sourceName: "react", path: "cjs/react.development.js",
  }), true);
  assert.equal(websiteExampleMatchesCandidate(react, {
    origin: "npm", sourceName: "react", path: "cjs/react.production.js",
  }), false);
});

test("mining reserves half its tokens for generated minified web code", async () => {
  const manifest = await readCorpusManifest();
  assert.equal(manifest.policy.miningMinifiedFraction, 0.5);
});

test("language weights are pinned percentages from the Innovation Graph", async () => {
  const popularity = await readLanguagePopularity();
  assert.equal(popularity.source.period, "2026-Q1");
  assert.match(popularity.source.page, /innovationgraph\.github\.com/);
  assert.ok(Math.abs(popularity.languages.reduce((sum, entry) => sum + entry.percent, 0) - 100) < 0.01);
  const weights = popularityWeights(popularity);
  assert.deepEqual(popularity.languages.slice(0, 5).map(({ github }) => github), [
    "JavaScript", "Python", "TypeScript", "Shell", "Dockerfile",
  ]);
  assert.equal(popularity.languages.filter(({ supplemental }) => !supplemental).length, 50);
  assert.equal(weights.get("html"), 0);
  assert.equal(popularityWeights(popularity, { html: 4 }).get("html"), 4);
  assert.equal(weights.get("javascript"), 19.8731);
  assert.equal(languageFamily("tsx"), "typescript");
});

test("Astro and Diff are supplemental language-agnostic training families", async () => {
  const popularity = await readLanguagePopularity();
  const weights = popularityWeights(popularity);
  assert.ok(popularity.languages.some(({ family, supplemental }) => family === "astro" && supplemental));
  assert.ok(popularity.languages.some(({ family, supplemental }) => family === "diff" && supplemental));
  assert.equal(weights.get("astro"), 0);
  assert.equal(weights.get("diff"), 0);
  assert.equal(popularityWeights(popularity, { python: 24, html: 10, diff: 2 }).get("python"), 24);
  assert.equal(popularityWeights(popularity, { python: 24, html: 10, diff: 2 }).get("html"), 10);
  assert.equal(popularityWeights(popularity, { python: 24, html: 10, diff: 2 }).get("diff"), 2);
});

test("Diff source and fixture paths use the Diff teacher", () => {
  assert.deepEqual(languageForPath("changes/fix.patch"), { language: "diff", family: "diff" });
  assert.deepEqual(languageForPath("tests/snippets/diff/unified.txt"), { language: "diff", family: "diff" });
  assert.deepEqual(languageForPath("test/markup/diff/comments.txt"), { language: "diff", family: "diff" });
});

test("Git paths are assigned to author, test, and documentation strata", () => {
  assert.equal(classifyGitPath("src/index.ts", "typescript"), "author");
  assert.equal(classifyGitPath("lib/rake.rb", "ruby"), "author");
  assert.equal(classifyGitPath("tests/fixtures/tricky.tsx", "tsx"), "tests");
  assert.equal(classifyGitPath("examples/basic.js", "javascript"), "tests");
  assert.equal(classifyGitPath("README.md", "markdown"), "documentation");
  assert.equal(classifyGitPath("docs/guide.html", "html"), "documentation");
  assert.equal(classifyGitPath("dist/index.js", "javascript"), null);
  assert.equal(classifyGitPath("msys2/usr/share/bison/skeletons/glr.cc", "cpp"), null);
  assert.equal(classifyGitPath("vendor/tests/example.js", "javascript"), null);
  assert.equal(isVendoredPath("src/index.ts"), false);
});

test("family quotas draw from independent sources before filling from one source", () => {
  const record = (sourceName, id) => ({
    stratum: "author", family: "cpp", sourceName, path: `${id}.cpp`, source: "x".repeat(100),
    parts: Array.from({ length: 100 }, (_, index) => ({ from: index, to: index + 1 })),
  });
  const selected = selectCorpus([
    record("first", "a"), record("first", "b"), record("second", "c"), record("second", "d"),
  ], {
    targetTokens: { verification: 200 }, strata: { author: 1 }, popularityWeightedStrata: ["author"],
    minimumPopularityTokens: 0,
  }, "verification", new Map([["cpp", 1]]));
  const tokens = Object.fromEntries(["first", "second"].map((source) => [source, selected
    .filter(({ sourceName }) => sourceName === source)
    .reduce((sum, item) => sum + item.parts.length, 0)]));
  assert.deepEqual(tokens, { first: 100, second: 100 });
});

test("verification selection reserves exact-language coverage inside broader family quotas", () => {
  const record = (language, id, tokens = 100) => ({
    stratum: "author", family: "typescript", language, sourceName: id, path: `${id}.tsx`,
    source: "x".repeat(tokens), parts: Array.from({ length: tokens }, (_, index) => ({ from: index, to: index + 1 })),
  });
  const selected = selectCorpus([
    record("typescript", "ts-a", 200), record("typescript", "ts-b", 200),
    record("tsx", "tsx-a", 100), record("tsx", "tsx-b", 100),
  ], {
    targetTokens: { verification: 300 }, strata: { author: 1 }, popularityWeightedStrata: ["author"],
    minimumPopularityTokens: 0, minimumLanguageTokens: { verification: { tsx: 150 } },
  }, "verification", new Map([["typescript", 1]]));
  assert.equal(selected.reduce((sum, item) => sum + item.parts.length, 0), 300);
  assert.equal(selected.filter(({ language }) => language === "tsx")
    .reduce((sum, item) => sum + item.parts.length, 0), 150);
});

test("selection enforces exact token strata", () => {
  const strata = { author: 0.70, tests: 0.15, documentation: 0.10, compiled: 0.05 };
  const records = Object.keys(strata).flatMap((stratum) => Array.from({ length: 4 }, (_, index) => ({
    stratum, family: index % 2 ? "typescript" : "javascript", sourceName: `${stratum}-${index}`,
    source: "x".repeat(100), parts: Array.from({ length: 100 }, (_, token) => ({ from: token, to: token + 1 })),
  })));
  const selected = selectCorpus(records, {
    targetTokens: { train: 200 }, strata, popularityWeightedStrata: ["author", "tests", "documentation", "compiled"],
    minimumPopularityTokens: 0,
  }, "train", new Map([["javascript", 60], ["typescript", 40]]));
  const counts = Object.fromEntries(Object.keys(strata).map((stratum) => [
    stratum, selected.filter((record) => record.stratum === stratum).reduce((sum, record) => sum + record.parts.length, 0),
  ]));
  assert.deepEqual(counts, { author: 140, tests: 30, documentation: 20, compiled: 10 });
});

test("training selection reserves compiled capacity for generated minified npm source", () => {
  const record = (id, minified) => ({
    stratum: "compiled", family: "javascript", sourceName: id, minified,
    source: "x".repeat(100), parts: Array.from({ length: 100 }, (_, token) => ({ from: token, to: token + 1 })),
  });
  const selected = selectCorpus([
    record("min-a", true), record("min-b", true), record("source-a", false), record("source-b", false),
  ], {
    targetTokens: { train: 200 }, strata: { compiled: 1 }, popularityWeightedStrata: ["compiled"],
    minimumPopularityTokens: 0, minimumCoverageTokens: 0,
  }, "train", new Map([["javascript", 1]]), { minifiedFraction: 0.5 });
  assert.equal(selected.filter(({ minified }) => minified).reduce((sum, item) => sum + item.parts.length, 0), 100);
  assert.equal(selected.filter(({ minified }) => !minified).reduce((sum, item) => sum + item.parts.length, 0), 100);
});

test("generated shards enforce the source mix and language coverage", async () => {
  const [manifest, summary] = await Promise.all([
    readCorpusManifest(),
    readFile(new URL("shards/summary.json", corpusRoot), "utf8").then(JSON.parse),
  ]);
  for (const split of ["train", "verification"]) {
    const result = summary.splits[split];
    const distribution = result.base ?? result;
    assert.equal(distribution.tokens, manifest.policy.targetTokens[split]);
    for (const [stratum, ratio] of Object.entries(manifest.policy.strata)) {
      assert.equal(distribution.strata[stratum], distribution.tokens * ratio);
    }
    for (const language of manifest.policy.requiredLanguagesPerSplit) {
      assert.ok(distribution.languageFamilies[language] > 0, `${split} must contain ${language}`);
    }
    for (const [language, minimum] of Object.entries(manifest.policy.minimumLanguageTokens?.[split] ?? {})) {
      assert.ok(distribution.languages[language] >= minimum, `${split} must contain at least ${minimum} ${language} tokens`);
    }
    for (const construct of manifest.policy.constructCoverage.required) {
      assert.ok(
        distribution.constructs?.[construct] >= manifest.policy.constructCoverage.minimumTokens[split],
        `${split} must contain enough ${construct}`,
      );
    }
    assert.equal(distribution.origins.git, distribution.tokens * 0.95);
    assert.equal(distribution.origins.npm, distribution.tokens * 0.05);
  }
  const examples = readWebsiteExamples();
  assert.equal(summary.splits.verification.websiteExamples.files, examples.length);
  assert.equal(summary.splits.verification.origins.website,
    summary.splits.verification.websiteExamples.tokens);
});

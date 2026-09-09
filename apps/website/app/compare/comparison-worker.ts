import type PrismType from 'prismjs'
import { createBundledHighlighter } from 'shiki/core'
import { createOnigurumaEngine } from 'shiki/engine/oniguruma'
import {
  createStarryNight as createStarry,
  type Grammar,
} from '@wooorm/starry-night'

import {
  completeSpans,
  labelsFromHighlightedHtml,
  scoreAgreement,
  spansFromShikiTokens,
} from './normalization'
import {
  engineIds,
  languages,
  type EngineId,
  type HighlightSpan,
  type LanguageConfig,
  type WorkerRequest,
  type WorkerResponse,
} from './types'

type WorkerScope = {
  addEventListener(
    type: 'message',
    listener: (event: MessageEvent<WorkerRequest>) => void,
  ): void
  postMessage(message: WorkerResponse): void
}

const worker = globalThis as unknown as WorkerScope
let latest: WorkerRequest | undefined
let processing = false

worker.addEventListener('message', ({ data }) => {
  latest = data
  if (!processing) void drain()
})

async function drain() {
  processing = true
  try {
    while (latest) {
      const request = latest
      latest = undefined
      await compare(request)
    }
  } finally {
    processing = false
  }
}

async function compare(request: WorkerRequest) {
  const language = languages.find((candidate) => candidate.id === request.language)
  if (!language) return
  for (const engine of engineIds) post(request.id, engine, 'running')
  const candidates = engineIds
    .filter((engine) => engine !== 'shiki')
    .map((engine) =>
      runEngine(engine, request.source, language).then(
        (spans) => ({ engine, spans }),
        (error: unknown) => ({ engine, error }),
      ),
    )

  let reference: HighlightSpan[]
  try {
    reference = await runShiki(request.source, language)
    worker.postMessage({
      id: request.id,
      engine: 'shiki',
      phase: 'done',
      spans: reference,
      agreement: scoreAgreement(request.source, reference, reference),
    })
  } catch (error) {
    fail(request.id, 'shiki', error)
    for (const engine of engineIds) {
      if (engine !== 'shiki') fail(request.id, engine, 'Shiki reference failed')
    }
    return
  }

  await Promise.all(candidates.map(async (candidate) => {
    const result = await candidate
    if ('error' in result) {
      fail(request.id, result.engine, result.error)
    } else {
      try {
        const { engine, spans } = result
          worker.postMessage({
            id: request.id,
            engine,
            phase: 'done',
            spans,
            agreement: scoreAgreement(request.source, reference, spans),
          })
      } catch (error) {
        fail(request.id, result.engine, error)
      }
    }
  }))
}

function post(id: number, engine: EngineId, phase: 'running') {
  worker.postMessage({ id, engine, phase })
}

function fail(id: number, engine: EngineId, error: unknown) {
  worker.postMessage({
    id,
    engine,
    phase: 'error',
    message: error instanceof Error ? error.message : String(error),
  })
}

async function runEngine(
  engine: Exclude<EngineId, 'shiki'>,
  source: string,
  language: LanguageConfig,
) {
  if (engine === 'gpu-lexer') {
    const { parse } = await import('gpu-lexer')
    return completeSpans(source.length, await parse(source))
  }
  if (engine === 'highlight') {
    const { default: highlighter } = await import('highlight.js')
    return labelsFromHighlightedHtml(
      source,
      highlighter.highlight(source, {
        language: language.highlight,
        ignoreIllegals: true,
      }).value,
      engine,
      language.id,
    )
  }
  if (engine === 'sugar-high') {
    const { highlight } = await import('sugar-high')
    return labelsFromHighlightedHtml(
      source,
      highlight(source, { lang: language.sugar }),
      engine,
      language.id,
    )
  }
  if (engine === 'starry-night') {
    const [{ toHtml }, highlighter] = await Promise.all([
      import('hast-util-to-html'),
      starryNight(language),
    ])
    return labelsFromHighlightedHtml(
      source,
      toHtml(highlighter.highlight(source, language.starry)),
      engine,
      language.id,
    )
  }
  const prism = await loadPrism()
  const grammar = prism.languages[language.prism]
  if (!grammar) throw new Error(`Prism grammar ${language.prism} is unavailable`)
  return labelsFromHighlightedHtml(
    source,
    prism.highlight(source, grammar, language.prism),
    engine,
    language.id,
  )
}

const oniguruma = createOnigurumaEngine(() =>
  import('shiki/wasm').then((module) => module.default),
)
const createShikiHighlighter = createBundledHighlighter({
  langs: {
    typescript: () => import('shiki/langs/typescript.mjs'),
    javascript: () => import('shiki/langs/javascript.mjs'),
    python: () => import('shiki/langs/python.mjs'),
    css: () => import('shiki/langs/css.mjs'),
    html: () => import('shiki/langs/html.mjs'),
    markdown: () => import('shiki/langs/markdown.mjs'),
    json: () => import('shiki/langs/json.mjs'),
    bash: () => import('shiki/langs/bash.mjs'),
    c: () => import('shiki/langs/c.mjs'),
    cpp: () => import('shiki/langs/cpp.mjs'),
    java: () => import('shiki/langs/java.mjs'),
    go: () => import('shiki/langs/go.mjs'),
    rust: () => import('shiki/langs/rust.mjs'),
  },
  themes: {
    'github-light': () => import('shiki/themes/github-light.mjs'),
  },
  engine: () => oniguruma,
})
const shikiHighlighters = new Map<
  LanguageConfig['shiki'],
  ReturnType<typeof createShikiHighlighter>
>()

function shiki(language: LanguageConfig) {
  let highlighter = shikiHighlighters.get(language.shiki)
  if (!highlighter) {
    highlighter = createShikiHighlighter({
      langs: [language.shiki],
      themes: ['github-light'],
    })
    shikiHighlighters.set(language.shiki, highlighter)
  }
  return highlighter
}

async function runShiki(source: string, language: LanguageConfig) {
  const highlighter = await shiki(language)
  const result = highlighter.codeToTokens(source, {
    lang: language.shiki,
    theme: 'github-light',
    includeExplanation: true,
  })
  return spansFromShikiTokens(source, result.tokens)
}

const starryGrammarLoaders = {
  typescript: () => oneGrammar(import('@wooorm/starry-night/source.ts')),
  javascript: () => oneGrammar(import('@wooorm/starry-night/source.js')),
  python: () => oneGrammar(import('@wooorm/starry-night/source.python')),
  css: () => oneGrammar(import('@wooorm/starry-night/source.css')),
  html: async () => Promise.all([
    import('@wooorm/starry-night/text.html.basic').then((module) => module.default),
    import('@wooorm/starry-night/source.js').then((module) => module.default),
    import('@wooorm/starry-night/source.css').then((module) => module.default),
  ]),
  markdown: async () => Promise.all([
    import('@wooorm/starry-night/text.md').then((module) => module.default),
    import('@wooorm/starry-night/text.html.basic').then((module) => module.default),
    import('@wooorm/starry-night/source.js').then((module) => module.default),
    import('@wooorm/starry-night/source.css').then((module) => module.default),
  ]),
  json: () => oneGrammar(import('@wooorm/starry-night/source.json')),
  shell: () => oneGrammar(import('@wooorm/starry-night/source.shell')),
  c: () => oneGrammar(import('@wooorm/starry-night/source.c')),
  cpp: async () => Promise.all([
    import('@wooorm/starry-night/source.c++').then((module) => module.default),
    import('@wooorm/starry-night/source.c').then((module) => module.default),
  ]),
  java: () => oneGrammar(import('@wooorm/starry-night/source.java')),
  go: () => oneGrammar(import('@wooorm/starry-night/source.go')),
  rust: () => oneGrammar(import('@wooorm/starry-night/source.rust')),
} satisfies Record<LanguageConfig['id'], () => Promise<Grammar[]>>
const starryHighlighters = new Map<
  LanguageConfig['id'],
  ReturnType<typeof createStarry>
>()

function starryNight(language: LanguageConfig) {
  let highlighter = starryHighlighters.get(language.id)
  if (!highlighter) {
    highlighter = starryGrammarLoaders[language.id]().then((grammars) =>
      createStarry(grammars),
    )
    starryHighlighters.set(language.id, highlighter)
  }
  return highlighter
}

async function oneGrammar(module: Promise<{ default: Grammar }>) {
  return [(await module).default]
}

let prismPromise: Promise<typeof PrismType> | undefined
function loadPrism() {
  return (prismPromise ??= createPrism())
}

async function createPrism() {
  const { default: prism } = await import('prismjs')
  ;(globalThis as unknown as { Prism: typeof prism }).Prism = prism
  await import('prismjs/components/prism-typescript.js')
  await import('prismjs/components/prism-python.js')
  await import('prismjs/components/prism-markdown.js')
  await import('prismjs/components/prism-json.js')
  await import('prismjs/components/prism-bash.js')
  await import('prismjs/components/prism-c.js')
  await import('prismjs/components/prism-cpp.js')
  await import('prismjs/components/prism-java.js')
  await import('prismjs/components/prism-go.js')
  await import('prismjs/components/prism-rust.js')
  return prism
}

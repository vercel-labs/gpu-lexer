export const syntaxTypes = [
  'plain',
  'comment',
  'string',
  'number',
  'keyword',
  'type',
  'function',
  'constant',
  'operator',
] as const

export type SyntaxType = (typeof syntaxTypes)[number]

export type HighlightSpan = {
  start: number
  end: number
  type: SyntaxType
}

export const engineIds = [
  'gpu-lexer',
  'shiki',
  'prism',
  'highlight',
  'sugar-high',
  'starry-night',
] as const

export type EngineId = (typeof engineIds)[number]

export const engineNames: Record<EngineId, string> = {
  'gpu-lexer': 'gpu-lexer',
  shiki: 'Shiki',
  prism: 'Prism.js',
  highlight: 'Highlight.js',
  'sugar-high': 'Sugar High',
  'starry-night': 'Starry Night',
}

export const languages = [
  {
    id: 'typescript',
    name: 'TypeScript',
    shiki: 'typescript',
    prism: 'typescript',
    highlight: 'typescript',
    sugar: 'typescript',
    starry: 'source.ts',
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    shiki: 'javascript',
    prism: 'javascript',
    highlight: 'javascript',
    sugar: 'javascript',
    starry: 'source.js',
  },
  {
    id: 'python',
    name: 'Python',
    shiki: 'python',
    prism: 'python',
    highlight: 'python',
    sugar: 'python',
    starry: 'source.python',
  },
  {
    id: 'css',
    name: 'CSS',
    shiki: 'css',
    prism: 'css',
    highlight: 'css',
    sugar: 'css',
    starry: 'source.css',
  },
  {
    id: 'html',
    name: 'HTML',
    shiki: 'html',
    prism: 'markup',
    highlight: 'xml',
    sugar: 'html',
    starry: 'text.html.basic',
  },
  {
    id: 'markdown',
    name: 'Markdown',
    shiki: 'markdown',
    prism: 'markdown',
    highlight: 'markdown',
    sugar: 'markdown',
    starry: 'text.md',
  },
  {
    id: 'json',
    name: 'JSON',
    shiki: 'json',
    prism: 'json',
    highlight: 'json',
    sugar: 'json',
    starry: 'source.json',
  },
  {
    id: 'shell',
    name: 'Shell',
    shiki: 'bash',
    prism: 'bash',
    highlight: 'bash',
    sugar: 'shell',
    starry: 'source.shell',
  },
  {
    id: 'c',
    name: 'C',
    shiki: 'c',
    prism: 'c',
    highlight: 'c',
    sugar: 'c',
    starry: 'source.c',
  },
  {
    id: 'cpp',
    name: 'C++',
    shiki: 'cpp',
    prism: 'cpp',
    highlight: 'cpp',
    sugar: 'cpp',
    starry: 'source.c++',
  },
  {
    id: 'java',
    name: 'Java',
    shiki: 'java',
    prism: 'java',
    highlight: 'java',
    sugar: 'java',
    starry: 'source.java',
  },
  {
    id: 'go',
    name: 'Go',
    shiki: 'go',
    prism: 'go',
    highlight: 'go',
    sugar: 'go',
    starry: 'source.go',
  },
  {
    id: 'rust',
    name: 'Rust',
    shiki: 'rust',
    prism: 'rust',
    highlight: 'rust',
    sugar: 'rust',
    starry: 'source.rust',
  },
] as const

export type LanguageId = (typeof languages)[number]['id']
export type LanguageConfig = (typeof languages)[number]

export type WorkerRequest = {
  id: number
  language: LanguageId
  source: string
}

export type Agreement = {
  correct: number
  total: number
  value: number
}

export type WorkerResponse =
  | { id: number; engine: EngineId; phase: 'running' }
  | {
      id: number
      engine: EngineId
      phase: 'done'
      spans: HighlightSpan[]
      agreement: Agreement
    }
  | { id: number; engine: EngineId; phase: 'error'; message: string }

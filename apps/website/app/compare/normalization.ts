import type {
  Agreement,
  HighlightSpan,
  SyntaxType,
} from './types'

const operator = /^[+*/%=!&|^~?:<>-]+$/
const primitiveType = /^(?:any|bigint|boolean|never|null|number|object|string|symbol|undefined|unknown|void)$/
const classRules: readonly [SyntaxType, RegExp][] = [
  ['comment', /^comment\./],
  [
    'string',
    /^(?:string|constant\.other\.symbol|constant\.regexp|markup\.inline\.raw|markup\.underline\.link)\./,
  ],
  ['number', /^constant\.numeric\./],
  ['operator', /^keyword\.operator(?:\.|$)/],
  [
    'keyword',
    /^(?:keyword(?!\.operator(?:\.|$))|storage|modifier|control|markup\.heading)(?:\.|$)/,
  ],
  [
    'type',
    /^(?:entity\.name\.(?:type|class|interface|struct|enum|tag)|support\.(?:type|class))\./,
  ],
  [
    'function',
    /^(?:entity\.name\.(?:function|method)|support\.function)\./,
  ],
  [
    'constant',
    /^(?:constant\.(?:language|other)|support\.constant)(?:\.|$)/,
  ],
]

type ShikiToken = {
  content: string
  offset: number
  explanation?: readonly {
    content: string
    scopes: readonly { scopeName: string }[]
  }[]
}

export function spansFromShikiTokens(
  source: string,
  lines: readonly (readonly ShikiToken[])[],
) {
  const spans: HighlightSpan[] = []
  let previousEnd = 0
  for (const line of lines) {
    for (const token of line) {
      if (
        !Number.isInteger(token.offset) ||
        token.offset < previousEnd ||
        source.slice(token.offset, token.offset + token.content.length) !==
          token.content
      ) {
        throw new Error(`Shiki output diverged from source at ${token.offset}`)
      }
      const explanation = token.explanation
      if (!explanation || explanation.map((item) => item.content).join('') !== token.content) {
        throw new Error(`Shiki omitted scope information at ${token.offset}`)
      }
      let offset = token.offset
      for (const item of explanation) {
        appendSpan(
          spans,
          offset,
          offset + item.content.length,
          classFromScopes(item.scopes.map((scope) => scope.scopeName)),
        )
        offset += item.content.length
      }
      previousEnd = offset
    }
  }
  return completeSpans(source.length, spans)
}

export function labelsFromHighlightedHtml(
  source: string,
  html: string,
  engine: 'highlight' | 'prism' | 'sugar-high' | 'starry-night',
  family: string,
) {
  const spans: HighlightSpan[] = []
  const stack: string[][] = [[]]
  let offset = 0
  for (const match of html.matchAll(/<[^>]*>|[^<]+/g)) {
    const fragment = match[0]
    if (fragment[0] === '<') {
      if (/^<\/span\b/i.test(fragment)) stack.pop()
      else if (/^<span\b/i.test(fragment)) {
        const classes = /\bclass=(?:"([^"]*)"|'([^']*)')/i.exec(fragment)
        stack.push(
          (classes?.[1] ?? classes?.[2] ?? '')
            .split(/\s+/)
            .filter(Boolean),
        )
      } else if (/^<br\s*\/?\s*>$/i.test(fragment)) append('\n')
      continue
    }
    append(decodeEntities(fragment))
  }
  if (offset !== source.length) {
    throw new Error(
      `${engine} reconstructed ${offset}/${source.length} source units`,
    )
  }
  return completeSpans(source.length, spans)

  function append(value: string) {
    if (!value) return
    if (source.slice(offset, offset + value.length) !== value) {
      throw new Error(`${engine} output diverged from source at ${offset}`)
    }
    appendSpan(
      spans,
      offset,
      offset + value.length,
      classifyHtmlFrame(engine, stack, value, family),
    )
    offset += value.length
  }
}

export function completeSpans(
  sourceLength: number,
  input: readonly HighlightSpan[],
) {
  const result: HighlightSpan[] = []
  let cursor = 0
  for (const span of input) {
    const start = Math.max(cursor, Math.min(sourceLength, span.start))
    const end = Math.max(start, Math.min(sourceLength, span.end))
    if (start > cursor) appendSpan(result, cursor, start, 'plain')
    appendSpan(result, start, end, span.type)
    cursor = end
  }
  if (cursor < sourceLength) appendSpan(result, cursor, sourceLength, 'plain')
  return result
}

export function scoreAgreement(
  source: string,
  reference: readonly HighlightSpan[],
  candidate: readonly HighlightSpan[],
): Agreement {
  let correct = 0
  let total = 0
  let offset = 0
  let referenceIndex = 0
  let candidateIndex = 0
  while (offset < source.length) {
    const from = offset
    const first = source.charCodeAt(offset)
    let kind: 'word' | 'space' | 'newline' | 'symbol'
    if (first === 10 || first === 13) kind = 'newline'
    else if (isHorizontalSpace(first)) kind = 'space'
    else if (isWord(first)) kind = 'word'
    else kind = 'symbol'

    if (kind === 'word') {
      do offset += 1
      while (offset < source.length && isWord(source.charCodeAt(offset)))
    } else if (kind === 'space') {
      do offset += 1
      while (
        offset < source.length &&
        isHorizontalSpace(source.charCodeAt(offset))
      )
    } else {
      offset += 1
      if (kind === 'newline' && first === 13 && source.charCodeAt(offset) === 10) {
        offset += 1
      }
    }
    if (kind === 'space' || kind === 'newline') continue
    const expected = dominantClass(reference, from, offset, referenceIndex)
    referenceIndex = expected.index
    const actual = dominantClass(candidate, from, offset, candidateIndex)
    candidateIndex = actual.index
    total += 1
    if (expected.type === actual.type) correct += 1
  }
  return { correct, total, value: total ? correct / total : 1 }
}

function classFromScopes(scopes: readonly string[]): SyntaxType {
  let normalized = normalizeScopes(scopes)
  if (
    normalized.some((scope) =>
      /^punctuation\.definition\.template-expression(?:\.|$)/.test(scope),
    )
  ) {
    return 'operator'
  }
  const embedded = normalized.some((scope) =>
    /^meta\.(?:embedded|template\.expression)(?:\.|$)/.test(scope),
  )
  if (embedded) {
    normalized = normalized.filter(
      (scope) => !/^string\.template(?:\.|$)/.test(scope),
    )
  }
  for (const [name, pattern] of classRules) {
    if (normalized.some((scope) => pattern.test(scope))) return name
  }
  return 'plain'
}

function classifyHtmlFrame(
  engine: 'highlight' | 'prism' | 'sugar-high' | 'starry-night',
  stack: readonly string[][],
  value: string,
  family: string,
): SyntaxType {
  if (engine === 'prism') {
    const all = new Set(stack.flat())
    if (all.has('interpolation-punctuation')) return 'operator'
    if (all.has('code-snippet')) return 'string'
    if (all.has('title') && all.has('important')) return 'keyword'
  }
  for (let index = stack.length - 1; index > 0; index -= 1) {
    const name = classifyHtmlClasses(engine, stack[index], value, family)
    if (name) return name
  }
  return 'plain'
}

function classifyHtmlClasses(
  engine: 'highlight' | 'prism' | 'sugar-high' | 'starry-night',
  classes: readonly string[],
  value: string,
  family: string,
): SyntaxType | undefined {
  if (engine === 'starry-night') {
    const names = new Set(classes)
    if (names.has('pl-c')) return 'comment'
    if (
      ['pl-s', 'pl-pds', 'pl-sr', 'pl-cce', 'pl-sre', 'pl-sra', 'pl-corl'].some(
        (name) => names.has(name),
      )
    ) return 'string'
    if (names.has('pl-c1')) {
      if (/^[-+]?(?:\d|\.\d)/.test(value)) return 'number'
      if (family === 'typescript' && primitiveType.test(value)) return 'type'
      return 'constant'
    }
    if (names.has('pl-kos') || names.has('pl-pse')) return 'operator'
    if (names.has('pl-k')) return operator.test(value) ? 'operator' : 'keyword'
    if (names.has('pl-mh') || names.has('pl-ms')) return 'keyword'
    if (names.has('pl-ent')) return 'type'
    if (names.has('pl-e') || names.has('pl-en')) {
      return family === 'typescript' && /^[A-Z]/.test(value)
        ? 'type'
        : 'function'
    }
    return 'plain'
  }

  if (engine === 'sugar-high') {
    const type = classes
      .find((name) => name.startsWith('sh__token--'))
      ?.slice(11)
    if (!type) return undefined
    if (type === 'comment') return 'comment'
    if (type === 'string') return 'string'
    if (type === 'keyword') return 'keyword'
    if (type === 'class') {
      if (/^\d/.test(value)) return 'number'
      if (/^(?:false|infinity|nan|null|true|undefined)$/i.test(value)) {
        return 'constant'
      }
      return 'type'
    }
    if (type === 'entity') return 'type'
    if (type === 'property') return family === 'css' ? 'type' : 'plain'
    if (type === 'sign') return operator.test(value) ? 'operator' : 'plain'
    return 'plain'
  }

  const names = new Set(
    classes.map((name) =>
      engine === 'highlight' ? name.replace(/^hljs-/, '') : name,
    ),
  )
  if (engine === 'highlight') {
    if (names.has('comment')) return 'comment'
    if (
      ['string', 'regexp', 'char', 'link', 'code', 'meta-string'].some((name) =>
        names.has(name),
      )
    ) return 'string'
    if (names.has('number')) return 'number'
    if (names.has('title') && names.has('class_')) return 'type'
    if (names.has('title') && names.has('function_')) return 'function'
    if (names.has('type') || names.has('name') || names.has('selector-tag')) {
      return 'type'
    }
    if (family === 'css' && (names.has('attribute') || names.has('property'))) {
      return 'type'
    }
    if (names.has('built_in') && family === 'typescript' && primitiveType.test(value)) {
      return 'type'
    }
    if (names.has('title') || names.has('function') || names.has('built_in')) {
      return 'function'
    }
    if (names.has('symbol')) return family === 'html' ? 'plain' : 'string'
    if (names.has('literal')) return 'constant'
    if (names.has('operator')) return 'operator'
    if (names.has('keyword') || names.has('meta-keyword') || names.has('section')) {
      return 'keyword'
    }
    if (
      [
        'subst',
        'template-variable',
        'params',
        'attr',
        'attribute',
        'variable',
        'property',
        'punctuation',
        'tag',
        'quote',
      ].some((name) => names.has(name))
    ) return 'plain'
    return undefined
  }

  if (names.has('comment')) return 'comment'
  if (
    ['string', 'char', 'regex', 'attr-value', 'template-string', 'cdata', 'symbol', 'url'].some(
      (name) => names.has(name),
    )
  ) return 'string'
  if (names.has('number')) return 'number'
  if (names.has('class-name') || names.has('builtin') || names.has('doctype-tag')) {
    return 'type'
  }
  if (family === 'css' && names.has('property')) return 'type'
  if (family === 'css' && names.has('selector') && /^[-a-z][\w-]*$/i.test(value)) {
    return 'type'
  }
  if (names.has('function') || names.has('function-variable')) return 'function'
  if (names.has('boolean') || names.has('constant')) return 'constant'
  if (names.has('operator') || names.has('interpolation-punctuation')) {
    return 'operator'
  }
  if (
    ['keyword', 'control-flow', 'directive', 'macro', 'important', 'atrule', 'rule'].some(
      (name) => names.has(name),
    )
  ) return 'keyword'
  if (names.has('tag')) return 'type'
  if (
    [
      'interpolation',
      'punctuation',
      'attr-name',
      'parameter',
      'property',
      'variable',
      'namespace',
      'plain-text',
      'prolog',
      'name',
      'blockquote',
      'entity',
    ].some((name) => names.has(name))
  ) return 'plain'
  return undefined
}

function dominantClass(
  spans: readonly HighlightSpan[],
  from: number,
  to: number,
  startIndex: number,
) {
  let index = startIndex
  while (index < spans.length && spans[index].end <= from) index += 1
  const widths = new Map<SyntaxType, number>()
  for (let cursor = index; cursor < spans.length && spans[cursor].start < to; cursor += 1) {
    const span = spans[cursor]
    const width = Math.max(0, Math.min(to, span.end) - Math.max(from, span.start))
    if (width) widths.set(span.type, (widths.get(span.type) ?? 0) + width)
  }
  let type: SyntaxType = 'plain'
  let width = -1
  for (const [candidate, candidateWidth] of widths) {
    if (candidateWidth > width) {
      type = candidate
      width = candidateWidth
    }
  }
  return { type, index }
}

function appendSpan(
  spans: HighlightSpan[],
  start: number,
  end: number,
  type: SyntaxType,
) {
  if (end <= start) return
  const previous = spans.at(-1)
  if (previous?.type === type && previous.end === start) previous.end = end
  else spans.push({ start, end, type })
}

function normalizeScopes(scopes: readonly string[]) {
  return [...new Set(scopes.map((scope) => scope.trim().toLowerCase()).filter(Boolean))]
}

function decodeEntities(value: string) {
  return value.replace(
    /&(?:#(\d+)|#x([\da-f]+)|amp|lt|gt|quot|apos|#039);/gi,
    (entity, decimal: string | undefined, hex: string | undefined) => {
      if (decimal) return String.fromCodePoint(Number(decimal))
      if (hex) return String.fromCodePoint(Number.parseInt(hex, 16))
      return {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&#039;': "'",
      }[entity.toLowerCase()] ?? entity
    },
  )
}

function isHorizontalSpace(code: number) {
  return code === 9 || code === 11 || code === 12 || code === 32
}

function isWord(code: number) {
  return (
    code > 127 ||
    code === 95 ||
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122)
  )
}

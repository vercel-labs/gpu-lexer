'use client'

import { useEffect, useState } from 'react'

const source = `import { parse } from 'gpu-lexer'

const spans = await parse('source code')
// {
//   type: 'plain' | 'comment' | 'string' | 'number' | 'keyword' | 'type' | 'function' | 'constant' | 'operator'
//   start: number
//   end: number
// }[]`

type HighlightSpan = {
  type: string
  start: number
  end: number
}

export function ApiExample() {
  const [spans, setSpans] = useState<HighlightSpan[] | null>(null)

  useEffect(() => {
    let active = true
    void import('gpu-lexer')
      .then(({ parse }) => parse(source))
      .then((result) => {
        if (active) setSpans(result)
      })
      .catch(() => {
        if (active) setSpans([])
      })
    return () => {
      active = false
    }
  }, [])

  return (
    <pre
      className='api-example'
      aria-label='gpu-lexer API example'
      aria-busy={spans === null}
    >
      <code>{spans === null ? source : renderHighlightedSource(spans)}</code>
    </pre>
  )
}

function renderHighlightedSource(spans: HighlightSpan[]) {
  const output = []
  let cursor = 0
  for (const span of spans) {
    if (span.start > cursor) output.push(source.slice(cursor, span.start))
    output.push(
      <span
        className={`syntax-${span.type}`}
        key={`${span.start}-${span.end}`}
      >
        {source.slice(span.start, span.end)}
      </span>,
    )
    cursor = span.end
  }
  if (cursor < source.length) output.push(source.slice(cursor))
  return output
}

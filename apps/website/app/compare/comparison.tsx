'use client'

import { memo, useEffect, useRef, useState } from 'react'

import {
  engineIds,
  engineNames,
  languages,
  type Agreement,
  type EngineId,
  type HighlightSpan,
  type LanguageId,
  type WorkerResponse,
} from './types'

const maxSourceLength = 20_000
const initialSource = `export type RawCreateParams =
  | {
      errorMap?: ZodErrorMap | undefined;
      invalid_type_error?: string | undefined;
      required_error?: string | undefined;
      message?: string | undefined;
      description?: string | undefined;
    }
  | undefined;
export type ProcessedCreateParams = {
  errorMap?: ZodErrorMap | undefined;
  description?: string | undefined;
};`

type EngineResult = {
  phase: 'queued' | 'running' | 'done' | 'error'
  source?: string
  spans?: HighlightSpan[]
  agreement?: Agreement
  message?: string
}

const initialResults = Object.fromEntries(
  engineIds.map((engine) => [engine, { phase: 'queued' }]),
) as Record<EngineId, EngineResult>

export function Comparison() {
  const [source, setSource] = useState(initialSource)
  const [language, setLanguage] = useState<LanguageId>('typescript')
  const [results, setResults] = useState(initialResults)
  const workerRef = useRef<Worker | null>(null)
  const requestId = useRef(0)
  const submitted = useRef({ id: 0, source: initialSource })

  useEffect(() => {
    const worker = new Worker(new URL('./comparison-worker.ts', import.meta.url), {
      type: 'module',
    })
    workerRef.current = worker
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.id !== requestId.current) return
      setResults((current) => ({
        ...current,
        [data.engine]:
          data.phase === 'running'
            ? { ...current[data.engine], phase: 'running' }
            : data.phase === 'error'
              ? {
                  ...current[data.engine],
                  phase: 'error',
                  message: data.message,
                }
              : {
                  phase: 'done',
                  source: submitted.current.source,
                  spans: data.spans,
                  agreement: data.agreement,
                },
      }))
    }
    worker.onerror = ({ message }) => {
      setResults(
        Object.fromEntries(
          engineIds.map((engine) => [
            engine,
            { phase: 'error', message: message || 'Comparison worker failed' },
          ]),
        ) as Record<EngineId, EngineResult>,
      )
    }
    return () => {
      workerRef.current = null
      worker.terminate()
    }
  }, [])

  useEffect(() => {
    const id = ++requestId.current
    setResults((current) =>
      Object.fromEntries(
        engineIds.map((engine) => [
          engine,
          { ...current[engine], phase: 'queued', message: undefined },
        ]),
      ) as Record<EngineId, EngineResult>,
    )
    const timer = setTimeout(() => {
      submitted.current = { id, source }
      workerRef.current?.postMessage({ id, language, source })
    }, 240)
    return () => clearTimeout(timer)
  }, [language, source])

  return (
    <>
      <section className='compare-input' aria-labelledby='compare-source-label'>
        <div className='compare-input-head'>
          <label id='compare-source-label' htmlFor='compare-source'>
            source
          </label>
          <div className='compare-controls'>
            <label htmlFor='compare-language'>grammar</label>
            <select
              id='compare-language'
              onChange={(event) =>
                setLanguage(event.target.value as LanguageId)
              }
              value={language}
            >
              {languages.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
            <span>
              {source.length.toLocaleString('en-US')} /{' '}
              {maxSourceLength.toLocaleString('en-US')}
            </span>
          </div>
        </div>
        <textarea
          aria-labelledby='compare-source-label'
          id='compare-source'
          maxLength={maxSourceLength}
          onChange={(event) => setSource(event.target.value)}
          spellCheck={false}
          value={source}
        />
      </section>

      <section className='compare-results' aria-label='Highlighter results'>
        {engineIds.map((engine) => (
          <EngineCard
            engine={engine}
            fallbackSource={source}
            key={engine}
            result={results[engine]}
          />
        ))}
      </section>
    </>
  )
}

const EngineCard = memo(function EngineCard({
  engine,
  fallbackSource,
  result,
}: {
  engine: EngineId
  fallbackSource: string
  result: EngineResult
}) {
  const renderedSource = result.source ?? fallbackSource
  const detail = result.phase === 'done' && result.agreement
    ? `${formatPercent(result.agreement.value)} · ${result.agreement.correct.toLocaleString('en-US')}/${result.agreement.total.toLocaleString('en-US')} parts`
    : result.phase === 'error'
      ? 'failed'
      : result.phase === 'queued'
        ? 'waiting for pause…'
        : 'classifying…'
  return (
    <article
      aria-busy={result.phase === 'queued' || result.phase === 'running'}
      className={`compare-result${engine === 'gpu-lexer' ? ' compare-result-featured' : ''}`}
    >
      <header>
        <h2>{engineNames[engine]}</h2>
        <span>{detail}</span>
      </header>
      <pre className='compare-output'>
        <code>
          {result.spans
            ? renderHighlightedSource(renderedSource, result.spans)
            : renderedSource}
        </code>
      </pre>
      {result.phase === 'error' ? (
        <p className='compare-error'>{result.message}</p>
      ) : null}
    </article>
  )
})

function renderHighlightedSource(
  source: string,
  spans: readonly HighlightSpan[],
) {
  return spans.map((span, index) => (
    <span
      className={`syntax-${span.type}`}
      key={`${span.start}-${span.end}-${index}`}
    >
      {source.slice(span.start, span.end)}
    </span>
  ))
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

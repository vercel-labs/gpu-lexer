import { correctnessComparison } from './correctness.generated'
import { modelStats } from './model-stats.generated'

type BarDatum = {
  label: string
  value: number
  featured?: boolean
}

const performance: readonly BarDatum[] = [
  { label: 'gpu-lexer', value: 402, featured: true },
  { label: 'Sugar High', value: 836.2 },
  { label: 'Prism.js', value: 1_155.3 },
  { label: 'Highlight.js', value: 1_291.4 },
  { label: 'Starry Night', value: 10_999.3 },
  { label: 'Shiki', value: 29_553.8 },
]

const bundleSizes: readonly BarDatum[] = [
  { label: 'Sugar High (major 6 web languages)', value: 5_515 },
  { label: 'Prism.js (major 6 web languages)', value: 8_848 },
  { label: 'Sugar High (all 29 languages)', value: 9_032 },
  { label: 'Highlight.js (major 6 web languages)', value: 15_261 },
  { label: 'gpu-lexer (one model)', value: 28_115, featured: true },
  { label: 'Prism.js (all 297 languages)', value: 165_992 },
  { label: 'Starry Night (major 6 web languages)', value: 189_775 },
  { label: 'Shiki (major 6 web languages)', value: 219_067 },
  { label: 'Highlight.js (all 193 languages)', value: 246_144 },
  { label: 'Shiki (all 242 grammars)', value: 1_015_269 },
  { label: 'Starry Night (all 710 grammars)', value: 1_526_733 },
]

if (correctnessComparison.runId !== modelStats.runId) {
  throw new Error('run pnpm benchmark:correctness after promoting a model')
}

export function ComparisonCharts() {
  return (
    <section
      className='comparisons'
      aria-label='Performance, size, and agreement comparisons'
    >
      <BarChart
        format={formatTime}
        max={32_000}
        note={
          <>
            One browser run after one warm-up on September 9, 2026. The input
            was 10 concatenated copies of{' '}
            <a
              href='https://unpkg.com/three@0.97.0/build/three.min.js'
              target='_blank'
              rel='noopener noreferrer'
            >
              three.min.js
            </a>{' '}
            (5.56M characters). MacBook Pro, Apple M4 Pro, 20-core GPU, 24GB,
            macOS 26.6.2, Chrome 152. Each engine ran in a dedicated worker; DOM
            rendering was excluded. gpu-lexer and Shiki returned token data,
            Starry Night returned a HAST tree, while Sugar High, Prism.js, and
            Highlight.js returned highlighted HTML. Sugar High 2.3.1, Prism.js
            1.30.0, Highlight.js 11.12.0, Starry Night 3.11.0, and Shiki 4.4.3.
          </>
        }
        rows={performance}
        subtitle='warmed browser time · lower is better'
        ticks={[0, 8_000, 16_000, 24_000, 32_000]}
        title='Highlight 10× three.min.js'
      />
      <BarChart
        format={formatSize}
        max={1_600_000}
        note={
          <>
            Minified and Brotli-compressed browser bundles measured on September
            9, 2026. Major web includes javascript, typescript, css, html, json,
            and markdown. gpu-lexer uses the same bundle for every language.
            Starry Night totals include its Oniguruma WASM payload.
          </>
        }
        rows={bundleSizes}
        subtitle='runtime + selected language coverage · lower is better'
        ticks={[0, 400_000, 800_000, 1_200_000, 1_600_000]}
        title='Loaded library size'
      />
      <BarChart
        format={formatPercent}
        max={100}
        note={
          <>
            Shiki is the 100% normalization reference. Each library&apos;s token
            names are mapped to the same nine classes: plain, comment, string,
            number, keyword, type, function, constant, and operator. Scores
            compare non-whitespace source parts across{' '}
            {correctnessComparison.files.toLocaleString('en-US')} held-out files
            in the{' '}
            <a
              href={correctnessComparison.source}
              target='_blank'
              rel='noopener noreferrer'
            >
              GitHub Innovation Graph
            </a>{' '}
            top 25 for {correctnessComparison.period}, weighted by each
            language&apos;s pusher count. Unsupported languages score zero;
            corpus size does not affect the weights.
          </>
        }
        rows={correctnessComparison.rows}
        subtitle='popularity-weighted agreement with Shiki · higher is better'
        ticks={[0, 25, 50, 75, 100]}
        title='Top-25 weighted agreement'
      />
    </section>
  )
}

function BarChart({
  format,
  max,
  note,
  rows,
  subtitle,
  ticks,
  title,
}: {
  format: (value: number) => string
  max: number
  note: React.ReactNode
  rows: readonly BarDatum[]
  subtitle: string
  ticks: readonly number[]
  title: string
}) {
  return (
    <figure className='bar-chart'>
      <figcaption>
        <b>{title}</b>
        <span>{subtitle}</span>
      </figcaption>
      <div
        aria-label={`${title} chart; scroll horizontally to see the full scale`}
        className='bar-chart-scroll'
        role='region'
        tabIndex={0}
      >
        <div className='bar-chart-body'>
          <div className='bar-axis' aria-hidden='true'>
            <span />
            <div className='bar-axis-ticks'>
              {ticks.map((tick) => (
                <span key={tick} style={{ left: `${(tick / max) * 100}%` }}>
                  {format(tick)}
                </span>
              ))}
            </div>
            <span />
          </div>
          <div className='bar-rows'>
            {rows.map((row) => (
              <div
                aria-label={`${row.label}: ${format(row.value)}`}
                className={`bar-row${row.featured ? ' bar-row-featured' : ''}`}
                key={row.label}
              >
                <span className='bar-name'>{row.label}</span>
                <span className='bar-meter' aria-hidden='true'>
                  <span
                    className='bar-fill'
                    style={{
                      width: `${Math.max((row.value / max) * 100, 0.25)}%`,
                    }}
                  />
                </span>
                <span className='bar-value'>{format(row.value)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className='bar-chart-note'>{note}</p>
    </figure>
  )
}

function formatTime(milliseconds: number) {
  if (milliseconds === 0) return '0s'
  if (milliseconds >= 1_000) {
    const seconds = milliseconds / 1_000
    return `${seconds >= 10 ? seconds.toFixed(1) : seconds.toFixed(2)}s`
  }
  return `${milliseconds.toFixed(1)}ms`
}

function formatSize(bytes: number) {
  if (bytes === 0) return '0KB'
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)}MB`
  return `${(bytes / 1024).toFixed(1)}KB`
}

function formatPercent(value: number) {
  return `${value.toFixed(value === 0 || value === 100 ? 0 : 2)}%`
}

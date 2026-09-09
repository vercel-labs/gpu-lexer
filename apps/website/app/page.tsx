import { ApiExample } from './api-example'
import { ComparisonCharts } from './comparison-charts'
import { Demo } from './demo'
import { languageProbes } from './language-probes.generated'
import { modelStats } from './model-stats.generated'

const matchingPercent = percent(modelStats.accuracy)
const disagreementPercent = percent(modelStats.shikiDisagreementRate)
const activeLanguageProbes =
  String(languageProbes.runId) === modelStats.runId
    ? languageProbes.results
    : []
const heldOutLanguages = new Set(
  modelStats.languageAccuracyDistribution.flatMap((band) => band.languages),
)
const languageAccuracyBands = modelStats.languageAccuracyDistribution.map(
  (band) => ({
    ...band,
    languages: languagesInBand(band.range, band.languages, heldOutLanguages),
  }),
)
const verifiedLanguageCount = new Set(
  languageAccuracyBands.flatMap((band) => band.languages),
).size

export default function Home() {
  return (
    <main>
      <section className='intro' id='top' aria-labelledby='title'>
        <h1 id='title'>
          27.5KB language-agnostic
          <sup className='footnote-ref'>
            <a
              href='#language-accuracy-note'
              aria-label='See language accuracy note'
            >
              1
            </a>
          </sup>{' '}
          WebGPU syntax highlighter
        </h1>
        <p className='byline'>
          <a href='https://x.com/shuding' target='_blank'>
            Shu Ding
          </a>{' '}
          at{' '}
          <a href='https://github.com/vercel-labs' target='_blank'>
            Vercel Labs
          </a>
        </p>
        <ApiExample />
        <p className='lede'>
          gpu-lexer splits source code into simple parts—words, whitespace,
          newlines, and symbols. Then a tiny WebGPU model combines local and
          whole-file context to label each part. It is designed for{' '}
          <span className='accent-word'>any language</span>: instead of choosing
          a grammar, it guesses each part&apos;s type from the surrounding
          source, even when it never saw that language or syntax during
          training. Adjacent labels become the syntax spans returned to your
          code.
        </p>
        <p className='lede'>
          This is an <span className='accent-word'>experiment</span>, not a
          grammar-equivalent highlighter. On files kept out of training,{' '}
          <strong>
            {disagreementPercent} of the current model&apos;s token labels
            differ from Shiki
          </strong>
          . This measures agreement with Shiki—not objective correctness—and
          unseen languages or real-world code may differ more often.
        </p>
        <aside className='qualification' id='language-accuracy-note'>
          <p>
            <sup>1</sup> "Language-agnostic" means one shared tokenizer and
            classifier, not equal accuracy for every language. {matchingPercent}{' '}
            is the share of held-out token labels that matched Shiki.
            Mixed-language code is supported too, including embedded{' '}
            <code>&lt;script&gt;</code> and <code>&lt;style&gt;</code> regions
            in HTML, Vue, and Svelte.
          </p>
          <table className='accuracy-distribution'>
            <caption>Held-out label agreement with Shiki by language</caption>
            <thead>
              <tr>
                <th scope='col'>agreement</th>
                <th scope='col'>
                  verified languages ({verifiedLanguageCount})
                </th>
              </tr>
            </thead>
            <tbody>
              {languageAccuracyBands.map((band) => (
                <tr key={band.range}>
                  <td>{band.range}</td>
                  <td className='language-list'>{band.languages.join(', ')}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope='row'>training epoch</th>
                <td>
                  {modelStats.trainingTokensPerEpoch.toLocaleString('en-US')}{' '}
                  tokens
                </td>
              </tr>
              <tr>
                <th scope='row'>model</th>
                <td>
                  {modelStats.modelParameters.toLocaleString('en-US')}{' '}
                  parameters
                </td>
              </tr>
            </tfoot>
          </table>
          <p className='table-note'>
            Each language appears in one band; results with fewer held-out
            labels are less stable. Training tokens include context-only tokens
            and replay.
          </p>
        </aside>
      </section>

      <ComparisonCharts />

      <Demo />

      <footer className='site-footer'>
        <p>
          Experimental software. Highlighting is probabilistic, may differ from
          Shiki, and is not a parser or a substitute for compiler, linter, or
          security analysis.
        </p>
        <p className='footer-author'>
          <a href='https://x.com/shuding' target='_blank'>
            Shu Ding
          </a>{' '}
          at{' '}
          <a href='https://github.com/vercel-labs' target='_blank'>
            Vercel Labs
          </a>
          .
        </p>
      </footer>
    </main>
  )
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function languagesInBand(
  range: string,
  verifiedLanguages: readonly string[],
  heldOutLanguages: ReadonlySet<string>,
) {
  return [
    ...verifiedLanguages,
    ...activeLanguageProbes
      .filter(
        (probe) => probe.range === range && !heldOutLanguages.has(probe.id),
      )
      .map((probe) => probe.id),
  ].sort((left, right) => left.localeCompare(right))
}

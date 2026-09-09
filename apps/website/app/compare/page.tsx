import type { Metadata } from 'next'

import { Comparison } from './comparison'

export const metadata: Metadata = {
  title: 'Compare syntax highlighters · gpu-lexer',
  description:
    'Compare gpu-lexer, Shiki, Prism.js, Highlight.js, Sugar High, and Starry Night on the same editable source.',
}

export default function ComparePage() {
  return (
    <main className='compare-page'>
      <header className='compare-intro'>
        <a className='compare-back' href='/'>
          ← gpu-lexer
        </a>
        <h1>same source, six highlighters.</h1>
        <p>
          Edit the source and choose the grammar used by the five conventional
          highlighters. gpu-lexer receives no language hint. Every result is
          mapped into the same nine visual classes before comparison.
        </p>
      </header>

      <Comparison />

      <footer className='compare-note'>
        Agreement excludes whitespace and counts the same simple source parts
        gpu-lexer classifies. Shiki is the 100% reference. The score measures
        normalized agreement with Shiki—not objective correctness. Work runs in
        a dedicated browser worker; the input is capped at 20,000 UTF-16 units.
      </footer>
    </main>
  )
}

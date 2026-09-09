import type { Metadata, Viewport } from 'next'
import { Geist_Mono } from 'next/font/google'
import type { ReactNode } from 'react'

import './styles.css'

export const metadata: Metadata = {
  title: '27.5KB language-agnostic WebGPU syntax highlighter',
  description:
    'An experimental, language-agnostic syntax highlighter powered by WebGPU.',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  colorScheme: 'light',
  themeColor: '#f5f4ee',
}

const geistMono = Geist_Mono({
  subsets: ['latin'],
  variable: '--font-geist-mono',
})

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang='en'>
      <body className={geistMono.variable}>{children}</body>
    </html>
  )
}

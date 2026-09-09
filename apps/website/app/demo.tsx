'use client'

import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'

import { languageProbes } from './language-probes.generated'

type DemoFile = { id: string; name: string; language: string; url: string }
type DemoFolder = { name: string; files: readonly DemoFile[] }
type HighlightSpan = {
  start: number
  end: number
  type:
    | 'plain'
    | 'comment'
    | 'string'
    | 'number'
    | 'keyword'
    | 'type'
    | 'function'
    | 'constant'
    | 'operator'
}

const styledSyntaxTypes = [
  'comment',
  'string',
  'number',
  'keyword',
  'type',
  'function',
  'constant',
  'operator',
] as const
type StyledSyntaxType = (typeof styledSyntaxTypes)[number]
const highlightName = (type: StyledSyntaxType) => `gpu-lexer-demo-${type}`

const demoFolders: readonly DemoFolder[] = [
  {
    name: 'javascript',
    files: [
      {
        id: 'react',
        name: 'react.development.js',
        language: 'javascript',
        url: 'https://unpkg.com/react@19.2.8/cjs/react.development.js',
      },
      {
        id: 'lodash',
        name: 'lodash.js',
        language: 'javascript',
        url: 'https://unpkg.com/lodash@4.17.21/lodash.js',
      },
      {
        id: 'three',
        name: 'three.min.js',
        language: 'javascript',
        url: 'https://unpkg.com/three@0.97.0/build/three.min.js',
      },
    ],
  },
  {
    name: 'react',
    files: [
      {
        id: 'react-app',
        name: 'App.jsx',
        language: 'jsx',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-react/src/App.jsx',
      },
      {
        id: 'react-main',
        name: 'main.jsx',
        language: 'jsx',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-react/src/main.jsx',
      },
      {
        id: 'react-root',
        name: 'root.tsx',
        language: 'tsx',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/playground/environment-react-ssr/src/root.tsx',
      },
    ],
  },
  {
    name: 'vue',
    files: [
      {
        id: 'vue-layout',
        name: 'Layout.vue',
        language: 'vue',
        url: 'https://raw.githubusercontent.com/vuejs/vitepress/09f9672ee108578808f793bb57e306bc92dff7bc/template/.vitepress/theme/Layout.vue',
      },
      {
        id: 'vue-modal',
        name: 'ModalDemo.vue',
        language: 'vue',
        url: 'https://raw.githubusercontent.com/vuejs/vitepress/09f9672ee108578808f793bb57e306bc92dff7bc/docs/components/ModalDemo.vue',
      },
      {
        id: 'vue-hero',
        name: 'VPHero.vue',
        language: 'vue',
        url: 'https://raw.githubusercontent.com/vuejs/vitepress/09f9672ee108578808f793bb57e306bc92dff7bc/src/client/theme-default/components/VPHero.vue',
      },
    ],
  },
  {
    name: 'svelte',
    files: [
      {
        id: 'svelte-app',
        name: 'App.svelte',
        language: 'svelte',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-svelte/src/App.svelte',
      },
      {
        id: 'svelte-counter',
        name: 'Counter.svelte',
        language: 'svelte',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-svelte/src/lib/Counter.svelte',
      },
    ],
  },
  {
    name: 'typescript',
    files: [
      {
        id: 'portless-certs',
        name: 'certs.ts',
        language: 'typescript',
        url: 'https://raw.githubusercontent.com/vercel-labs/portless/main/packages/portless/src/certs.ts',
      },
      {
        id: 'zod-v3-types',
        name: 'types.ts',
        language: 'typescript',
        url: 'https://raw.githubusercontent.com/colinhacks/zod/main/packages/zod/src/v3/types.ts',
      },
    ],
  },
  {
    name: 'css',
    files: [
      {
        id: 'bootstrap',
        name: 'bootstrap.css',
        language: 'css',
        url: 'https://unpkg.com/bootstrap@5.3.8/dist/css/bootstrap.css',
      },
      {
        id: 'tailwind-experimental',
        name: 'tailwind-experimental.css',
        language: 'css',
        url: 'https://unpkg.com/tailwindcss@2.0.1/dist/tailwind-experimental.css',
      },
      {
        id: 'normalize',
        name: 'normalize.css',
        language: 'css',
        url: 'https://unpkg.com/normalize.css@8.0.1/normalize.css',
      },
    ],
  },
  {
    name: 'scss',
    files: [
      {
        id: 'bootstrap-variables',
        name: 'bootstrap._variables.scss',
        language: 'scss',
        url: 'https://raw.githubusercontent.com/twbs/bootstrap/12cb8b902d175a0d612f1280f58b386a09921a16/scss/_variables.scss',
      },
    ],
  },
  {
    name: 'html',
    files: [
      {
        id: 'html5',
        name: 'html5-boilerplate.html',
        language: 'html',
        url: 'https://unpkg.com/html5-boilerplate@9.0.1/dist/index.html',
      },
      {
        id: 'react-fixture',
        name: 'react-fixture.html',
        language: 'html',
        url: 'https://raw.githubusercontent.com/facebook/react/v19.2.0/fixtures/dom/public/index.html',
      },
    ],
  },
  {
    name: 'xslt',
    files: [
      {
        id: 'linguist-xslt',
        name: 'test.xslt',
        language: 'xsl',
        url: 'https://raw.githubusercontent.com/github-linguist/linguist/d5214e1612c858ba14bf98edeca57e1683276f1d/samples/XSLT/test.xslt',
      },
    ],
  },
  {
    name: 'python',
    files: [
      {
        id: 'requests-sessions',
        name: 'sessions.py',
        language: 'python',
        url: 'https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py',
      },
      {
        id: 'pydantic-main',
        name: 'pydantic.main.py',
        language: 'python',
        url: 'https://raw.githubusercontent.com/pydantic/pydantic/v2.11.7/pydantic/main.py',
      },
    ],
  },
  {
    name: 'go',
    files: [
      {
        id: 'gin',
        name: 'gin.go',
        language: 'go',
        url: 'https://raw.githubusercontent.com/gin-gonic/gin/v1.11.0/gin.go',
      },
      {
        id: 'prometheus-counter',
        name: 'counter.go',
        language: 'go',
        url: 'https://raw.githubusercontent.com/prometheus/client_golang/v1.19.1/prometheus/counter.go',
      },
    ],
  },
  {
    name: 'rust',
    files: [
      {
        id: 'tokio-worker',
        name: 'tokio.worker.rs',
        language: 'rust',
        url: 'https://raw.githubusercontent.com/tokio-rs/tokio/tokio-1.47.1/tokio/src/runtime/scheduler/multi_thread/worker.rs',
      },
      {
        id: 'axum-hello-world',
        name: 'axum.main.rs',
        language: 'rust',
        url: 'https://raw.githubusercontent.com/tokio-rs/axum/axum-v0.8.4/examples/hello-world/src/main.rs',
      },
      {
        id: 'ripgrep-core',
        name: 'ripgrep.main.rs',
        language: 'rust',
        url: 'https://raw.githubusercontent.com/BurntSushi/ripgrep/14.1.1/crates/core/main.rs',
      },
    ],
  },
  {
    name: 'zig',
    files: [
      {
        id: 'fx-worker-runtime',
        name: 'worker_runtime.zig',
        language: 'zig',
        url: 'https://raw.githubusercontent.com/vercel-labs/fx/main/src/core/agent/worker_runtime.zig',
      },
    ],
  },
  {
    name: 'java',
    files: [
      {
        id: 'junit-assertions',
        name: 'Assertions.java',
        language: 'java',
        url: 'https://raw.githubusercontent.com/junit-team/junit5/main/junit-jupiter-api/src/main/java/org/junit/jupiter/api/Assertions.java',
      },
    ],
  },
  {
    name: 'kotlin',
    files: [
      {
        id: 'ktor-application',
        name: 'Application.kt',
        language: 'kotlin',
        url: 'https://raw.githubusercontent.com/ktorio/ktor/main/ktor-server/ktor-server-core/common/src/io/ktor/server/application/Application.kt',
      },
    ],
  },
  {
    name: 'swift',
    files: [
      {
        id: 'alamofire-session',
        name: 'Session.swift',
        language: 'swift',
        url: 'https://raw.githubusercontent.com/Alamofire/Alamofire/master/Source/Core/Session.swift',
      },
    ],
  },
  {
    name: 'objective-c',
    files: [
      {
        id: 'afnetworking-session-manager',
        name: 'AFHTTPSessionManager.m',
        language: 'objective-c',
        url: 'https://raw.githubusercontent.com/AFNetworking/AFNetworking/master/AFNetworking/AFHTTPSessionManager.m',
      },
    ],
  },
  {
    name: 'php',
    files: [
      {
        id: 'laravel-model',
        name: 'Model.php',
        language: 'php',
        url: 'https://raw.githubusercontent.com/laravel/framework/12.x/src/Illuminate/Database/Eloquent/Model.php',
      },
    ],
  },
  {
    name: 'c',
    files: [
      {
        id: 'redis',
        name: 'redis.server.c',
        language: 'c',
        url: 'https://raw.githubusercontent.com/redis/redis/8.2.1/src/server.c',
      },
    ],
  },
  {
    name: 'c#',
    files: [
      {
        id: 'dotnet-string',
        name: 'String.cs',
        language: 'csharp',
        url: 'https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Private.CoreLib/src/System/String.cs',
      },
    ],
  },
  {
    name: 'ruby',
    files: [
      {
        id: 'rails',
        name: 'active_record.base.rb',
        language: 'ruby',
        url: 'https://raw.githubusercontent.com/rails/rails/v8.0.2/activerecord/lib/active_record/base.rb',
      },
    ],
  },
  {
    name: 'dart',
    files: [
      {
        id: 'flutter-framework',
        name: 'framework.dart',
        language: 'dart',
        url: 'https://raw.githubusercontent.com/flutter/flutter/master/packages/flutter/lib/src/widgets/framework.dart',
      },
    ],
  },
  {
    name: 'solidity',
    files: [
      {
        id: 'openzeppelin-erc20',
        name: 'ERC20.sol',
        language: 'solidity',
        url: 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/master/contracts/token/ERC20/ERC20.sol',
      },
    ],
  },
  {
    name: 'cuda',
    files: [
      {
        id: 'pytorch-cuda-sort',
        name: 'Sort.cu',
        language: 'cuda',
        url: 'https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Sort.cu',
      },
    ],
  },
  {
    name: 'hlsl',
    files: [
      {
        id: 'directx-hello-triangle-shaders',
        name: 'hello-triangle.shaders.hlsl',
        language: 'hlsl',
        url: 'https://raw.githubusercontent.com/microsoft/DirectX-Graphics-Samples/master/Samples/Desktop/D3D12HelloWorld/src/HelloTriangle/shaders.hlsl',
      },
    ],
  },
  {
    name: 'glsl',
    files: [
      {
        id: 'godot-canvas-shader',
        name: 'godot.canvas.glsl',
        language: 'glsl',
        url: 'https://raw.githubusercontent.com/godotengine/godot/master/servers/rendering/renderer_rd/shaders/canvas.glsl',
      },
    ],
  },
  {
    name: 'svg',
    files: [
      {
        id: 'github-icon',
        name: 'github.svg',
        language: 'svg',
        url: 'https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg',
      },
    ],
  },
  {
    name: 'json',
    files: [
      {
        id: 'typescript-package',
        name: 'typescript.package.json',
        language: 'json',
        url: 'https://raw.githubusercontent.com/microsoft/TypeScript/main/package.json',
      },
    ],
  },
  {
    name: 'yaml',
    files: [
      {
        id: 'checkout-action',
        name: 'checkout.action.yml',
        language: 'yaml',
        url: 'https://raw.githubusercontent.com/actions/checkout/main/action.yml',
      },
    ],
  },
  {
    name: 'c++',
    files: [
      {
        id: 'protobuf-ascii',
        name: 'ascii.cpp',
        language: 'cpp',
        url: 'https://raw.githubusercontent.com/protocolbuffers/protobuf/main/third_party/utf8_range/ascii.cpp',
      },
    ],
  },
  {
    name: 'docker',
    files: [
      {
        id: 'node-dockerfile',
        name: 'node.bookworm.Dockerfile',
        language: 'dockerfile',
        url: 'https://raw.githubusercontent.com/nodejs/docker-node/main/24/bookworm/Dockerfile',
      },
    ],
  },
  {
    name: 'makefile',
    files: [
      {
        id: 'redis-makefile',
        name: 'Makefile',
        language: 'makefile',
        url: 'https://raw.githubusercontent.com/redis/redis/unstable/src/Makefile',
      },
    ],
  },
  {
    name: 'cmake',
    files: [
      {
        id: 'cmake-root',
        name: 'cmake.CMakeLists.txt',
        language: 'cmake',
        url: 'https://raw.githubusercontent.com/Kitware/CMake/06675ff22831820a1bd7463643505728cc3d1441/CMakeLists.txt',
      },
    ],
  },
  {
    name: 'hcl',
    files: [
      {
        id: 'terraform-aws-vpc',
        name: 'vpc.main.tf',
        language: 'hcl',
        url: 'https://raw.githubusercontent.com/terraform-aws-modules/terraform-aws-vpc/master/main.tf',
      },
    ],
  },
  {
    name: 'astro',
    files: [
      {
        id: 'astro-blog-index',
        name: 'blog.index.astro',
        language: 'astro',
        url: 'https://raw.githubusercontent.com/withastro/astro/main/examples/blog/src/pages/index.astro',
      },
    ],
  },
  {
    name: 'solidjs',
    files: [
      {
        id: 'solid-app',
        name: 'App.tsx',
        language: 'tsx',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-solid-ts/src/App.tsx',
      },
      {
        id: 'solid-index',
        name: 'index.tsx',
        language: 'tsx',
        url: 'https://raw.githubusercontent.com/vitejs/vite/f40efefbb3630cdb7235286bc2b51673d9fbfc27/packages/create-vite/template-solid-ts/src/index.tsx',
      },
    ],
  },
  {
    name: 'haskell',
    files: [
      {
        id: 'haskell-sudoku',
        name: 'Sudoku.hs',
        language: 'haskell',
        url: 'https://raw.githubusercontent.com/github-linguist/linguist/d5214e1612c858ba14bf98edeca57e1683276f1d/samples/Haskell/Sudoku.hs',
      },
    ],
  },
  {
    name: 'markdown',
    files: [
      {
        id: 'react-readme',
        name: 'react.README.md',
        language: 'markdown',
        url: 'https://raw.githubusercontent.com/facebook/react/v19.2.0/README.md',
      },
    ],
  },
  {
    name: 'shell',
    files: [
      {
        id: 'nvm-install',
        name: 'nvm.install.sh',
        language: 'shell',
        url: 'https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh',
      },
    ],
  },
  {
    name: 'powershell',
    files: [
      {
        id: 'powershell-installer',
        name: 'install-powershell.ps1',
        language: 'powershell',
        url: 'https://raw.githubusercontent.com/PowerShell/PowerShell/6ca24ccf024140ca061580c43751db27605c4815/tools/install-powershell.ps1',
      },
    ],
  },
  ...languageProbes.results.map((probe) => ({
    name: probe.id,
    files: [
      {
        id: `probe-${probe.id}`,
        name: probe.fileName,
        language: probe.id,
        url: probe.url,
      },
    ],
  })),
] as const

const demoFiles = demoFolders.flatMap((folder) => folder.files)
const demoLanguageCount = new Set(demoFiles.map((file) => file.language)).size
const initialFile = demoFiles[0]
const sourceCache = new Map<string, string>()
const warmupCode = 'const value = 42; // warm WebGPU'
const EDIT_HIGHLIGHT_DELAY_MS = 120
const MAX_RENDERED_SOURCE_LENGTH = 20_000
const REDACTION_NOTICE = '\n[...redacted for DOM render perf...]'

type State =
  | { kind: 'loading' }
  | { kind: 'running' }
  | { kind: 'done'; spans: HighlightSpan[]; elapsed: number; source: string }
  | { kind: 'error'; message: string }

type Runtime = typeof import('gpu-lexer')
let runtimePromise: Promise<Runtime> | undefined
let warmedRuntimePromise: Promise<Runtime> | undefined
function loadRuntime() {
  runtimePromise ??= import('gpu-lexer')
  return runtimePromise
}

function loadWarmedRuntime() {
  warmedRuntimePromise ??= loadRuntime().then(async (runtime) => {
    await runtime.parse(warmupCode)
    return runtime
  })
  return warmedRuntimePromise
}

export function Demo() {
  const [activeId, setActiveId] = useState<string>(initialFile.id)
  const [code, setCode] = useState<string>('')
  const [state, setState] = useState<State>({ kind: 'loading' })
  const editorRef = useRef<HTMLPreElement>(null)
  const editorSnapshotRef = useRef<string | undefined>(undefined)
  const requestId = useRef(0)
  const editHighlightTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  const activeFile =
    demoFiles.find((file) => file.id === activeId) ?? initialFile

  useEffect(() => {
    const warm = () => void loadWarmedRuntime().catch(() => {})
    if ('requestIdleCallback' in window) {
      const id = window.requestIdleCallback(warm, { timeout: 500 })
      return () => window.cancelIdleCallback(id)
    }
    const id = globalThis.setTimeout(warm)
    return () => globalThis.clearTimeout(id)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void openFile(activeFile, controller.signal)
    return () => {
      controller.abort()
      clearTimeout(editHighlightTimer.current)
    }
  }, [activeFile])

  async function openFile(file: DemoFile, signal: AbortSignal) {
    const currentRequest = ++requestId.current
    clearTimeout(editHighlightTimer.current)
    editorSnapshotRef.current = undefined
    setCode('')
    setState({ kind: 'loading' })
    try {
      let source = sourceCache.get(file.url)
      if (source === undefined) {
        const response = await fetch(file.url, { signal })
        if (!response.ok)
          throw new Error(`download failed with HTTP ${response.status}`)
        source = await response.text()
        sourceCache.set(file.url, source)
      }
      if (signal.aborted || requestId.current !== currentRequest) return
      const rendered = renderableSource(source)
      setCode(rendered.text)
      setState({ kind: 'running' })
      const { parse } = await loadWarmedRuntime()
      const started = performance.now()
      const spans = await parse(source.slice(0, rendered.sourceEnd))
      if (requestId.current === currentRequest) {
        startTransition(() => {
          setState({
            kind: 'done',
            spans,
            elapsed: performance.now() - started,
            source: rendered.text,
          })
        })
      }
    } catch (error) {
      if (requestId.current === currentRequest) {
        setState({
          kind: 'error',
          message:
            error instanceof Error ? error.message : 'Highlighting failed.',
        })
      }
    }
  }

  function scheduleEditorHighlight(immediate = false) {
    requestId.current += 1
    clearTimeout(editHighlightTimer.current)
    editHighlightTimer.current = setTimeout(
      () => {
        const editor = editorRef.current
        const source = editor?.textContent ?? ''
        const rendered = renderableSource(source)
        if (editor && source !== rendered.text) {
          const selection = readSelection(editor)
          editor.replaceChildren(document.createTextNode(rendered.text))
          if (selection) restoreSelection(editor, selection)
        }
        editorSnapshotRef.current = rendered.text
        setCode(rendered.text)
        if (source) void highlightSource(source, rendered)
        else setState({ kind: 'done', spans: [], elapsed: 0, source: '' })
      },
      immediate ? 0 : EDIT_HIGHLIGHT_DELAY_MS,
    )
  }

  async function highlightSource(
    source: string,
    rendered = renderableSource(source),
  ) {
    const currentRequest = ++requestId.current
    try {
      const { parse } = await loadWarmedRuntime()
      const started = performance.now()
      const spans = await parse(source.slice(0, rendered.sourceEnd))
      if (requestId.current === currentRequest) {
        startTransition(() => {
          setState({
            kind: 'done',
            spans,
            elapsed: performance.now() - started,
            source: rendered.text,
          })
        })
      }
    } catch (error) {
      if (requestId.current === currentRequest) {
        setState({
          kind: 'error',
          message:
            error instanceof Error ? error.message : 'Highlighting failed.',
        })
      }
    }
  }

  function insertIndent(event: React.KeyboardEvent<HTMLPreElement>) {
    if (event.key !== 'Tab') return
    event.preventDefault()
    const editor = event.currentTarget
    insertTextAtSelection(editor, '  ')
    scheduleEditorHighlight()
  }

  function insertLineBreak(event: React.FormEvent<HTMLPreElement>) {
    const inputType = (event.nativeEvent as InputEvent).inputType
    if (inputType !== 'insertParagraph' && inputType !== 'insertLineBreak')
      return
    event.preventDefault()
    const editor = event.currentTarget
    insertTextAtSelection(editor, '\n')
    scheduleEditorHighlight()
  }

  function insertPaste(event: React.ClipboardEvent<HTMLPreElement>) {
    event.preventDefault()
    const editor = event.currentTarget
    insertTextAtSelection(editor, event.clipboardData.getData('text/plain'))
    scheduleEditorHighlight(true)
  }

  const placeholder = !code
    ? state.kind === 'error'
      ? `error: ${state.message}`
      : state.kind === 'running'
        ? '# classifying the complete file on WebGPU…'
        : '# waiting for source…'
    : undefined

  useLayoutEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    if (editorSnapshotRef.current === code) {
      editorSnapshotRef.current = undefined
      return
    }
    if (editor.textContent === code) return
    const selection = readSelection(editor)
    editor.replaceChildren(document.createTextNode(code))
    if (selection) restoreSelection(editor, selection)
  }, [code])

  useLayoutEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    if (state.kind === 'done') {
      paintSyntaxHighlights(editor, state.spans)
    } else {
      clearSyntaxHighlights()
    }
  }, [state])

  useEffect(() => () => clearSyntaxHighlights(), [])

  return (
    <section className='workbench' aria-label='Syntax highlighting demo'>
      <div className='charts-head'>
        <p>[ live demos ]</p>
        <span>{demoLanguageCount} languages</span>
      </div>
      <div className='code-browser'>
        <aside
          className='file-explorer'
          aria-label='Famous project source files'
        >
          <div className='file-list' aria-label='Project source files'>
            {demoFiles.map((file) => (
              <button
                aria-pressed={file.id === activeId}
                className='file-row'
                key={file.id}
                onClick={() => setActiveId(file.id)}
                type='button'
              >
                <span aria-hidden='true'>·</span>
                <span>{file.name}</span>
              </button>
            ))}
          </div>
        </aside>

        <div className='code-workspace'>
          <div className='pane output-pane'>
            <div className='pane-label'>
              <span>{activeFile.name}</span>
              <span>
                {isRenderedSourceCapped(code)
                  ? '20KB (capped)'
                  : formatUnits(code.length)}
              </span>
            </div>
            <div
              className='highlight-editor'
              onPointerDown={(event) => {
                if (event.target !== event.currentTarget) return
                editorRef.current?.focus({ preventScroll: true })
              }}
            >
              <pre
                aria-label={`Edit highlighted code from ${activeFile.name}`}
                aria-multiline='true'
                autoCapitalize='off'
                className='highlight-surface'
                contentEditable='plaintext-only'
                data-error={state.kind === 'error' ? '' : undefined}
                data-placeholder={placeholder}
                onBeforeInput={insertLineBreak}
                onInput={(event) => {
                  const inputType = (event.nativeEvent as InputEvent).inputType
                  scheduleEditorHighlight(inputType === 'insertFromPaste')
                }}
                onKeyDown={insertIndent}
                onPaste={insertPaste}
                ref={editorRef}
                role='textbox'
                spellCheck={false}
                suppressContentEditableWarning
              />
            </div>
          </div>
          <div className='file-statusbar'>
            <a href={activeFile.url} rel='noreferrer' target='_blank'>
              {activeFile.url}
            </a>
            <span>
              {state.kind === 'done' && state.source === code
                ? `${state.elapsed.toFixed(2)}ms`
                : code
                  ? '…'
                  : '—'}
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}

type HighlightRegistry = {
  delete(name: string): boolean
  set(name: string, highlight: unknown): void
}

type TextRun = { node: Text; start: number; end: number }
type SelectionSnapshot = { anchor: number; focus: number }

function highlightApi() {
  const registry = (CSS as typeof CSS & { highlights?: HighlightRegistry })
    .highlights
  const HighlightClass = (
    globalThis as typeof globalThis & {
      Highlight?: new (...ranges: Range[]) => unknown
    }
  ).Highlight
  return registry && HighlightClass ? { HighlightClass, registry } : undefined
}

function paintSyntaxHighlights(
  editor: HTMLElement,
  spans: readonly HighlightSpan[],
) {
  const api = highlightApi()
  if (!api) return
  const ranges = Object.fromEntries(
    styledSyntaxTypes.map((type) => [type, [] as Range[]]),
  ) as Record<StyledSyntaxType, Range[]>
  const runs = textRuns(editor)
  const length = editor.textContent?.length ?? 0
  for (const span of spans) {
    if (span.start >= length) break
    if (span.type === 'plain') continue
    const range = rangeAtOffsets(runs, span.start, Math.min(span.end, length))
    if (range) ranges[span.type].push(range)
  }
  for (const type of styledSyntaxTypes) {
    api.registry.set(
      highlightName(type),
      new api.HighlightClass(...ranges[type]),
    )
  }
}

function clearSyntaxHighlights() {
  const api = highlightApi()
  if (!api) return
  for (const type of styledSyntaxTypes) api.registry.delete(highlightName(type))
}

function textRuns(root: HTMLElement) {
  const runs: TextRun[] = []
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let start = 0
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node as Text
    const end = start + text.data.length
    runs.push({ node: text, start, end })
    start = end
  }
  return runs
}

function rangeAtOffsets(runs: readonly TextRun[], start: number, end: number) {
  if (start >= end) return
  const from = textPoint(runs, start)
  const to = textPoint(runs, end)
  if (!from || !to) return
  const range = document.createRange()
  range.setStart(from.node, from.offset)
  range.setEnd(to.node, to.offset)
  return range
}

function textPoint(runs: readonly TextRun[], offset: number) {
  let low = 0
  let high = runs.length - 1
  while (low <= high) {
    const middle = (low + high) >> 1
    const run = runs[middle]
    if (offset < run.start) high = middle - 1
    else if (offset > run.end) low = middle + 1
    else return { node: run.node, offset: offset - run.start }
  }
}

function insertTextAtSelection(root: HTMLElement, text: string) {
  const selection = document.getSelection()
  if (!selection?.rangeCount || !root.contains(selection.anchorNode)) {
    root.append(document.createTextNode(text))
    return
  }
  const range = selection.getRangeAt(0)
  range.deleteContents()
  const node = document.createTextNode(text)
  range.insertNode(node)
  range.setStart(node, node.data.length)
  range.collapse(true)
  selection.removeAllRanges()
  selection.addRange(range)
}

function readSelection(root: HTMLElement): SelectionSnapshot | undefined {
  if (document.activeElement !== root) return
  const selection = document.getSelection()
  if (
    !selection?.anchorNode ||
    !selection.focusNode ||
    !root.contains(selection.anchorNode) ||
    !root.contains(selection.focusNode)
  )
    return
  return {
    anchor: textOffset(root, selection.anchorNode, selection.anchorOffset),
    focus: textOffset(root, selection.focusNode, selection.focusOffset),
  }
}

function restoreSelection(root: HTMLElement, snapshot: SelectionSnapshot) {
  const runs = textRuns(root)
  const length = root.textContent?.length ?? 0
  const anchor = textPoint(runs, Math.min(snapshot.anchor, length))
  const focus = textPoint(runs, Math.min(snapshot.focus, length))
  if (anchor && focus) {
    document
      .getSelection()
      ?.setBaseAndExtent(anchor.node, anchor.offset, focus.node, focus.offset)
  }
}

function textOffset(root: HTMLElement, node: Node, offset: number) {
  const range = document.createRange()
  range.selectNodeContents(root)
  range.setEnd(node, offset)
  return range.toString().length
}

function formatUnits(value: number) {
  if (value < 1000) return `${value} bytes`
  return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)} KB`
}

function renderableSource(source: string) {
  if (source.length <= MAX_RENDERED_SOURCE_LENGTH)
    return { text: source, sourceEnd: source.length }
  return {
    text: source.slice(0, MAX_RENDERED_SOURCE_LENGTH) + REDACTION_NOTICE,
    sourceEnd: MAX_RENDERED_SOURCE_LENGTH,
  }
}

function isRenderedSourceCapped(source: string) {
  return (
    source.length === MAX_RENDERED_SOURCE_LENGTH + REDACTION_NOTICE.length &&
    source.endsWith(REDACTION_NOTICE)
  )
}

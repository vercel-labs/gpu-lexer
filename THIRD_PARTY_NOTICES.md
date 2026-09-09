# Third-party notices

`gpu-lexer` depends on and was evaluated with third-party software. Their names identify upstream projects and do not imply endorsement.

The training pipeline uses [Shiki](https://github.com/shikijs/shiki) under the MIT License to produce offline TextMate-scope labels. The benchmark and comparison site also use Shiki, Highlight.js, Prism, Sugar High, Starry Night, and their transitive dependencies under their respective licenses. These libraries are development or website dependencies and are not bundled into the published `gpu-lexer` core package.

The build uses [esbuild](https://github.com/evanw/esbuild) and [wgslender](https://github.com/marijnh/wgslender), both under their respective open-source licenses. Build tools are not runtime dependencies of the published package.

Training and verification source repositories and npm packages are pinned with declared licenses in `packages/training/data/corpus.json`. Downloaded source archives and generated corpus shards are intentionally excluded from this repository and from the npm package. Copyright in those source files remains with their original authors.

The explainer source references Geist and Geist Mono font files from [vercel/geist-font](https://github.com/vercel/geist-font), distributed by their authors under the SIL Open Font License 1.1.

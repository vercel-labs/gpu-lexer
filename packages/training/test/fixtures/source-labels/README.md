# Source-label fixtures

These small files exercise direct Shiki-to-runtime-part alignment for Batch, HTML with embedded code, Make, Markdown fences, and ShaderLab. The audit is representation-only: it checks source offsets, coverage, and selected semantic spans without running a learned model.

Run it with:

```sh
node packages/training/src/source-label-baseline.js
```

Missing or malformed direct source labels fail closed. There is no tokenizer-label fallback.

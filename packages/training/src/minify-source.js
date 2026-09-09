import { transform as transformJavaScript } from "esbuild";
import { minify as minifyHtml } from "html-minifier-terser";
import { transform as transformCss } from "lightningcss";

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

/** Deterministically minify web source for repository-disjoint hard-example mining. */
export async function minifySource(source, language, path = "input") {
  if (["javascript", "jsx", "typescript", "tsx"].includes(language)) {
    const loader = language === "javascript" ? "js" : language === "typescript" ? "ts" : language;
    return (await transformJavaScript(source, {
      loader,
      minify: true,
      legalComments: "none",
      target: "esnext",
      sourcefile: path,
    })).code.trim();
  }
  if (["css", "scss"].includes(language)) {
    return textDecoder.decode(transformCss({
      filename: path,
      code: textEncoder.encode(source),
      minify: true,
    }).code);
  }
  if (language === "html") {
    return minifyHtml(source, {
      collapseWhitespace: true,
      removeComments: true,
      removeAttributeQuotes: true,
      removeRedundantAttributes: true,
      removeScriptTypeAttributes: true,
      removeStyleLinkTypeAttributes: true,
      sortAttributes: true,
      sortClassName: true,
      useShortDoctype: true,
      minifyCSS: true,
      minifyJS: true,
    });
  }
  return null;
}

const OPERATOR = /^[+*/%=!&|^~?:<>-]+$/;

export function labelsFromHighlightedHtml(source, html, engine, { family = "" } = {}) {
  const labels = [];
  const stack = [[]];
  let offset = 0;
  for (const match of html.matchAll(/<[^>]*>|[^<]+/g)) {
    const fragment = match[0];
    if (fragment[0] === "<") {
      if (/^<\/span\b/i.test(fragment)) stack.pop();
      else if (/^<span\b/i.test(fragment)) {
        const classes = /\bclass=(?:"([^"]*)"|'([^']*)')/i.exec(fragment);
        stack.push((classes?.[1] ?? classes?.[2] ?? "").split(/\s+/).filter(Boolean));
      } else if (/^<br\s*\/?\s*>$/i.test(fragment)) append("\n");
      continue;
    }
    append(decodeEntities(fragment));
  }
  if (offset !== source.length) {
    throw new Error(`${engine} output reconstructed ${offset}/${source.length} source units`);
  }
  return labels;

  function append(value) {
    if (!value) return;
    if (source.slice(offset, offset + value.length) !== value) {
      throw new Error(`${engine} output diverged from source at ${offset}`);
    }
    const name = classify(engine, stack, value, family);
    const previous = labels.at(-1);
    if (previous?.class === name && previous.to === offset) previous.to += value.length;
    else labels.push({ from: offset, to: offset + value.length, class: name, confidence: 1 });
    offset += value.length;
  }
}

function classify(engine, stack, value, family) {
  if (engine === "prism.js" || engine === "prism") {
    const all = new Set(stack.flat());
    if (all.has("interpolation-punctuation")) return "operator";
    if (all.has("code-snippet")) return "string";
    if (all.has("title") && all.has("important")) return "keyword";
  }
  for (let index = stack.length - 1; index > 0; index--) {
    const name = classifyFrame(engine, stack[index], value, family);
    if (name !== undefined) return name;
  }
  return "plain";
}

function classifyFrame(engine, classes, value, family) {
  if (engine === "starry-night") {
    const names = new Set(classes);
    if (names.has("pl-c")) return "comment";
    if (["pl-s", "pl-pds", "pl-sr", "pl-cce", "pl-sre", "pl-sra", "pl-corl"]
      .some((name) => names.has(name))) return "string";
    if (names.has("pl-c1")) return /^[-+]?(?:\d|\.\d)/.test(value) ? "number" : "constant";
    if (names.has("pl-kos")) return OPERATOR.test(value) ? "operator" : "keyword";
    if (names.has("pl-k") || names.has("pl-mh") || names.has("pl-ms")) return "keyword";
    if (names.has("pl-ent")) return "type";
    if (names.has("pl-e") || names.has("pl-en")) return "function";
    if (names.has("pl-pse")) return "operator";
    return "plain";
  }
  if (engine === "sugar-high") {
    const type = classes.find((name) => name.startsWith("sh__token--"))?.slice(11);
    if (!type) return undefined;
    if (type === "comment") return "comment";
    if (type === "string") return "string";
    if (type === "keyword") return "keyword";
    if (type === "class") {
      if (/^\d/.test(value)) return "number";
      if (/^(?:false|infinity|nan|null|true|undefined)$/i.test(value)) return "constant";
      return "type";
    }
    if (type === "entity") return "type";
    if (type === "property") return family === "css" ? "type" : "plain";
    if (type === "sign") return OPERATOR.test(value) ? "operator" : "plain";
    return "plain";
  }

  const names = new Set(classes.map((name) => engine === "highlight.js"
    ? name.replace(/^hljs-/, "") : name));
  if (engine === "highlight.js") {
    if (names.has("comment")) return "comment";
    if (["string", "regexp", "char", "link", "code", "meta-string"]
      .some((name) => names.has(name))) return "string";
    if (names.has("number")) return "number";
    if (names.has("title") && names.has("class_")) return "type";
    if (names.has("title") && names.has("function_")) return "function";
    if (names.has("type") || names.has("name") || names.has("selector-tag")) return "type";
    if (family === "css" && (names.has("attribute") || names.has("property"))) return "type";
    if (names.has("title") || names.has("function") || names.has("built_in")) return "function";
    if (names.has("symbol")) return family === "html" ? "plain" : "string";
    if (names.has("literal")) return "constant";
    if (names.has("operator")) return "operator";
    if (names.has("keyword") || names.has("meta-keyword") || names.has("section")) return "keyword";
    if (["subst", "template-variable", "params", "attr", "attribute", "variable",
      "property", "punctuation", "tag", "quote"].some((name) => names.has(name))) return "plain";
    return undefined;
  }

  if (names.has("comment")) return "comment";
  if (["string", "char", "regex", "attr-value", "template-string", "cdata", "symbol", "url"]
    .some((name) => names.has(name))) return "string";
  if (names.has("number")) return "number";
  if (names.has("class-name") || names.has("builtin") || names.has("doctype-tag")) return "type";
  if (family === "css" && names.has("property")) return "type";
  if (family === "css" && names.has("selector") && /^[-a-z][\w-]*$/i.test(value)) return "type";
  if (names.has("function") || names.has("function-variable")) return "function";
  if (["boolean", "constant"].some((name) => names.has(name))) return "constant";
  if (names.has("operator") || names.has("interpolation-punctuation")) return "operator";
  if (["keyword", "control-flow", "directive", "macro", "important", "atrule", "rule"]
    .some((name) => names.has(name))) return "keyword";
  if (names.has("tag")) return "type";
  if (["interpolation", "punctuation", "attr-name", "parameter", "property", "variable",
    "namespace", "plain-text", "prolog", "name", "blockquote", "entity"]
    .some((name) => names.has(name))) return "plain";
  return undefined;
}

function decodeEntities(value) {
  return value.replace(/&(?:#(\d+)|#x([\da-f]+)|amp|lt|gt|quot|apos|#039);/gi, (entity, decimal, hex) => {
    if (decimal) return String.fromCodePoint(Number(decimal));
    if (hex) return String.fromCodePoint(Number.parseInt(hex, 16));
    return { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'", "&#039;": "'" }[entity.toLowerCase()];
  });
}

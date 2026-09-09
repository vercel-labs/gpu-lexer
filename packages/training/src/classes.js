export const classNames = Object.freeze([
  "plain",
  "comment",
  "string",
  "number",
  "keyword",
  "type",
  "function",
  "constant",
  "operator",
]);

export const auxiliaryNames = Object.freeze([
  "comment-state",
  "string-state",
  "directive-line",
  "markup-tag",
  "embedded-region",
  "selector-region",
  "declaration-name",
  "declaration-value",
  "member-access",
  "clause-region",
]);

const auxiliaryRules = [
  /(?:^|\.)comment(?:\.|$)/,
  /(?:^|\.)(?:string|regexp)(?:\.|$)/,
  /(?:^|\.)(?:preprocessor|directive|at-rule)(?:\.|$)/,
  /(?:^|\.)(?:tag|markup|attribute-name|attribute-value)(?:\.|$)/,
  /(?:^|\.)(?:embedded|template\.expression)(?:\.|$)/,
  /(?:^|\.)(?:selector|attribute-name\.class|attribute-name\.id)(?:\.|$)/,
  /(?:^|\.)(?:property-name)(?:\.|$)/,
  /(?:^|\.)(?:property-value|declaration-value)(?:\.|$)/,
  /(?:^|\.)(?:property-access|object-member|member|variable\.other\.property|support\.variable\.property)(?:\.|$)/,
  /(?:^|\.)(?:select|insert|update|delete|create|alter|from|where|join|clause)(?:\.|$)/,
];

const rules = [
  ["comment", /^comment\./],
  ["string", /^(?:string|constant\.other\.symbol|constant\.regexp|markup\.inline\.raw|markup\.underline\.link)\./],
  ["number", /^constant\.numeric\./],
  ["operator", /^keyword\.operator(?:\.|$)/],
  ["keyword", /^(?:keyword(?!\.operator(?:\.|$))|storage|modifier|control|markup\.heading)(?:\.|$)/],
  ["type", /^(?:entity\.name\.(?:type|class|interface|struct|enum|tag)|support\.(?:type|class))\./],
  ["function", /^(?:entity\.name\.(?:function|method)|support\.function)\./],
  ["constant", /^(?:constant\.(?:language|other)|support\.constant)(?:\.|$)/],
];

/** Collapse nested TextMate scopes into gpu-lexer's display taxonomy. */
export function classFromScopes(scopes) {
  scopes = normalizeScopes(scopes);
  if (scopes.some((scope) => /^punctuation\.definition\.template-expression(?:\.|$)/.test(scope))) {
    return "operator";
  }
  scopes = withoutOuterTemplateString(scopes);
  for (const [name, pattern] of rules) {
    if (scopes.some((scope) => pattern.test(scope))) return name;
  }
  return "plain";
}

export function auxiliaryFromScopes(scopes) {
  scopes = normalizeScopes(scopes);
  scopes = withoutOuterTemplateString(scopes);
  let bits = 0;
  for (let index = 0; index < auxiliaryRules.length; index++) {
    if (scopes.some((scope) => auxiliaryRules[index].test(scope))) bits |= 1 << index;
  }
  return bits;
}

export function confidenceFromScopes(scopes) {
  scopes = withoutOuterTemplateString(normalizeScopes(scopes));
  if (scopes.some((scope) => /^invalid(?:\.|$)/.test(scope))) return 0.25;
  const matches = new Set();
  for (const [name, pattern] of rules) {
    if (scopes.some((scope) => pattern.test(scope))) matches.add(name);
  }
  if (matches.size > 1) return 0.5;
  if (scopes.some((scope) => /(?:^|\.)(?:unknown|unparsed|illegal)(?:\.|$)/.test(scope))) return 0.5;
  return 1;
}

export function normalizeScopes(scopes) {
  return [...new Set(scopes
    .filter((scope) => typeof scope === "string")
    .map((scope) => scope.trim().toLowerCase())
    .filter(Boolean))];
}

function withoutOuterTemplateString(scopes) {
  const embedded = scopes.some((scope) => /^meta\.(?:embedded|template\.expression)(?:\.|$)/.test(scope));
  return embedded ? scopes.filter((scope) => !/^string\.template(?:\.|$)/.test(scope)) : scopes;
}

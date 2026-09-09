export function confusionMatrix(expected, actual, classes) {
  if (expected.length !== actual.length) throw new Error("label arrays must have equal length");
  const matrix = Object.fromEntries(
    classes.map((name) => [name, Object.fromEntries(classes.map((actualName) => [actualName, 0]))]),
  );
  for (let index = 0; index < expected.length; index++) {
    matrix[expected[index]][actual[index]] += 1;
  }
  return matrix;
}

export function classificationMetrics(expected, actual, classes, { exclude = [] } = {}) {
  const matrix = confusionMatrix(expected, actual, classes);
  const excluded = new Set(exclude);
  const perClass = {};
  let macroF1 = 0;
  let macroCount = 0;

  for (const name of classes) {
    const truePositive = matrix[name][name];
    const falsePositive = classes.reduce(
      (total, expectedName) => total + (expectedName === name ? 0 : matrix[expectedName][name]), 0,
    );
    const falseNegative = classes.reduce(
      (total, actualName) => total + (actualName === name ? 0 : matrix[name][actualName]), 0,
    );
    const precision = truePositive / (truePositive + falsePositive || 1);
    const recall = truePositive / (truePositive + falseNegative || 1);
    const f1 = (2 * precision * recall) / (precision + recall || 1);
    perClass[name] = { precision, recall, f1, support: truePositive + falseNegative };
    if (!excluded.has(name)) {
      macroF1 += f1;
      macroCount += 1;
    }
  }

  const correct = expected.reduce((total, label, index) => total + Number(label === actual[index]), 0);
  return {
    accuracy: correct / (expected.length || 1),
    macroF1: macroF1 / (macroCount || 1),
    perClass,
    confusion: matrix,
  };
}

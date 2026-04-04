import { describe, expect, test } from "bun:test";
import { LogisticRegression } from "../../src/models/logistic-regression";

describe("LogisticRegression", () => {
  test("separable data: converges to correct boundary", () => {
    // Simple linearly separable data
    const X = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];
    const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);

    // Should classify correctly
    expect(model.predict([-3])[0]).toBe(0);
    expect(model.predict([3])[0]).toBe(1);
  });

  test("probabilities are between 0 and 1", () => {
    const X = [-3, -2, -1, 0, 1, 2, 3];
    const y = [0, 0, 0, 0, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);

    const probs = model.predictProbability([-10, 0, 10]);
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  test("multi-feature classification", () => {
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [2, 2],
      [2, 3],
      [3, 2],
      [3, 3],
    ];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);

    expect(model.predict([[0, 0]])[0]).toBe(0);
    expect(model.predict([[3, 3]])[0]).toBe(1);
  });

  test("statistics returns valid metrics", () => {
    const X = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];
    const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);
    const stats = model.statistics();

    expect(stats.accuracy).toBe(1);
    expect(stats.precision).toBe(1);
    expect(stats.recall).toBe(1);
    expect(stats.f1Score).toBe(1);
    expect(stats.confusionMatrix.truePositives).toBe(5);
    expect(stats.confusionMatrix.trueNegatives).toBe(5);
    expect(stats.confusionMatrix.falsePositives).toBe(0);
    expect(stats.confusionMatrix.falseNegatives).toBe(0);
    expect(stats.pseudoRSquared).toBeGreaterThan(0);
    expect(stats.logLikelihood).toBeLessThan(0);
  });

  test("rejects non-binary y", () => {
    const model = new LogisticRegression();
    expect(() => model.fit([1, 2, 3], [0, 1, 2])).toThrow("binary y");
  });

  test("throws if not fitted", () => {
    const model = new LogisticRegression();
    expect(() => model.coefficients).toThrow("not been fitted");
  });
});

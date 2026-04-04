import { describe, expect, test } from "bun:test";
import { LassoRegression } from "../../src/models/lasso-regression";

describe("LassoRegression", () => {
  test("with very small alpha, approximates OLS", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x + 1);

    const model = new LassoRegression({ alpha: 1e-6 });
    model.fit(X, y);
    expect(model.coefficients[0]).toBeCloseTo(2, 1);
    expect(model.intercept).toBeCloseTo(1, 0);
  });

  test("sparsity: large alpha zeros out coefficients", () => {
    const X = Array.from({ length: 20 }, (_, i) => [i, i * 0.001 + Math.sin(i)]);
    const y = X.map(([x1]) => 2 * x1! + 1);

    const model = new LassoRegression({ alpha: 10 });
    model.fit(X, y);

    // At least one coefficient should be zero (the irrelevant one)
    const hasZero = model.coefficients.some((c) => Math.abs(c) < 1e-10);
    expect(hasZero).toBe(true);
  });

  test("very large alpha makes all coefficients zero", () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const y = [1, 2, 3];
    const model = new LassoRegression({ alpha: 1000 });
    model.fit(X, y);

    for (const c of model.coefficients) {
      expect(Math.abs(c)).toBeLessThan(1e-6);
    }
  });

  test("prediction works", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 3 * x);
    const model = new LassoRegression({ alpha: 0.01 });
    model.fit(X, y);
    const pred = model.predict([5]);
    expect(pred[0]).toBeCloseTo(15, 0);
  });
});

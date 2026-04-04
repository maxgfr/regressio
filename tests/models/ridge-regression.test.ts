import { describe, expect, test } from "bun:test";
import { LinearRegression } from "../../src/models/linear-regression";
import { RidgeRegression } from "../../src/models/ridge-regression";

describe("RidgeRegression", () => {
  test("alpha=0 matches OLS", () => {
    const X = [
      [1, 3],
      [3, 1],
      [5, 7],
      [7, 2],
      [9, 5],
    ];
    const y = [3, 7, 11, 15, 19];

    const ols = new LinearRegression();
    ols.fit(X, y);
    const ridge = new RidgeRegression({ alpha: 0 });
    ridge.fit(X, y);

    expect(ridge.intercept).toBeCloseTo(ols.intercept, 4);
    for (let i = 0; i < ols.coefficients.length; i++) {
      expect(ridge.coefficients[i]).toBeCloseTo(ols.coefficients[i]!, 4);
    }
  });

  test("large alpha shrinks coefficients toward zero", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x + 5);

    const small = new RidgeRegression({ alpha: 0.01 });
    small.fit(X, y);

    const large = new RidgeRegression({ alpha: 1000 });
    large.fit(X, y);

    expect(Math.abs(large.coefficients[0]!)).toBeLessThan(Math.abs(small.coefficients[0]!));
  });

  test("intercept is not regularized", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x + 100); // large intercept

    const ridge = new RidgeRegression({ alpha: 10 });
    ridge.fit(X, y);

    // Intercept should still be large since it's not penalized
    expect(ridge.intercept).toBeGreaterThan(50);
  });

  test("prediction works", () => {
    const X = [1, 2, 3, 4, 5];
    const y = X.map((x) => 3 * x + 1);
    const model = new RidgeRegression({ alpha: 0.01 });
    model.fit(X, y);
    const pred = model.predict([10]);
    expect(pred[0]).toBeCloseTo(31, 0);
  });
});

import { describe, expect, test } from "bun:test";
import { ElasticNet } from "../../src/models/elastic-net";
import { LassoRegression } from "../../src/models/lasso-regression";

describe("ElasticNet", () => {
  test("l1Ratio=1 matches Lasso", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 3 * x + 1);

    const lasso = new LassoRegression({ alpha: 0.1 });
    lasso.fit(X, y);

    const elastic = new ElasticNet({ alpha: 0.1, l1Ratio: 1.0 });
    elastic.fit(X, y);

    expect(elastic.coefficients[0]).toBeCloseTo(lasso.coefficients[0]!, 2);
  });

  test("intermediate l1Ratio produces reasonable results", () => {
    const X = Array.from({ length: 20 }, (_, i) => [i, i * 2 + 1]);
    const y = X.map(([x1, x2]) => 2 * x1! + 0.5 * x2! + 1);

    const model = new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 });
    model.fit(X, y);

    // Should produce non-zero coefficients
    expect(model.coefficients.some((c) => Math.abs(c) > 0.01)).toBe(true);
  });

  test("l1Ratio=0 approximates Ridge (pure L2, no sparsity)", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 3 * x + 1);

    const elastic = new ElasticNet({ alpha: 0.01, l1Ratio: 0.0 });
    elastic.fit(X, y);

    // With l1Ratio=0, no sparsity — coefficient should be non-zero and close to true
    expect(Math.abs(elastic.coefficients[0]!)).toBeGreaterThan(0.1);
    expect(elastic.coefficients[0]).toBeCloseTo(3, 0);
  });

  test("prediction works", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x);
    const model = new ElasticNet({ alpha: 0.01 });
    model.fit(X, y);
    const pred = model.predict([5]);
    expect(pred[0]).toBeCloseTo(10, 0);
  });
});

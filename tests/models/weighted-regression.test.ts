import { describe, expect, test } from "bun:test";
import { LinearRegression } from "../../src/models/linear-regression";
import { WeightedRegression } from "../../src/models/weighted-regression";

describe("WeightedRegression", () => {
  test("uniform weights match OLS", () => {
    const X = [1, 2, 3, 4, 5];
    const y = [2.1, 3.9, 6.2, 7.8, 10.1];
    const weights = [1, 1, 1, 1, 1];

    const ols = new LinearRegression();
    ols.fit(X, y);

    const wls = new WeightedRegression();
    wls.fit(X, y, weights);

    expect(wls.intercept).toBeCloseTo(ols.intercept, 6);
    expect(wls.coefficients[0]).toBeCloseTo(ols.coefficients[0]!, 6);
  });

  test("higher weight on an observation pulls the line toward it", () => {
    // Point (3, 100) is an outlier
    const X = [1, 2, 3, 4, 5];
    const y = [2, 4, 100, 8, 10];

    // Without weighting: outlier has full influence
    const ols = new LinearRegression();
    ols.fit(X, y);

    // With low weight on outlier: less influence
    const wls = new WeightedRegression();
    wls.fit(X, y, [1, 1, 0.01, 1, 1]);

    // WLS slope should be closer to 2 (the true slope without outlier)
    const wlsDiff = Math.abs(wls.coefficients[0]! - 2);
    const olsDiff = Math.abs(ols.coefficients[0]! - 2);
    expect(wlsDiff).toBeLessThanOrEqual(olsDiff + 1e-10);
  });

  test("prediction works", () => {
    const X = [1, 2, 3, 4, 5];
    const y = [3, 5, 7, 9, 11];
    const model = new WeightedRegression();
    model.fit(X, y, [1, 1, 1, 1, 1]);
    const pred = model.predict([6]);
    expect(pred[0]).toBeCloseTo(13, 0);
  });

  test("throws without weights", () => {
    const model = new WeightedRegression();
    expect(() => model.fit([1, 2], [1, 2])).toThrow("Weights must be provided");
  });
});

import { describe, expect, test } from "bun:test";
import { LinearRegression } from "../../src/models/linear-regression";

describe("LinearRegression", () => {
  test("simple y = 2x + 1 (exact fit)", () => {
    const X = [1, 2, 3, 4, 5];
    const y = X.map((x) => 2 * x + 1);
    const model = new LinearRegression();
    model.fit(X, y);
    expect(model.intercept).toBeCloseTo(1, 8);
    expect(model.coefficients[0]).toBeCloseTo(2, 8);
  });

  test("predict on new data", () => {
    const X = [1, 2, 3, 4, 5];
    const y = X.map((x) => 2 * x + 1);
    const model = new LinearRegression();
    model.fit(X, y);
    const pred = model.predict([10]);
    expect(pred[0]).toBeCloseTo(21, 8);
  });

  test("multiple regression y = 1 + 2x1 + 3x2", () => {
    const X = [
      [1, 1],
      [2, 1],
      [1, 2],
      [3, 2],
      [2, 3],
    ];
    const y = X.map(([x1, x2]) => 1 + 2 * x1! + 3 * x2!);
    const model = new LinearRegression();
    model.fit(X, y);
    expect(model.intercept).toBeCloseTo(1, 6);
    expect(model.coefficients[0]).toBeCloseTo(2, 6);
    expect(model.coefficients[1]).toBeCloseTo(3, 6);
  });

  test("fitIntercept=false (regression through origin)", () => {
    const X = [1, 2, 3, 4, 5];
    const y = X.map((x) => 3 * x);
    const model = new LinearRegression({ fitIntercept: false });
    model.fit(X, y);
    expect(model.intercept).toBe(0);
    expect(model.coefficients[0]).toBeCloseTo(3, 8);
  });

  test("R² = 1 for perfect fit", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x + 1);
    const model = new LinearRegression();
    model.fit(X, y);
    const stats = model.statistics();
    expect(stats.rSquared).toBeCloseTo(1, 8);
  });

  test("statistics p-values for significant relationship", () => {
    // y = 2x + noise, x is significant
    const X = Array.from({ length: 50 }, (_, i) => i);
    const y = X.map((x) => 2 * x + 5 + Math.sin(x) * 2); // deterministic "noise"
    const model = new LinearRegression();
    model.fit(X, y);
    const stats = model.statistics();
    expect(stats.rSquared).toBeGreaterThan(0.9);
    // The slope should be highly significant
    expect(stats.pValues[1]).toBeLessThan(0.05);
  });

  test("residuals sum to approximately zero", () => {
    const X = [1, 2, 3, 4, 5];
    const y = [2.1, 3.9, 6.2, 7.8, 10.1];
    const model = new LinearRegression();
    model.fit(X, y);
    const res = model.residuals();
    const sum = res.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum)).toBeLessThan(1e-10);
  });

  test("summary returns formatted string", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x + 1);
    const model = new LinearRegression();
    model.fit(X, y);
    const s = model.summary();
    expect(s).toContain("Coefficients:");
    expect(s).toContain("R-squared:");
  });

  test("throws if not fitted", () => {
    const model = new LinearRegression();
    expect(() => model.coefficients).toThrow("not been fitted");
    expect(() => model.statistics()).toThrow("not been fitted");
  });
});

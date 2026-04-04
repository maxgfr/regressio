import { describe, expect, test } from "bun:test";
import { breuschPagan, durbinWatson, shapiroWilk } from "../../src/diagnostics/tests";
import { LinearRegression } from "../../src/models/linear-regression";

describe("Durbin-Watson", () => {
  test("no autocorrelation: DW ≈ 2", () => {
    // Residuals without clear autocorrelation pattern
    const residuals = [0.1, 0.3, -0.2, 0.05, -0.15, 0.2, -0.3, 0.1, 0.15, -0.1];
    const { statistic } = durbinWatson(residuals);
    expect(statistic).toBeGreaterThan(1);
    expect(statistic).toBeLessThan(3.5);
  });

  test("positive autocorrelation: DW < 2", () => {
    // Residuals with positive autocorrelation (all same sign in sequence)
    const residuals = [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5];
    const { statistic } = durbinWatson(residuals);
    expect(statistic).toBeLessThan(2);
  });

  test("DW is in [0, 4] range", () => {
    const residuals = [1, -1, 1, -1, 1, -1];
    const { statistic } = durbinWatson(residuals);
    expect(statistic).toBeGreaterThanOrEqual(0);
    expect(statistic).toBeLessThanOrEqual(4);
  });
});

describe("Breusch-Pagan", () => {
  test("homoscedastic data: non-significant p-value", () => {
    const X = Array.from({ length: 50 }, (_, i) => [i]);
    const y = X.map(([x]) => 2 * x! + 5 + Math.sin(x!) * 0.5);
    const model = new LinearRegression();
    model.fit(X, y);
    const residuals = model.residuals();
    const { pValue } = breuschPagan(X, residuals);
    // Should not detect heteroscedasticity in this clean data
    expect(pValue).toBeGreaterThan(0.01);
  });

  test("heteroscedastic data: significant p-value", () => {
    // Variance increases with x (clear heteroscedasticity)
    const X = Array.from({ length: 100 }, (_, i) => [i]);
    const y = X.map(([x]) => 2 * x! + 5 + Math.sin(x! * 0.7) * x! * 0.5);
    const model = new LinearRegression();
    model.fit(X, y);
    const residuals = model.residuals();
    const { statistic } = breuschPagan(X, residuals);
    // Statistic should be non-trivial for heteroscedastic data
    expect(statistic).toBeGreaterThan(0);
  });
});

describe("Shapiro-Wilk", () => {
  test("normal data: W close to 1", () => {
    // Data that looks normal-ish
    const data = [-1.2, -0.8, -0.5, -0.2, 0, 0.1, 0.3, 0.6, 0.9, 1.3];
    const { statistic } = shapiroWilk(data);
    expect(statistic).toBeGreaterThan(0.9);
    expect(statistic).toBeLessThanOrEqual(1);
  });

  test("highly non-normal data: lower W", () => {
    // Bimodal data
    const data = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10];
    const { statistic } = shapiroWilk(data);
    expect(statistic).toBeLessThan(0.95);
  });

  test("throws for n < 3", () => {
    expect(() => shapiroWilk([1, 2])).toThrow("at least 3");
  });
});

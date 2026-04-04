import { describe, expect, test } from "bun:test";
import { LinearRegression } from "../../src/models/linear-regression";
import { RobustRegression } from "../../src/models/robust-regression";

describe("RobustRegression", () => {
  test("without outliers, similar to OLS", () => {
    const X = Array.from({ length: 20 }, (_, i) => i + 1);
    const y = X.map((x) => 2 * x + 5);

    const ols = new LinearRegression();
    ols.fit(X, y);

    const robust = new RobustRegression({ method: "huber" });
    robust.fit(X, y);

    expect(robust.coefficients[0]).toBeCloseTo(ols.coefficients[0]!, 2);
    expect(robust.intercept).toBeCloseTo(ols.intercept, 1);
  });

  test("with outliers, more resistant than OLS (Huber)", () => {
    const X = Array.from({ length: 20 }, (_, i) => i + 1);
    const y = X.map((x) => 2 * x + 5);
    // Add extreme outliers
    y[18] = 500;
    y[19] = -200;

    const ols = new LinearRegression();
    ols.fit(X, y);

    const robust = new RobustRegression({ method: "huber" });
    robust.fit(X, y);

    // Robust slope should be closer to the true slope (2)
    expect(Math.abs(robust.coefficients[0]! - 2)).toBeLessThan(Math.abs(ols.coefficients[0]! - 2));
  });

  test("Tukey bisquare works", () => {
    const X = Array.from({ length: 20 }, (_, i) => i + 1);
    const y = X.map((x) => 3 * x + 1);
    y[0] = 1000; // outlier

    const robust = new RobustRegression({ method: "tukey" });
    robust.fit(X, y);

    expect(robust.coefficients[0]).toBeCloseTo(3, 0);
  });

  test("prediction works", () => {
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = X.map((x) => 2 * x);
    const model = new RobustRegression();
    model.fit(X, y);
    const pred = model.predict([5]);
    expect(pred[0]).toBeCloseTo(10, 0);
  });
});

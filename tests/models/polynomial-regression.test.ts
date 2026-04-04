import { describe, expect, test } from "bun:test";
import { PolynomialRegression } from "../../src/models/polynomial-regression";

describe("PolynomialRegression", () => {
  test("fits y = x² exactly (degree 2)", () => {
    const X = [-2, -1, 0, 1, 2, 3];
    const y = X.map((x) => x * x);
    const model = new PolynomialRegression({ degree: 2 });
    model.fit(X, y);
    expect(model.intercept).toBeCloseTo(0, 6);
    expect(model.coefficients[0]).toBeCloseTo(0, 6); // x coeff
    expect(model.coefficients[1]).toBeCloseTo(1, 6); // x² coeff
  });

  test("fits y = 3x² + 2x + 1", () => {
    const X = [-3, -2, -1, 0, 1, 2, 3];
    const y = X.map((x) => 3 * x * x + 2 * x + 1);
    const model = new PolynomialRegression({ degree: 2 });
    model.fit(X, y);
    expect(model.intercept).toBeCloseTo(1, 6);
    expect(model.coefficients[0]).toBeCloseTo(2, 6);
    expect(model.coefficients[1]).toBeCloseTo(3, 6);
  });

  test("degree 3 fits cubic", () => {
    const X = [-2, -1, 0, 1, 2, 3, 4];
    const y = X.map((x) => x * x * x - 2 * x + 1);
    const model = new PolynomialRegression({ degree: 3 });
    model.fit(X, y);
    expect(model.intercept).toBeCloseTo(1, 4);
    expect(model.coefficients[0]).toBeCloseTo(-2, 4); // x
    expect(model.coefficients[1]).toBeCloseTo(0, 4); // x²
    expect(model.coefficients[2]).toBeCloseTo(1, 4); // x³
  });

  test("predict at new points", () => {
    const X = [0, 1, 2, 3, 4];
    const y = X.map((x) => x * x);
    const model = new PolynomialRegression({ degree: 2 });
    model.fit(X, y);
    const pred = model.predict([5]);
    expect(pred[0]).toBeCloseTo(25, 2);
  });

  test("rejects multi-feature input", () => {
    const X = [
      [1, 2],
      [3, 4],
    ];
    const y = [1, 2];
    const model = new PolynomialRegression();
    expect(() => model.fit(X, y)).toThrow("single feature");
  });

  test("default degree is 2", () => {
    const X = [1, 2, 3, 4, 5];
    const y = [1, 4, 9, 16, 25];
    const model = new PolynomialRegression();
    model.fit(X, y);
    expect(model.coefficients.length).toBe(2);
  });
});

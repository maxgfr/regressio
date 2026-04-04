import { describe, expect, test } from "bun:test";
import {
  cooksDistance,
  leverage,
  residualDiagnostics,
  studentizedResiduals,
} from "../../src/diagnostics/residuals";
import { LinearRegression } from "../../src/models/linear-regression";

describe("Residual Diagnostics", () => {
  const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1];
  const model = new LinearRegression();
  model.fit(X, y);
  const yHat = model.predict(X);

  test("raw residuals sum to ~0", () => {
    const diag = residualDiagnostics(X, y, yHat);
    const sum = diag.raw.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum)).toBeLessThan(1e-10);
  });

  test("leverage values between 0 and 1", () => {
    const diag = residualDiagnostics(X, y, yHat);
    for (const h of diag.leverage) {
      expect(h).toBeGreaterThanOrEqual(0);
      expect(h).toBeLessThanOrEqual(1);
    }
  });

  test("leverage sums to k (number of parameters)", () => {
    const diag = residualDiagnostics(X, y, yHat);
    const sum = diag.leverage.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(2, 6); // intercept + 1 feature = 2
  });

  test("Cook's distances are non-negative", () => {
    const diag = residualDiagnostics(X, y, yHat);
    for (const d of diag.cooksDistance) {
      expect(d).toBeGreaterThanOrEqual(0);
    }
  });

  test("studentized residuals have mean ~0", () => {
    const diag = residualDiagnostics(X, y, yHat);
    const mean = diag.studentized.reduce((a, b) => a + b, 0) / diag.studentized.length;
    expect(Math.abs(mean)).toBeLessThan(1);
  });

  test("leverage function works standalone", () => {
    const lev = leverage(X);
    expect(lev.length).toBe(10);
    expect(lev.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 6);
  });

  test("studentizedResiduals standalone function", () => {
    const sr = studentizedResiduals(X, y, yHat);
    expect(sr.length).toBe(10);
    // Should match residualDiagnostics result
    const diag = residualDiagnostics(X, y, yHat);
    for (let i = 0; i < sr.length; i++) {
      expect(sr[i]).toBeCloseTo(diag.studentized[i]!, 10);
    }
  });

  test("cooksDistance standalone function", () => {
    const cd = cooksDistance(X, y, yHat);
    expect(cd.length).toBe(10);
    for (const d of cd) {
      expect(d).toBeGreaterThanOrEqual(0);
    }
  });
});

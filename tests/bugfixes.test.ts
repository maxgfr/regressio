import { describe, expect, test } from "bun:test";
import { eigenvalues } from "../src/core/decompositions";
import { Matrix } from "../src/core/matrix";
import { correlationMatrix, vif } from "../src/diagnostics/multicollinearity";
import { shapiroWilk } from "../src/diagnostics/tests";
import { LogisticRegression } from "../src/models/logistic-regression";
import { RobustRegression } from "../src/models/robust-regression";
import { bootstrapCoefficients } from "../src/predictions/bootstrap";

// -----------------------------------------------------------------------
// Bug fix: shapiroWilk require() → ES import
// -----------------------------------------------------------------------
describe("shapiroWilk ES import fix", () => {
  test("shapiroWilk works after require→import migration", () => {
    const data = [-1.5, -1.0, -0.5, 0, 0.2, 0.5, 0.8, 1.1, 1.4, 1.8];
    const result = shapiroWilk(data);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  test("shapiroWilk returns valid p-value for large sample", () => {
    // Generate approximately normal data (Box-Muller)
    const data: number[] = [];
    for (let i = 0; i < 50; i++) {
      data.push(Math.sin(i) * 2 + Math.cos(i * 3) * 0.5);
    }
    const result = shapiroWilk(data);
    expect(Number.isFinite(result.statistic)).toBe(true);
    expect(Number.isFinite(result.pValue)).toBe(true);
  });
});

// -----------------------------------------------------------------------
// Bug fix: RobustRegression MAD calculation
// -----------------------------------------------------------------------
describe("RobustRegression MAD fix", () => {
  test("MAD uses median centering, not zero centering", () => {
    // With asymmetric outliers, the old MAD-from-zero gave different results
    const X = Array.from({ length: 30 }, (_, i) => i + 1);
    const y = X.map((x) => 2 * x + 10);
    // Add a few large outliers on one side
    y[27] = 500;
    y[28] = 600;
    y[29] = 700;

    const robust = new RobustRegression({ method: "huber" });
    robust.fit(X, y);

    // True slope is 2, robust should be close
    expect(Math.abs(robust.coefficients[0]! - 2)).toBeLessThan(1);
  });

  test("Tukey bisquare still works after MAD fix", () => {
    const X = Array.from({ length: 25 }, (_, i) => i + 1);
    const y = X.map((x) => 5 * x + 3);
    y[0] = 999; // outlier
    y[24] = -999; // outlier

    const robust = new RobustRegression({ method: "tukey" });
    robust.fit(X, y);

    expect(robust.coefficients[0]).toBeCloseTo(5, 0);
  });
});

// -----------------------------------------------------------------------
// Bug fix: LogisticRegression._X removed
// -----------------------------------------------------------------------
describe("LogisticRegression undeclared _X fix", () => {
  test("logistic regression fit works without _X property", () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
    const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);

    expect(model.coefficients.length).toBe(1);
    expect(Number.isFinite(model.intercept)).toBe(true);

    const predictions = model.predict(X);
    expect(predictions.length).toBe(10);
  });

  test("logistic statistics still work", () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
    const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const model = new LogisticRegression();
    model.fit(X, y);
    const stats = model.statistics();

    expect(stats.accuracy).toBeGreaterThanOrEqual(0);
    expect(stats.accuracy).toBeLessThanOrEqual(1);
    expect(Number.isFinite(stats.aic)).toBe(true);
    expect(Number.isFinite(stats.logLikelihood)).toBe(true);
  });
});

// -----------------------------------------------------------------------
// Eigenvalue tridiagonal algorithm
// -----------------------------------------------------------------------
describe("eigenvalues tridiagonal improvement", () => {
  test("4x4 symmetric matrix eigenvalues", () => {
    const A = Matrix.fromArray([
      [4, 1, 0, 0],
      [1, 3, 1, 0],
      [0, 1, 2, 1],
      [0, 0, 1, 1],
    ]);
    const eigs = eigenvalues(A);
    expect(eigs.length).toBe(4);
    // Eigenvalues should sum to trace = 10
    const sum = eigs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(10, 6);
    // Sorted descending
    for (let i = 0; i < eigs.length - 1; i++) {
      expect(eigs[i]).toBeGreaterThanOrEqual(eigs[i + 1]! - 1e-8);
    }
  });

  test("5x5 symmetric matrix eigenvalues", () => {
    const A = Matrix.fromArray([
      [5, 2, 0, 0, 0],
      [2, 4, 1, 0, 0],
      [0, 1, 3, 1, 0],
      [0, 0, 1, 2, 1],
      [0, 0, 0, 1, 1],
    ]);
    const eigs = eigenvalues(A);
    expect(eigs.length).toBe(5);
    const sum = eigs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(15, 6); // trace = 5+4+3+2+1 = 15
    // All eigenvalues should be positive for this PD matrix
    for (const e of eigs) {
      expect(e).toBeGreaterThan(-1e-8);
    }
  });

  test("eigenvalues of known 3x3 matrix", () => {
    // [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] => eigenvalues 2-√2, 2, 2+√2
    const A = Matrix.fromArray([
      [2, -1, 0],
      [-1, 2, -1],
      [0, -1, 2],
    ]);
    const eigs = eigenvalues(A);
    expect(eigs[0]).toBeCloseTo(2 + Math.SQRT2, 6);
    expect(eigs[1]).toBeCloseTo(2, 6);
    expect(eigs[2]).toBeCloseTo(2 - Math.SQRT2, 6);
  });
});

// -----------------------------------------------------------------------
// VIF optimization (correlation matrix inverse diagonal)
// -----------------------------------------------------------------------
describe("VIF optimization", () => {
  test("VIF with nearly orthogonal features is low", () => {
    // Use actual orthogonal columns
    const X = [
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [2, 0, 0],
      [0, 2, 1],
      [1, 0, 0],
      [0, 1, 1],
    ];
    const vifs = vif(X);
    expect(vifs.length).toBe(3);
    for (const v of vifs) {
      expect(v).toBeLessThan(10);
    }
  });

  test("VIF with 2 correlated + 1 independent feature", () => {
    const n = 50;
    const X = Array.from({ length: n }, (_, i) => [
      i,
      i * 2 + 0.001 * Math.sin(i), // highly correlated with first
      Math.sin(i * 3), // independent
    ]);
    const vifs = vif(X);
    // First two should have high VIF, third should be low
    expect(vifs[0]).toBeGreaterThan(5);
    expect(vifs[1]).toBeGreaterThan(5);
    expect(vifs[2]).toBeLessThan(5);
  });
});

// -----------------------------------------------------------------------
// Bootstrap with aggregation
// -----------------------------------------------------------------------
describe("bootstrap aggregation", () => {
  test("bootstrap with multiple features", () => {
    const X = Array.from({ length: 30 }, (_, i) => [i, i * 2 + 1]);
    const y = X.map(([a, b]) => 3 * a! + 2 * b! + 5);

    const result = bootstrapCoefficients(X, y, 200);
    // intercept + 2 features = 3 params
    expect(result.coefficients.length).toBe(3);
    expect(result.standardErrors.length).toBe(3);
    expect(result.confidenceIntervals.length).toBe(3);

    // All SEs should be finite and non-negative
    for (const se of result.standardErrors) {
      expect(Number.isFinite(se)).toBe(true);
      expect(se).toBeGreaterThanOrEqual(0);
    }
  });
});

// -----------------------------------------------------------------------
// Correlation matrix
// -----------------------------------------------------------------------
describe("correlation matrix", () => {
  test("3 features: correct shape and diagonal", () => {
    const X = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ];
    const corr = correlationMatrix(X);
    expect(corr.length).toBe(3);
    expect(corr[0]!.length).toBe(3);
    // Diagonal = 1
    for (let i = 0; i < 3; i++) {
      expect(corr[i]![i]).toBeCloseTo(1, 10);
    }
  });

  test("negatively correlated features", () => {
    const X = [
      [1, 10],
      [2, 8],
      [3, 6],
      [4, 4],
      [5, 2],
    ];
    const corr = correlationMatrix(X);
    expect(corr[0]![1]).toBeCloseTo(-1, 6);
  });
});

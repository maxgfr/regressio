import { describe, expect, test } from "bun:test";
import { bootstrapCoefficients } from "../../src/predictions/bootstrap";

describe("bootstrapCoefficients", () => {
  const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1];

  test("returns correct number of coefficients", () => {
    const result = bootstrapCoefficients(X, y, 100);
    // intercept + 1 feature = 2 coefficients
    expect(result.coefficients.length).toBe(2);
    expect(result.standardErrors.length).toBe(2);
    expect(result.confidenceIntervals.length).toBe(2);
  });

  test("bootstrap slope CI contains true slope (~2)", () => {
    const result = bootstrapCoefficients(X, y, 500);
    // Slope is the second coefficient (index 1)
    const [lower, upper] = result.confidenceIntervals[1]!;
    expect(lower).toBeLessThan(2.1);
    expect(upper).toBeGreaterThan(1.9);
  });

  test("standard errors are positive", () => {
    const result = bootstrapCoefficients(X, y, 100);
    for (const se of result.standardErrors) {
      expect(se).toBeGreaterThan(0);
    }
  });

  test("confidence intervals have lower < upper", () => {
    const result = bootstrapCoefficients(X, y, 100);
    for (const [lower, upper] of result.confidenceIntervals) {
      expect(lower).toBeLessThan(upper);
    }
  });

  test("fitIntercept=false returns p coefficients", () => {
    const result = bootstrapCoefficients(X, y, 50, 0.05, false);
    expect(result.coefficients.length).toBe(1); // just the slope
  });
});

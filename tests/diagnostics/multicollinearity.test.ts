import { describe, expect, test } from "bun:test";
import { conditionNumber, correlationMatrix, vif } from "../../src/diagnostics/multicollinearity";

describe("VIF", () => {
  test("orthogonal features: VIF ≈ 1", () => {
    // Nearly orthogonal features
    const X = [
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
      [2, 0],
      [0, 2],
    ];
    const vifs = vif(X);
    for (const v of vifs) {
      expect(v).toBeLessThan(2);
    }
  });

  test("highly correlated features: high VIF", () => {
    // x2 ≈ x1 (almost perfect correlation)
    const X = Array.from({ length: 20 }, (_, i) => [i, i + Math.random() * 0.01]);
    const vifs = vif(X);
    expect(vifs[0]).toBeGreaterThan(10);
    expect(vifs[1]).toBeGreaterThan(10);
  });

  test("single feature returns VIF = 1", () => {
    const X = [[1], [2], [3]];
    expect(vif(X)).toEqual([1]);
  });
});

describe("Correlation Matrix", () => {
  test("diagonal is all 1s", () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const corr = correlationMatrix(X);
    expect(corr[0]![0]).toBeCloseTo(1, 10);
    expect(corr[1]![1]).toBeCloseTo(1, 10);
  });

  test("perfectly correlated: r = 1", () => {
    const X = [
      [1, 2],
      [2, 4],
      [3, 6],
    ];
    const corr = correlationMatrix(X);
    expect(corr[0]![1]).toBeCloseTo(1, 8);
    expect(corr[1]![0]).toBeCloseTo(1, 8);
  });

  test("symmetric matrix", () => {
    const X = [
      [1, 5],
      [2, 3],
      [3, 7],
      [4, 2],
    ];
    const corr = correlationMatrix(X);
    expect(corr[0]![1]).toBeCloseTo(corr[1]![0]!, 10);
  });
});

describe("Condition Number", () => {
  test("identity-like matrix: low condition number", () => {
    const X = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const kappa = conditionNumber(X);
    expect(kappa).toBeLessThan(10);
  });

  test("nearly singular matrix: high condition number", () => {
    const X = [
      [1, 1.0001],
      [2, 2.0002],
      [3, 3.0003],
    ];
    const kappa = conditionNumber(X);
    expect(kappa).toBeGreaterThan(100);
  });
});

import { describe, expect, test } from "bun:test";
import { interactionFeatures, polynomialFeatures } from "../../src/preprocessing/features";

describe("polynomialFeatures", () => {
  test("degree 2 expands correctly", () => {
    const X = [[2], [3]];
    const result = polynomialFeatures(X, 2);
    expect(result).toEqual([
      [2, 4], // x, x²
      [3, 9],
    ]);
  });

  test("degree 3 with multiple features", () => {
    const X = [[2, 3]];
    const result = polynomialFeatures(X, 3);
    // For each feature: x, x², x³
    expect(result).toEqual([[2, 4, 8, 3, 9, 27]]);
  });
});

describe("interactionFeatures", () => {
  test("all pairs interaction", () => {
    const X = [[1, 2, 3]];
    const result = interactionFeatures(X);
    // Pairs: (0,1), (0,2), (1,2) → 1*2, 1*3, 2*3
    expect(result).toEqual([[2, 3, 6]]);
  });

  test("specified pairs only", () => {
    const X = [[2, 3, 4]];
    const result = interactionFeatures(X, [[0, 2]]);
    expect(result).toEqual([[8]]); // 2 * 4
  });

  test("multiple rows", () => {
    const X = [
      [1, 2],
      [3, 4],
    ];
    const result = interactionFeatures(X);
    expect(result).toEqual([
      [2], // 1*2
      [12], // 3*4
    ]);
  });
});

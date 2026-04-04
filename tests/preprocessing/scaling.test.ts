import { describe, expect, test } from "bun:test";
import {
  normalize,
  standardize,
  unnormalize,
  unstandardize,
} from "../../src/preprocessing/scaling";

describe("Standardize", () => {
  test("mean ≈ 0 and std ≈ 1 after standardization", () => {
    const X = [
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
      [5, 50],
    ];
    const { transformed } = standardize(X);

    for (let j = 0; j < 2; j++) {
      let mean = 0;
      for (const row of transformed) mean += row[j]!;
      mean /= transformed.length;
      expect(Math.abs(mean)).toBeLessThan(1e-10);

      let variance = 0;
      for (const row of transformed) variance += (row[j]! - mean) ** 2;
      variance /= transformed.length;
      expect(variance).toBeCloseTo(1, 6);
    }
  });

  test("round-trip: unstandardize(standardize(X)) ≈ X", () => {
    const X = [
      [1, 10],
      [2, 20],
      [3, 30],
    ];
    const { transformed, means, stds } = standardize(X);
    const restored = unstandardize(transformed, { means, stds });
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[0]!.length; j++) {
        expect(restored[i]![j]).toBeCloseTo(X[i]![j]!, 10);
      }
    }
  });
});

describe("Normalize", () => {
  test("values between 0 and 1 after normalization", () => {
    const X = [
      [1, 100],
      [5, 200],
      [3, 150],
    ];
    const { transformed } = normalize(X);
    for (const row of transformed) {
      for (const val of row) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    }
  });

  test("min maps to 0, max maps to 1", () => {
    const X = [
      [2, 10],
      [5, 30],
      [8, 20],
    ];
    const { transformed } = normalize(X);
    // Column 0: min=2, max=8
    expect(transformed[0]![0]).toBeCloseTo(0, 10);
    expect(transformed[1]![0]).toBeCloseTo(0.5, 10);
    expect(transformed[2]![0]).toBeCloseTo(1, 10);
  });

  test("round-trip: unnormalize(normalize(X)) ≈ X", () => {
    const X = [
      [1, 10],
      [5, 50],
      [3, 30],
    ];
    const { transformed, mins, maxs } = normalize(X);
    const restored = unnormalize(transformed, { mins, maxs });
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[0]!.length; j++) {
        expect(restored[i]![j]).toBeCloseTo(X[i]![j]!, 10);
      }
    }
  });
});

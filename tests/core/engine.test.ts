import { describe, expect, test } from "bun:test";
import {
  engineAdd,
  engineCholesky,
  engineCoordinateDescent,
  engineDeterminant,
  engineDot,
  engineEigenvalues,
  engineEuclideanDistances,
  engineForwardSubstitution,
  engineManhattanDistances,
  engineMatrixMultiply,
  engineNorm,
  engineQR,
  engineScale,
  engineSoftmaxRows,
  engineSolveTriangular,
  engineSubtract,
  engineSVD,
  engineTranspose,
  isWasmActive,
} from "../../src/core/engine";

/**
 * Engine tests: verify that dispatch functions work correctly.
 * In the test environment, WASM is not loaded (bundler target), so all
 * dispatch functions exercise the TypeScript fallback (returning null).
 * engineMatrixMultiply has a built-in TS fallback and always returns a result.
 */

describe("isWasmActive", () => {
  test("returns false when WASM is not loaded", () => {
    expect(isWasmActive()).toBe(false);
  });
});

describe("engineMatrixMultiply (always has TS fallback)", () => {
  test("multiplies 2x2 matrices", () => {
    const a = new Float64Array([1, 2, 3, 4]);
    const b = new Float64Array([5, 6, 7, 8]);
    const result = engineMatrixMultiply(a, 2, 2, b, 2, 2);
    expect(result[0]).toBe(19); // 1*5 + 2*7
    expect(result[1]).toBe(22); // 1*6 + 2*8
    expect(result[2]).toBe(43); // 3*5 + 4*7
    expect(result[3]).toBe(50); // 3*6 + 4*8
  });

  test("multiplies non-square matrices", () => {
    // (2x3) * (3x1) = (2x1)
    const a = new Float64Array([1, 2, 3, 4, 5, 6]);
    const b = new Float64Array([1, 1, 1]);
    const result = engineMatrixMultiply(a, 2, 3, b, 3, 1);
    expect(result[0]).toBe(6); // 1+2+3
    expect(result[1]).toBe(15); // 4+5+6
  });
});

describe("engine dispatch returns null without WASM", () => {
  test("engineTranspose returns null", () => {
    expect(engineTranspose(new Float64Array([1, 2, 3, 4]), 2, 2)).toBeNull();
  });

  test("engineAdd returns null", () => {
    expect(engineAdd(new Float64Array([1]), new Float64Array([2]))).toBeNull();
  });

  test("engineSubtract returns null", () => {
    expect(engineSubtract(new Float64Array([1]), new Float64Array([2]))).toBeNull();
  });

  test("engineScale returns null", () => {
    expect(engineScale(new Float64Array([1]), 2)).toBeNull();
  });

  test("engineDot returns null", () => {
    expect(engineDot(new Float64Array([1]), new Float64Array([2]))).toBeNull();
  });

  test("engineNorm returns null", () => {
    expect(engineNorm(new Float64Array([3, 4]))).toBeNull();
  });

  test("engineDeterminant returns null", () => {
    expect(engineDeterminant(new Float64Array([1, 0, 0, 1]), 2)).toBeNull();
  });

  test("engineQR returns null", () => {
    expect(engineQR(new Float64Array([1, 0, 0, 1]), 2, 2)).toBeNull();
  });

  test("engineCholesky returns null", () => {
    expect(engineCholesky(new Float64Array([4, 2, 2, 5]), 2)).toBeNull();
  });

  test("engineSolveTriangular returns null", () => {
    expect(
      engineSolveTriangular(new Float64Array([1, 0, 0, 1]), new Float64Array([1, 1]), 2),
    ).toBeNull();
  });

  test("engineForwardSubstitution returns null", () => {
    expect(
      engineForwardSubstitution(new Float64Array([1, 0, 0, 1]), new Float64Array([1, 1]), 2),
    ).toBeNull();
  });

  test("engineSVD returns null", () => {
    expect(engineSVD(new Float64Array([1, 0, 0, 1]), 2, 2)).toBeNull();
  });

  test("engineEigenvalues returns null", () => {
    expect(engineEigenvalues(new Float64Array([2, 1, 1, 2]), 2)).toBeNull();
  });

  test("engineCoordinateDescent returns null", () => {
    expect(
      engineCoordinateDescent(
        new Float64Array([1, 2, 3, 4]),
        new Float64Array([1, 2]),
        0.1,
        1,
        100,
        1e-4,
        2,
        2,
        true,
      ),
    ).toBeNull();
  });

  test("engineSoftmaxRows returns null", () => {
    expect(engineSoftmaxRows(new Float64Array([1, 2, 3, 4]), 2, 2)).toBeNull();
  });

  test("engineEuclideanDistances returns null", () => {
    expect(
      engineEuclideanDistances(new Float64Array([0, 0]), new Float64Array([1, 1]), 1, 1, 2),
    ).toBeNull();
  });

  test("engineManhattanDistances returns null", () => {
    expect(
      engineManhattanDistances(new Float64Array([0, 0]), new Float64Array([1, 1]), 1, 1, 2),
    ).toBeNull();
  });
});

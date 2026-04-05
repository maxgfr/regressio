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
 * Engine tests: verify that dispatch functions return correct results.
 * WASM may or may not be loaded depending on the environment.
 * engineMatrixMultiply always returns a result (has TS fallback inline).
 */

describe("isWasmActive", () => {
  test("returns a boolean", () => {
    expect(typeof isWasmActive()).toBe("boolean");
  });
});

describe("engineMatrixMultiply (always has TS fallback)", () => {
  test("multiplies 2x2 matrices", () => {
    const a = new Float64Array([1, 2, 3, 4]);
    const b = new Float64Array([5, 6, 7, 8]);
    const result = engineMatrixMultiply(a, 2, 2, b, 2, 2);
    expect(result[0]).toBe(19);
    expect(result[1]).toBe(22);
    expect(result[2]).toBe(43);
    expect(result[3]).toBe(50);
  });

  test("multiplies non-square matrices", () => {
    const a = new Float64Array([1, 2, 3, 4, 5, 6]);
    const b = new Float64Array([1, 1, 1]);
    const result = engineMatrixMultiply(a, 2, 3, b, 3, 1);
    expect(result[0]).toBe(6);
    expect(result[1]).toBe(15);
  });
});

describe("engine dispatch functions", () => {
  test("engineTranspose returns correct result or null", () => {
    const result = engineTranspose(new Float64Array([1, 2, 3, 4]), 2, 2);
    if (result) {
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(3);
      expect(result[2]).toBe(2);
      expect(result[3]).toBe(4);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineAdd returns correct result or null", () => {
    const result = engineAdd(new Float64Array([1, 2]), new Float64Array([3, 4]));
    if (result) {
      expect(result[0]).toBe(4);
      expect(result[1]).toBe(6);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineSubtract returns correct result or null", () => {
    const result = engineSubtract(new Float64Array([5]), new Float64Array([3]));
    if (result) {
      expect(result[0]).toBe(2);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineScale returns correct result or null", () => {
    const result = engineScale(new Float64Array([2, 3]), 4);
    if (result) {
      expect(result[0]).toBe(8);
      expect(result[1]).toBe(12);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineDot returns correct result or null", () => {
    const result = engineDot(new Float64Array([1, 2]), new Float64Array([3, 4]));
    if (result !== null) {
      expect(result).toBe(11);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineNorm returns correct result or null", () => {
    const result = engineNorm(new Float64Array([3, 4]));
    if (result !== null) {
      expect(result).toBe(5);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineDeterminant returns correct result or null", () => {
    const result = engineDeterminant(new Float64Array([1, 2, 3, 4]), 2);
    if (result !== null) {
      expect(result).toBeCloseTo(-2, 10);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineQR returns result or null", () => {
    const result = engineQR(new Float64Array([1, 0, 0, 1]), 2, 2);
    if (result) {
      expect(result.Q.length).toBe(4);
      expect(result.R.length).toBe(4);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineCholesky returns result or null", () => {
    const result = engineCholesky(new Float64Array([4, 2, 2, 5]), 2);
    if (result) {
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(2, 10);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineSolveTriangular returns result or null", () => {
    const result = engineSolveTriangular(
      new Float64Array([1, 0, 0, 1]),
      new Float64Array([1, 1]),
      2,
    );
    if (result) {
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(1);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineForwardSubstitution returns result or null", () => {
    const result = engineForwardSubstitution(
      new Float64Array([1, 0, 0, 1]),
      new Float64Array([1, 1]),
      2,
    );
    if (result) {
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(1);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineSVD returns result or null", () => {
    const result = engineSVD(new Float64Array([1, 0, 0, 1]), 2, 2);
    if (result) {
      expect(result.U.length).toBe(4);
      expect(result.S.length).toBe(2);
      expect(result.V.length).toBe(4);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineEigenvalues returns result or null", () => {
    const result = engineEigenvalues(new Float64Array([2, 1, 1, 2]), 2);
    if (result) {
      expect(result.length).toBe(2);
      expect(result[0]).toBeCloseTo(3, 6);
      expect(result[1]).toBeCloseTo(1, 6);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineCoordinateDescent returns result or null", () => {
    const result = engineCoordinateDescent(
      new Float64Array([1, 2, 3, 4]),
      new Float64Array([1, 2]),
      0.1,
      1,
      100,
      1e-4,
      2,
      2,
      true,
    );
    if (result) {
      expect(result.length).toBe(3); // intercept + 2 coefficients
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineSoftmaxRows returns result or null", () => {
    const result = engineSoftmaxRows(new Float64Array([1, 2, 3, 4]), 2, 2);
    if (result) {
      expect(result.length).toBe(4);
      // Each row should sum to ~1
      expect(result[0]! + result[1]!).toBeCloseTo(1, 10);
      expect(result[2]! + result[3]!).toBeCloseTo(1, 10);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineEuclideanDistances returns result or null", () => {
    const result = engineEuclideanDistances(
      new Float64Array([0, 0]),
      new Float64Array([1, 1]),
      1,
      1,
      2,
    );
    if (result) {
      expect(result[0]).toBeCloseTo(Math.SQRT2, 10);
    } else {
      expect(result).toBeNull();
    }
  });

  test("engineManhattanDistances returns result or null", () => {
    const result = engineManhattanDistances(
      new Float64Array([0, 0]),
      new Float64Array([1, 1]),
      1,
      1,
      2,
    );
    if (result) {
      expect(result[0]).toBe(2);
    } else {
      expect(result).toBeNull();
    }
  });
});

import { describe, expect, test } from "bun:test";
import {
  backSubstitution,
  choleskyDecomposition,
  eigenvalues,
  qrDecomposition,
  solveCholesky,
  solveQR,
  solveSVD,
  svd,
} from "../../src/core/decompositions";
import { Matrix } from "../../src/core/matrix";

const TOLERANCE = 1e-8;

function expectClose(actual: number, expected: number, tol = TOLERANCE) {
  expect(Math.abs(actual - expected)).toBeLessThan(tol);
}

describe("QR Decomposition", () => {
  test("Q is orthogonal: Q^T Q = I", () => {
    const A = Matrix.fromArray([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const { Q } = qrDecomposition(A);
    const QtQ = Q.transpose().multiply(Q);
    const I = Matrix.identity(Q.cols);
    for (let i = 0; i < QtQ.data.length; i++) {
      expectClose(QtQ.data[i]!, I.data[i]!, 1e-10);
    }
  });

  test("R is upper triangular", () => {
    const A = Matrix.fromArray([
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ]);
    const { R } = qrDecomposition(A);
    for (let i = 0; i < R.rows; i++) {
      for (let j = 0; j < Math.min(i, R.cols); j++) {
        expectClose(R.get(i, j), 0, 1e-10);
      }
    }
  });

  test("A = QR reconstruction", () => {
    const A = Matrix.fromArray([
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ]);
    const { Q, R } = qrDecomposition(A);
    const QR = Q.multiply(R);
    for (let i = 0; i < A.rows; i++) {
      for (let j = 0; j < A.cols; j++) {
        expectClose(QR.get(i, j), A.get(i, j), 1e-8);
      }
    }
  });

  test("solveQR solves overdetermined system (least squares)", () => {
    // A = [[1,1],[1,2],[1,3]], b = [1,2,2]
    // Known OLS: β₀ = 2/3, β₁ = 1/2
    const A = Matrix.fromArray([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const b = Matrix.columnVector([1, 2, 2]);
    const { Q, R } = qrDecomposition(A);
    const x = solveQR(Q, R, b);
    expectClose(x.get(0, 0), 2 / 3, 1e-10);
    expectClose(x.get(1, 0), 1 / 2, 1e-10);
  });

  test("solveQR solves exact square system", () => {
    // [2 1; 1 3] x = [5; 7] => x = [8/5; 9/5]
    const A = Matrix.fromArray([
      [2, 1],
      [1, 3],
    ]);
    const b = Matrix.columnVector([5, 7]);
    const { Q, R } = qrDecomposition(A);
    const x = solveQR(Q, R, b);
    expectClose(x.get(0, 0), 8 / 5, 1e-10);
    expectClose(x.get(1, 0), 9 / 5, 1e-10);
  });
});

describe("Back Substitution", () => {
  test("solves upper triangular system", () => {
    const R = Matrix.fromArray([
      [2, 1],
      [0, 3],
    ]);
    const b = Matrix.columnVector([5, 9]);
    const x = backSubstitution(R, b);
    expectClose(x.get(0, 0), 1); // 5 - 1*3 / 2
    expectClose(x.get(1, 0), 3);
  });
});

describe("Cholesky Decomposition", () => {
  test("L is lower triangular", () => {
    const A = Matrix.fromArray([
      [4, 2],
      [2, 5],
    ]);
    const { L } = choleskyDecomposition(A);
    expect(L.get(0, 1)).toBe(0);
    expect(L.get(0, 0)).toBeGreaterThan(0);
    expect(L.get(1, 1)).toBeGreaterThan(0);
  });

  test("A = LL^T reconstruction", () => {
    const A = Matrix.fromArray([
      [4, 12, -16],
      [12, 37, -43],
      [-16, -43, 98],
    ]);
    const { L } = choleskyDecomposition(A);
    const LLt = L.multiply(L.transpose());
    for (let i = 0; i < A.rows; i++) {
      for (let j = 0; j < A.cols; j++) {
        expectClose(LLt.get(i, j), A.get(i, j), 1e-10);
      }
    }
  });

  test("solveCholesky solves Ax = b", () => {
    // A = [4 2; 2 5], b = [14; 17] => x = [2; 3]
    // Verify: 4*2 + 2*3 = 14, 2*2 + 5*3 = 19 ... let me fix
    // [4 2; 2 5] [2; 3] = [14; 19]
    const A = Matrix.fromArray([
      [4, 2],
      [2, 5],
    ]);
    const b = Matrix.columnVector([14, 19]);
    const { L } = choleskyDecomposition(A);
    const x = solveCholesky(L, b);
    expectClose(x.get(0, 0), 2, 1e-10);
    expectClose(x.get(1, 0), 3, 1e-10);
  });

  test("throws for non-positive-definite", () => {
    const A = Matrix.fromArray([
      [1, 2],
      [2, 1],
    ]); // eigenvalues: 3 and -1
    expect(() => choleskyDecomposition(A)).toThrow("not positive-definite");
  });
});

describe("SVD", () => {
  test("A = U·diag(S)·V^T reconstruction", () => {
    const A = Matrix.fromArray([
      [3, 2, 2],
      [2, 3, -2],
    ]);
    const { U, S, V } = svd(A);
    // Reconstruct: A ≈ U * diag(S) * V^T
    const Smat = Matrix.zeros(U.cols, V.cols);
    for (let i = 0; i < S.length; i++) {
      Smat.set(i, i, S[i]!);
    }
    const reconstructed = U.multiply(Smat).multiply(V.transpose());
    for (let i = 0; i < A.rows; i++) {
      for (let j = 0; j < A.cols; j++) {
        expectClose(reconstructed.get(i, j), A.get(i, j), 1e-6);
      }
    }
  });

  test("singular values are non-negative and sorted descending", () => {
    const A = Matrix.fromArray([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const { S } = svd(A);
    for (let i = 0; i < S.length; i++) {
      expect(S[i]).toBeGreaterThanOrEqual(-1e-10);
    }
    for (let i = 0; i < S.length - 1; i++) {
      expect(S[i]).toBeGreaterThanOrEqual(S[i + 1]! - 1e-10);
    }
  });

  test("SVD detects rank: rank-1 matrix has one significant singular value", () => {
    // Rank-1: [[1,2],[2,4]]
    const A = Matrix.fromArray([
      [1, 2],
      [2, 4],
    ]);
    const { S } = svd(A);
    expect(S[0]).toBeGreaterThan(1);
    expectClose(S[1]!, 0, 1e-8);
  });

  test("solveSVD solves least squares", () => {
    // Same system as QR test
    const A = Matrix.fromArray([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const b = Matrix.columnVector([1, 2, 2]);
    const { U, S, V } = svd(A);
    const x = solveSVD(U, S, V, b);
    expectClose(x.get(0, 0), 2 / 3, 1e-6);
    expectClose(x.get(1, 0), 1 / 2, 1e-6);
  });
});

describe("Eigenvalues", () => {
  test("eigenvalues of identity are all 1", () => {
    const eigs = eigenvalues(Matrix.identity(3));
    for (const e of eigs) {
      expectClose(e, 1, 1e-8);
    }
  });

  test("eigenvalues of diagonal matrix", () => {
    const D = Matrix.diagonal([5, 3, 1]);
    const eigs = eigenvalues(D);
    expectClose(eigs[0]!, 5, 1e-8);
    expectClose(eigs[1]!, 3, 1e-8);
    expectClose(eigs[2]!, 1, 1e-8);
  });

  test("eigenvalues of symmetric 2x2", () => {
    // [[2, 1], [1, 2]] has eigenvalues 3, 1
    const A = Matrix.fromArray([
      [2, 1],
      [1, 2],
    ]);
    const eigs = eigenvalues(A);
    expectClose(eigs[0]!, 3, 1e-8);
    expectClose(eigs[1]!, 1, 1e-8);
  });

  test("eigenvalues sorted descending", () => {
    const A = Matrix.fromArray([
      [1, 0.5],
      [0.5, 2],
    ]);
    const eigs = eigenvalues(A);
    expect(eigs[0]).toBeGreaterThan(eigs[1]!);
  });
});

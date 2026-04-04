import { engineCholesky, engineQR, engineSolveTriangular, isWasmActive } from "./engine";
import { Matrix } from "./matrix";

// ---------------------------------------------------------------------------
// QR Decomposition (Householder reflections)
// ---------------------------------------------------------------------------

export interface QRResult {
  Q: Matrix;
  R: Matrix;
}

/**
 * QR decomposition via Householder reflections.
 * For m×n matrix A (m ≥ n), produces Q (m×m orthogonal) and R (m×n upper triangular)
 * such that A = Q·R.
 */
export function qrDecomposition(A: Matrix): QRResult {
  const m = A.rows;
  const n = A.cols;

  // WASM fast path
  if (isWasmActive()) {
    const wasmResult = engineQR(A.data, m, n);
    if (wasmResult) {
      return {
        Q: new Matrix(m, m, wasmResult.Q),
        R: new Matrix(m, n, wasmResult.R),
      };
    }
  }

  const R = A.clone();
  const Q = Matrix.identity(m);
  const minMN = Math.min(m, n);

  for (let k = 0; k < minMN; k++) {
    // Extract column k from row k downward
    const x = new Float64Array(m - k);
    for (let i = 0; i < m - k; i++) {
      x[i] = R.get(k + i, k);
    }

    // Compute norm of x
    let xNorm = 0;
    for (let i = 0; i < x.length; i++) {
      xNorm += x[i]! * x[i]!;
    }
    xNorm = Math.sqrt(xNorm);

    if (xNorm < 1e-15) continue;

    // alpha = -sign(x[0]) * ||x||
    const alpha = x[0]! >= 0 ? -xNorm : xNorm;

    // v = x - alpha * e1
    const v = new Float64Array(x);
    v[0] -= alpha;

    // Normalize v
    let vNorm = 0;
    for (let i = 0; i < v.length; i++) {
      vNorm += v[i]! * v[i]!;
    }
    vNorm = Math.sqrt(vNorm);
    if (vNorm < 1e-15) continue;
    for (let i = 0; i < v.length; i++) {
      v[i] /= vNorm;
    }

    // Apply reflection to R[k:, k:]: R -= 2 * v * (v^T * R)
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) {
        dot += v[i]! * R.get(k + i, j);
      }
      for (let i = 0; i < v.length; i++) {
        R.set(k + i, j, R.get(k + i, j) - 2 * v[i]! * dot);
      }
    }

    // Apply reflection to Q[:, k:]: Q -= 2 * (Q * v) * v^T
    for (let i = 0; i < m; i++) {
      let dot = 0;
      for (let j = 0; j < v.length; j++) {
        dot += Q.get(i, k + j) * v[j]!;
      }
      for (let j = 0; j < v.length; j++) {
        Q.set(i, k + j, Q.get(i, k + j) - 2 * dot * v[j]!);
      }
    }
  }

  return { Q, R };
}

/** Back-substitution for upper triangular system Rx = b. */
export function backSubstitution(R: Matrix, b: Matrix): Matrix {
  const n = R.cols;

  // WASM fast path
  if (isWasmActive()) {
    const bFlat = new Float64Array(n);
    for (let i = 0; i < n; i++) bFlat[i] = b.get(i, 0);
    const wasmResult = engineSolveTriangular(R.data, bFlat, n);
    if (wasmResult) {
      return Matrix.columnVector(Array.from(wasmResult));
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b.get(i, 0);
    for (let j = i + 1; j < n; j++) {
      sum -= R.get(i, j) * x[j]!;
    }
    const diag = R.get(i, i);
    if (Math.abs(diag) < 1e-15) {
      throw new Error("Matrix is singular or near-singular (zero diagonal in R)");
    }
    x[i] = sum / diag;
  }
  return Matrix.columnVector(Array.from(x));
}

/** Solve Ax = b via QR: x = R⁻¹(Q^T b). */
export function solveQR(Q: Matrix, R: Matrix, b: Matrix): Matrix {
  const Qtb = Q.transpose().multiply(b);
  // R is m×n, we only need the top n×n part
  const n = R.cols;
  const Rsq = R.submatrix(0, n, 0, n);
  const bTop = Qtb.submatrix(0, n, 0, 1);
  return backSubstitution(Rsq, bTop);
}

// ---------------------------------------------------------------------------
// Cholesky Decomposition
// ---------------------------------------------------------------------------

export interface CholeskyResult {
  L: Matrix;
}

/**
 * Cholesky decomposition: A = L·L^T for symmetric positive-definite A.
 * Returns lower triangular L.
 */
export function choleskyDecomposition(A: Matrix): CholeskyResult {
  if (A.rows !== A.cols) {
    throw new Error("Cholesky: matrix must be square");
  }
  const n = A.rows;

  // WASM fast path
  if (isWasmActive()) {
    const wasmResult = engineCholesky(A.data, n);
    if (wasmResult) {
      return { L: new Matrix(n, n, wasmResult) };
    }
  }

  const L = Matrix.zeros(n, n);

  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let k = 0; k < j; k++) {
      sum += L.get(j, k) * L.get(j, k);
    }
    const diag = A.get(j, j) - sum;
    if (diag <= 0) {
      throw new Error("Cholesky: matrix is not positive-definite");
    }
    L.set(j, j, Math.sqrt(diag));

    for (let i = j + 1; i < n; i++) {
      let s = 0;
      for (let k = 0; k < j; k++) {
        s += L.get(i, k) * L.get(j, k);
      }
      L.set(i, j, (A.get(i, j) - s) / L.get(j, j));
    }
  }

  return { L };
}

/** Forward substitution for lower triangular system Ly = b. */
export function forwardSubstitution(L: Matrix, b: Matrix): Matrix {
  const n = L.rows;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b.get(i, 0);
    for (let j = 0; j < i; j++) {
      sum -= L.get(i, j) * y[j]!;
    }
    y[i] = sum / L.get(i, i);
  }
  return Matrix.columnVector(Array.from(y));
}

/** Solve Ax = b via Cholesky where A = LL^T. Solves Ly = b then L^T x = y. */
export function solveCholesky(L: Matrix, b: Matrix): Matrix {
  const y = forwardSubstitution(L, b);
  return backSubstitution(L.transpose(), y);
}

// ---------------------------------------------------------------------------
// SVD (Singular Value Decomposition)
// One-sided Jacobi method for A = U·S·V^T
// ---------------------------------------------------------------------------

export interface SVDResult {
  U: Matrix;
  S: number[];
  V: Matrix;
}

/**
 * SVD via one-sided Jacobi rotations.
 * A = U·diag(S)·V^T where U is m×k, S has k singular values, V is n×k (k = min(m,n)).
 */
export function svd(A: Matrix): SVDResult {
  const m = A.rows;
  const n = A.cols;
  const k = Math.min(m, n);

  // Work on a copy: we'll iteratively orthogonalize the columns
  const W = A.clone();
  const V = Matrix.identity(n);

  const maxIter = 100;
  const tol = 1e-12;

  for (let iter = 0; iter < maxIter; iter++) {
    let converged = true;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        // Compute 2x2 subproblem: W^T W columns p,q
        let app = 0;
        let aqq = 0;
        let apq = 0;
        for (let i = 0; i < m; i++) {
          const wp = W.get(i, p);
          const wq = W.get(i, q);
          app += wp * wp;
          aqq += wq * wq;
          apq += wp * wq;
        }

        if (Math.abs(apq) < tol * Math.sqrt(app * aqq)) continue;
        converged = false;

        // Jacobi rotation angle
        const tau = (aqq - app) / (2 * apq);
        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Rotate columns p, q in W
        for (let i = 0; i < m; i++) {
          const wp = W.get(i, p);
          const wq = W.get(i, q);
          W.set(i, p, c * wp - s * wq);
          W.set(i, q, s * wp + c * wq);
        }

        // Accumulate V
        for (let i = 0; i < n; i++) {
          const vp = V.get(i, p);
          const vq = V.get(i, q);
          V.set(i, p, c * vp - s * vq);
          V.set(i, q, s * vp + c * vq);
        }
      }
    }

    if (converged) break;
  }

  // Extract singular values and form U
  const S: number[] = [];
  const U = Matrix.zeros(m, k);

  for (let j = 0; j < k; j++) {
    let norm = 0;
    for (let i = 0; i < m; i++) {
      norm += W.get(i, j) * W.get(i, j);
    }
    norm = Math.sqrt(norm);
    S.push(norm);
    if (norm > 1e-15) {
      for (let i = 0; i < m; i++) {
        U.set(i, j, W.get(i, j) / norm);
      }
    }
  }

  // Sort by descending singular value
  const indices = S.map((_, i) => i).sort((a, b) => S[b]! - S[a]!);
  const sortedS = indices.map((i) => S[i]!);
  const sortedU = Matrix.zeros(m, k);
  const sortedV = Matrix.zeros(n, k);
  for (let j = 0; j < k; j++) {
    const srcIdx = indices[j]!;
    for (let i = 0; i < m; i++) {
      sortedU.set(i, j, U.get(i, srcIdx));
    }
    for (let i = 0; i < n; i++) {
      sortedV.set(i, j, V.get(i, srcIdx));
    }
  }

  return { U: sortedU, S: sortedS, V: sortedV };
}

/** Solve Ax = b via SVD: x = V·S⁻¹·U^T·b, with threshold for near-zero singular values. */
export function solveSVD(U: Matrix, S: number[], V: Matrix, b: Matrix): Matrix {
  const threshold = 1e-10 * (S[0] ?? 1);
  const Utb = U.transpose().multiply(b);
  const n = V.cols;
  const x = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const si = i < S.length ? S[i]! : 0;
    if (si > threshold) {
      const coeff = Utb.get(i, 0) / si;
      for (let j = 0; j < n; j++) {
        x[j] += V.get(j, i) * coeff;
      }
    }
  }

  return Matrix.columnVector(Array.from(x));
}

// ---------------------------------------------------------------------------
// Eigenvalues (for symmetric matrices, via QR algorithm)
// ---------------------------------------------------------------------------

/**
 * Compute eigenvalues of a symmetric matrix using the iterative QR algorithm.
 * Returns eigenvalues sorted in descending order.
 */
export function eigenvalues(A: Matrix): number[] {
  if (A.rows !== A.cols) {
    throw new Error("Eigenvalues: matrix must be square");
  }
  const n = A.rows;
  if (n === 0) return [];
  if (n === 1) return [A.get(0, 0)];

  let T = A.clone();
  const maxIter = 200;

  for (let iter = 0; iter < maxIter; iter++) {
    // Check convergence first: off-diagonal elements near zero
    let maxOff = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) maxOff = Math.max(maxOff, Math.abs(T.get(i, j)));
      }
    }
    if (maxOff < 1e-12) break;

    // Wilkinson shift: use last 2x2 block
    const a = T.get(n - 2, n - 2);
    const b = T.get(n - 2, n - 1);
    const c = T.get(n - 1, n - 1);
    const delta = (a - c) / 2;
    let mu: number;
    if (Math.abs(b) < 1e-15) {
      mu = c;
    } else {
      const sign = delta >= 0 ? 1 : -1;
      mu = c - (sign * b * b) / (Math.abs(delta) + Math.sqrt(delta * delta + b * b));
    }

    // Shifted QR step
    const shifted = T.clone();
    for (let i = 0; i < n; i++) {
      shifted.set(i, i, shifted.get(i, i) - mu);
    }
    const { Q, R } = qrDecomposition(shifted);
    T = R.submatrix(0, n, 0, n).multiply(Q.submatrix(0, n, 0, n));
    for (let i = 0; i < n; i++) {
      T.set(i, i, T.get(i, i) + mu);
    }
  }

  const eigs: number[] = [];
  for (let i = 0; i < n; i++) {
    eigs.push(T.get(i, i));
  }
  return eigs.sort((a, b) => b - a);
}

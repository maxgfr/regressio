/**
 * Computation engine abstraction.
 * WASM is auto-loaded at module init when available (silent fallback to TypeScript).
 *
 * Accelerated operations: matrix multiply/transpose/add/subtract/scale/dot/norm/determinant,
 * QR/Cholesky/SVD/eigenvalues, forward/back substitution, coordinate descent (Lasso/ElasticNet),
 * softmax, KNN distance matrices.
 */

interface WasmModule {
  // Linear algebra
  matrix_multiply(
    a: Float64Array,
    a_rows: number,
    a_cols: number,
    b: Float64Array,
    b_rows: number,
    b_cols: number,
  ): Float64Array;
  matrix_transpose(data: Float64Array, rows: number, cols: number): Float64Array;
  matrix_add(a: Float64Array, b: Float64Array): Float64Array;
  matrix_subtract(a: Float64Array, b: Float64Array): Float64Array;
  matrix_scale(a: Float64Array, scalar: number): Float64Array;
  vector_dot(a: Float64Array, b: Float64Array): number;
  frobenius_norm(a: Float64Array): number;
  determinant(data: Float64Array, n: number): number;
  // Decompositions
  qr_decompose(data: Float64Array, rows: number, cols: number): Float64Array;
  cholesky(data: Float64Array, n: number): Float64Array;
  solve_triangular(r: Float64Array, b: Float64Array, n: number): Float64Array;
  forward_substitution(l: Float64Array, b: Float64Array, n: number): Float64Array;
  svd(data: Float64Array, rows: number, cols: number): Float64Array;
  eigenvalues(data: Float64Array, n: number): Float64Array;
  // Model algorithms
  coordinate_descent(
    x: Float64Array,
    y: Float64Array,
    alpha: number,
    l1_ratio: number,
    max_iter: number,
    tolerance: number,
    n: number,
    p: number,
    fit_intercept: boolean,
  ): Float64Array;
  softmax_rows(data: Float64Array, rows: number, cols: number): Float64Array;
  euclidean_distances(
    train: Float64Array,
    test: Float64Array,
    n_train: number,
    n_test: number,
    dim: number,
  ): Float64Array;
  manhattan_distances(
    train: Float64Array,
    test: Float64Array,
    n_train: number,
    n_test: number,
    dim: number,
  ): Float64Array;
  correlation_matrix(x: Float64Array, n: number, p: number): Float64Array;
  bootstrap_ols(
    x: Float64Array,
    y: Float64Array,
    n: number,
    p: number,
    fit_intercept: boolean,
    n_bootstrap: number,
    seed: number,
  ): Float64Array;
  vif(x: Float64Array, n: number, p: number): Float64Array;
  irls_logistic(
    x: Float64Array,
    y: Float64Array,
    n: number,
    k: number,
    max_iter: number,
    tolerance: number,
  ): Float64Array;
}

interface ComputeEngine {
  name: "typescript" | "wasm";
  wasm?: WasmModule;
}

let currentEngine: ComputeEngine = { name: "typescript" };

// Auto-init: silently try to load WASM at module load time.
import("#wasm-engine")
  .then(async (mod: { initWasm: () => Promise<void> }) => {
    await mod.initWasm();
    if (currentEngine.name === "typescript") {
      currentEngine = { name: "wasm", wasm: mod as unknown as WasmModule };
    }
  })
  .catch(() => {
    // WASM not available — stay on TypeScript silently
  });

/** Check if the WASM engine is active. */
export function isWasmActive(): boolean {
  return currentEngine.name === "wasm" && currentEngine.wasm != null;
}

// ===========================================================================
// Dispatch — Linear Algebra
// ===========================================================================

export function engineMatrixMultiply(
  a: Float64Array,
  aRows: number,
  aCols: number,
  b: Float64Array,
  bRows: number,
  bCols: number,
): Float64Array {
  if (currentEngine.wasm) {
    return currentEngine.wasm.matrix_multiply(a, aRows, aCols, b, bRows, bCols);
  }
  const result = new Float64Array(aRows * bCols);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i * aCols + k]!;
      for (let j = 0; j < bCols; j++) {
        result[i * bCols + j] += aik * b[k * bCols + j]!;
      }
    }
  }
  return result;
}

export function engineTranspose(
  data: Float64Array,
  rows: number,
  cols: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.matrix_transpose(data, rows, cols);
  return null;
}

export function engineAdd(a: Float64Array, b: Float64Array): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.matrix_add(a, b);
  return null;
}

export function engineSubtract(a: Float64Array, b: Float64Array): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.matrix_subtract(a, b);
  return null;
}

export function engineScale(a: Float64Array, scalar: number): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.matrix_scale(a, scalar);
  return null;
}

export function engineDot(a: Float64Array, b: Float64Array): number | null {
  if (currentEngine.wasm) return currentEngine.wasm.vector_dot(a, b);
  return null;
}

export function engineNorm(a: Float64Array): number | null {
  if (currentEngine.wasm) return currentEngine.wasm.frobenius_norm(a);
  return null;
}

export function engineDeterminant(data: Float64Array, n: number): number | null {
  if (currentEngine.wasm) return currentEngine.wasm.determinant(data, n);
  return null;
}

// ===========================================================================
// Dispatch — Decompositions
// ===========================================================================

export function engineQR(
  data: Float64Array,
  rows: number,
  cols: number,
): { Q: Float64Array; R: Float64Array } | null {
  if (currentEngine.wasm) {
    const result = currentEngine.wasm.qr_decompose(data, rows, cols);
    const qSize = rows * rows;
    return { Q: result.slice(0, qSize), R: result.slice(qSize) };
  }
  return null;
}

export function engineCholesky(data: Float64Array, n: number): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.cholesky(data, n);
  return null;
}

export function engineSolveTriangular(
  r: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.solve_triangular(r, b, n);
  return null;
}

export function engineForwardSubstitution(
  l: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.forward_substitution(l, b, n);
  return null;
}

export function engineSVD(
  data: Float64Array,
  rows: number,
  cols: number,
): { U: Float64Array; S: Float64Array; V: Float64Array } | null {
  if (currentEngine.wasm) {
    const k = Math.min(rows, cols);
    const result = currentEngine.wasm.svd(data, rows, cols);
    const uSize = rows * k;
    return {
      U: result.slice(0, uSize),
      S: result.slice(uSize, uSize + k),
      V: result.slice(uSize + k),
    };
  }
  return null;
}

export function engineEigenvalues(data: Float64Array, n: number): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.eigenvalues(data, n);
  return null;
}

// ===========================================================================
// Dispatch — Model algorithms
// ===========================================================================

/** Coordinate descent for Lasso/ElasticNet. Returns [intercept, ...coefficients] or null. */
export function engineCoordinateDescent(
  x: Float64Array,
  y: Float64Array,
  alpha: number,
  l1Ratio: number,
  maxIter: number,
  tolerance: number,
  n: number,
  p: number,
  fitIntercept: boolean,
): Float64Array | null {
  if (currentEngine.wasm) {
    return currentEngine.wasm.coordinate_descent(
      x,
      y,
      alpha,
      l1Ratio,
      maxIter,
      tolerance,
      n,
      p,
      fitIntercept,
    );
  }
  return null;
}

/** Row-wise softmax. Returns flat probability matrix or null. */
export function engineSoftmaxRows(
  data: Float64Array,
  rows: number,
  cols: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.softmax_rows(data, rows, cols);
  return null;
}

/** Euclidean distance matrix. Returns flat n_test × n_train or null. */
export function engineEuclideanDistances(
  train: Float64Array,
  test: Float64Array,
  nTrain: number,
  nTest: number,
  dim: number,
): Float64Array | null {
  if (currentEngine.wasm)
    return currentEngine.wasm.euclidean_distances(train, test, nTrain, nTest, dim);
  return null;
}

/** Manhattan distance matrix. Returns flat n_test × n_train or null. */
export function engineManhattanDistances(
  train: Float64Array,
  test: Float64Array,
  nTrain: number,
  nTest: number,
  dim: number,
): Float64Array | null {
  if (currentEngine.wasm)
    return currentEngine.wasm.manhattan_distances(train, test, nTrain, nTest, dim);
  return null;
}

/** Pearson correlation matrix. Returns flat p × p or null. */
export function engineCorrelationMatrix(
  x: Float64Array,
  n: number,
  p: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.correlation_matrix(x, n, p);
  return null;
}

/** Bootstrap OLS. Returns flat (nBootstrap × nParams) or null. NaN rows = singular sample. */
export function engineBootstrapOLS(
  x: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  fitIntercept: boolean,
  nBootstrap: number,
  seed: number,
): Float64Array | null {
  if (currentEngine.wasm)
    return currentEngine.wasm.bootstrap_ols(x, y, n, p, fitIntercept, nBootstrap, seed);
  return null;
}

/** VIF from X. Returns Float64Array of p VIF values or null. */
export function engineVIF(x: Float64Array, n: number, p: number): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.vif(x, n, p);
  return null;
}

/** Full IRLS logistic regression. x = design matrix (n×k), y = labels (n). Returns beta (k) or null. */
export function engineIRLSLogistic(
  x: Float64Array,
  y: Float64Array,
  n: number,
  k: number,
  maxIter: number,
  tolerance: number,
): Float64Array | null {
  if (currentEngine.wasm) return currentEngine.wasm.irls_logistic(x, y, n, k, maxIter, tolerance);
  return null;
}

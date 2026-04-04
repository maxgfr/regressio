/* tslint:disable */
/* eslint-disable */

/**
 * Cholesky decomposition: A = L·L^T. Returns L (n*n flat).
 */
export function cholesky(data: Float64Array, n: number): Float64Array;

/**
 * Lasso / Elastic Net coordinate descent.
 * Input: flat X (n×p row-major), y (n), params.
 * Returns [intercept, coeff_0, …, coeff_{p-1}].
 */
export function coordinate_descent(x: Float64Array, y: Float64Array, alpha: number, l1_ratio: number, max_iter: number, tolerance: number, n: number, p: number, fit_intercept: boolean): Float64Array;

export function determinant(data: Float64Array, n: number): number;

/**
 * Eigenvalues of a symmetric matrix (QR + Wilkinson shift). Sorted descending.
 */
export function eigenvalues(data: Float64Array, n: number): Float64Array;

/**
 * Euclidean distance matrix: test (n_test × dim) vs train (n_train × dim).
 * Returns flat n_test × n_train matrix.
 */
export function euclidean_distances(train: Float64Array, test: Float64Array, n_train: number, n_test: number, dim: number): Float64Array;

/**
 * Forward substitution: Lx = b (lower triangular).
 */
export function forward_substitution(l: Float64Array, b: Float64Array, n: number): Float64Array;

export function frobenius_norm(a: Float64Array): number;

/**
 * Manhattan distance matrix.
 */
export function manhattan_distances(train: Float64Array, test: Float64Array, n_train: number, n_test: number, dim: number): Float64Array;

export function matrix_add(a: Float64Array, b: Float64Array): Float64Array;

export function matrix_multiply(a: Float64Array, a_rows: number, a_cols: number, b: Float64Array, b_rows: number, b_cols: number): Float64Array;

export function matrix_scale(a: Float64Array, scalar: number): Float64Array;

export function matrix_subtract(a: Float64Array, b: Float64Array): Float64Array;

export function matrix_transpose(data: Float64Array, rows: number, cols: number): Float64Array;

/**
 * QR decomposition. Returns Q (m*m) ‖ R (m*n) packed.
 */
export function qr_decompose(data: Float64Array, rows: number, cols: number): Float64Array;

/**
 * Row-wise softmax: (rows × cols) → probabilities.
 */
export function softmax_rows(data: Float64Array, rows: number, cols: number): Float64Array;

/**
 * Back substitution: Rx = b (upper triangular).
 */
export function solve_triangular(r: Float64Array, b: Float64Array, n: number): Float64Array;

/**
 * SVD via one-sided Jacobi. Returns U (m*k) ‖ S (k) ‖ V (n*k), k = min(m,n).
 */
export function svd(data: Float64Array, rows: number, cols: number): Float64Array;

export function vector_dot(a: Float64Array, b: Float64Array): number;

/**
 * Bootstrap OLS: run n_bootstrap resamples via efficient QR solve (no Q formation).
 * Returns flat (n_bootstrap × n_params). NaN rows = singular sample.
 * @param {Float64Array} x
 * @param {Float64Array} y
 * @param {number} n
 * @param {number} p
 * @param {boolean} fit_intercept
 * @param {number} n_bootstrap
 * @param {number} seed
 * @returns {Float64Array}
 */
export function bootstrap_ols(x, y, n, p, fit_intercept, n_bootstrap, seed) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.bootstrap_ols(ptr0, len0, ptr1, len1, n, p, fit_intercept, n_bootstrap, seed);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Cholesky decomposition: A = L·L^T. Returns L (n*n flat).
 * @param {Float64Array} data
 * @param {number} n
 * @returns {Float64Array}
 */
export function cholesky(data, n) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cholesky(ptr0, len0, n);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Lasso / Elastic Net coordinate descent.
 * Input: flat X (n×p row-major), y (n), params.
 * Returns [intercept, coeff_0, …, coeff_{p-1}].
 * @param {Float64Array} x
 * @param {Float64Array} y
 * @param {number} alpha
 * @param {number} l1_ratio
 * @param {number} max_iter
 * @param {number} tolerance
 * @param {number} n
 * @param {number} p
 * @param {boolean} fit_intercept
 * @returns {Float64Array}
 */
export function coordinate_descent(x, y, alpha, l1_ratio, max_iter, tolerance, n, p, fit_intercept) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.coordinate_descent(ptr0, len0, ptr1, len1, alpha, l1_ratio, max_iter, tolerance, n, p, fit_intercept);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Pearson correlation matrix from flat row-major X (n × p). Returns flat p × p.
 * @param {Float64Array} x
 * @param {number} n
 * @param {number} p
 * @returns {Float64Array}
 */
export function correlation_matrix(x, n, p) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.correlation_matrix(ptr0, len0, n, p);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {Float64Array} data
 * @param {number} n
 * @returns {number}
 */
export function determinant(data, n) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.determinant(ptr0, len0, n);
    return ret;
}

/**
 * Eigenvalues of a symmetric matrix via Householder tridiagonalization + QL iteration.
 * O(n³) tridiag + O(n·iter) QL, much faster than the old O(n³·iter) full QR approach.
 * Sorted descending.
 * @param {Float64Array} data
 * @param {number} n
 * @returns {Float64Array}
 */
export function eigenvalues(data, n) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.eigenvalues(ptr0, len0, n);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Euclidean distance matrix: test (n_test × dim) vs train (n_train × dim).
 * Returns flat n_test × n_train matrix.
 * @param {Float64Array} train
 * @param {Float64Array} test
 * @param {number} n_train
 * @param {number} n_test
 * @param {number} dim
 * @returns {Float64Array}
 */
export function euclidean_distances(train, test, n_train, n_test, dim) {
    const ptr0 = passArrayF64ToWasm0(train, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(test, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.euclidean_distances(ptr0, len0, ptr1, len1, n_train, n_test, dim);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Forward substitution: Lx = b (lower triangular).
 * @param {Float64Array} l
 * @param {Float64Array} b
 * @param {number} n
 * @returns {Float64Array}
 */
export function forward_substitution(l, b, n) {
    const ptr0 = passArrayF64ToWasm0(l, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.forward_substitution(ptr0, len0, ptr1, len1, n);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * @param {Float64Array} a
 * @returns {number}
 */
export function frobenius_norm(a) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.frobenius_norm(ptr0, len0);
    return ret;
}

/**
 * Full IRLS (Newton-Raphson) for binary logistic regression.
 * x: design matrix (n × k, row-major, INCLUDING intercept column if needed)
 * y: binary labels (0/1), length n
 * Returns: beta coefficients (length k)
 * @param {Float64Array} x
 * @param {Float64Array} y
 * @param {number} n
 * @param {number} k
 * @param {number} max_iter
 * @param {number} tolerance
 * @returns {Float64Array}
 */
export function irls_logistic(x, y, n, k, max_iter, tolerance) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.irls_logistic(ptr0, len0, ptr1, len1, n, k, max_iter, tolerance);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Manhattan distance matrix.
 * @param {Float64Array} train
 * @param {Float64Array} test
 * @param {number} n_train
 * @param {number} n_test
 * @param {number} dim
 * @returns {Float64Array}
 */
export function manhattan_distances(train, test, n_train, n_test, dim) {
    const ptr0 = passArrayF64ToWasm0(train, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(test, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.manhattan_distances(ptr0, len0, ptr1, len1, n_train, n_test, dim);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * @param {Float64Array} a
 * @param {Float64Array} b
 * @returns {Float64Array}
 */
export function matrix_add(a, b) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.matrix_add(ptr0, len0, ptr1, len1);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * @param {Float64Array} a
 * @param {number} a_rows
 * @param {number} a_cols
 * @param {Float64Array} b
 * @param {number} b_rows
 * @param {number} b_cols
 * @returns {Float64Array}
 */
export function matrix_multiply(a, a_rows, a_cols, b, b_rows, b_cols) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.matrix_multiply(ptr0, len0, a_rows, a_cols, ptr1, len1, b_rows, b_cols);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * @param {Float64Array} a
 * @param {number} scalar
 * @returns {Float64Array}
 */
export function matrix_scale(a, scalar) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.matrix_scale(ptr0, len0, scalar);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {Float64Array} a
 * @param {Float64Array} b
 * @returns {Float64Array}
 */
export function matrix_subtract(a, b) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.matrix_subtract(ptr0, len0, ptr1, len1);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * @param {Float64Array} data
 * @param {number} rows
 * @param {number} cols
 * @returns {Float64Array}
 */
export function matrix_transpose(data, rows, cols) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.matrix_transpose(ptr0, len0, rows, cols);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * QR decomposition. Returns Q (m*m) ‖ R (m*n) packed.
 * @param {Float64Array} data
 * @param {number} rows
 * @param {number} cols
 * @returns {Float64Array}
 */
export function qr_decompose(data, rows, cols) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.qr_decompose(ptr0, len0, rows, cols);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Row-wise softmax: (rows × cols) → probabilities.
 * @param {Float64Array} data
 * @param {number} rows
 * @param {number} cols
 * @returns {Float64Array}
 */
export function softmax_rows(data, rows, cols) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.softmax_rows(ptr0, len0, rows, cols);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Back substitution: Rx = b (upper triangular).
 * @param {Float64Array} r
 * @param {Float64Array} b
 * @param {number} n
 * @returns {Float64Array}
 */
export function solve_triangular(r, b, n) {
    const ptr0 = passArrayF64ToWasm0(r, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.solve_triangular(ptr0, len0, ptr1, len1, n);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * SVD via one-sided Jacobi. Returns U (m*k) ‖ S (k) ‖ V (n*k), k = min(m,n).
 * @param {Float64Array} data
 * @param {number} rows
 * @param {number} cols
 * @returns {Float64Array}
 */
export function svd(data, rows, cols) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.svd(ptr0, len0, rows, cols);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {Float64Array} a
 * @param {Float64Array} b
 * @returns {number}
 */
export function vector_dot(a, b) {
    const ptr0 = passArrayF64ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vector_dot(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Compute VIF from X (n × p) via correlation matrix inverse diagonal.
 * VIF_j = (C⁻¹)_{jj} where C is the Pearson correlation matrix.
 * @param {Float64Array} x
 * @param {number} n
 * @param {number} p
 * @returns {Float64Array}
 */
export function vif(x, n, p) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vif(ptr0, len0, n, p);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}
export function __wbindgen_init_externref_table() {
    const table = wasm.__wbindgen_externrefs;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
}
function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;


let wasm;
export function __wbg_set_wasm(val) {
    wasm = val;
}

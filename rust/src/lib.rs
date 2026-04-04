use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matrix_multiply_internal(
    a: &[f64], a_rows: usize, a_cols: usize,
    b: &[f64], _b_rows: usize, b_cols: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let a_ik = a[i * a_cols + k];
            for j in 0..b_cols {
                result[i * b_cols + j] += a_ik * b[k * b_cols + j];
            }
        }
    }
    result
}

fn qr_decompose_internal(data: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = data.to_vec();
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    let min_mn = m.min(n);
    for k in 0..min_mn {
        let len = m - k;
        let mut x = vec![0.0; len];
        for i in 0..len {
            x[i] = r[(k + i) * n + k];
        }

        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if x_norm < 1e-15 { continue; }

        let alpha = if x[0] >= 0.0 { -x_norm } else { x_norm };
        let mut v = x.clone();
        v[0] -= alpha;

        let v_norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if v_norm < 1e-15 { continue; }
        for vi in v.iter_mut() { *vi /= v_norm; }

        for j in k..n {
            let dot: f64 = (0..len).map(|i| v[i] * r[(k + i) * n + j]).sum();
            for i in 0..len {
                r[(k + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        for i in 0..m {
            let dot: f64 = (0..len).map(|j| q[i * m + (k + j)] * v[j]).sum();
            for j in 0..len {
                q[i * m + (k + j)] -= 2.0 * dot * v[j];
            }
        }
    }

    (q, r)
}

fn soft_threshold(rho: f64, lambda: f64) -> f64 {
    if rho < -lambda { rho + lambda }
    else if rho > lambda { rho - lambda }
    else { 0.0 }
}

// ===========================================================================
// WASM exports — Linear Algebra
// ===========================================================================

#[wasm_bindgen]
pub fn matrix_multiply(
    a: &[f64], a_rows: usize, a_cols: usize,
    b: &[f64], b_rows: usize, b_cols: usize,
) -> Vec<f64> {
    assert_eq!(a_cols, b_rows, "Incompatible matrix dimensions");
    matrix_multiply_internal(a, a_rows, a_cols, b, b_rows, b_cols)
}

#[wasm_bindgen]
pub fn matrix_transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

#[wasm_bindgen]
pub fn matrix_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[wasm_bindgen]
pub fn matrix_subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[wasm_bindgen]
pub fn matrix_scale(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|x| x * scalar).collect()
}

#[wasm_bindgen]
pub fn vector_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[wasm_bindgen]
pub fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[wasm_bindgen]
pub fn determinant(data: &[f64], n: usize) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return data[0]; }
    if n == 2 { return data[0] * data[3] - data[1] * data[2]; }

    let mut a = data.to_vec();
    let mut det = 1.0;

    for col in 0..n {
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val { max_val = val; max_row = row; }
        }
        if max_val < 1e-15 { return 0.0; }

        if max_row != col {
            det = -det;
            for j in 0..n { a.swap(col * n + j, max_row * n + j); }
        }

        det *= a[col * n + col];
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in (col + 1)..n {
                let val = a[col * n + j];
                a[row * n + j] -= factor * val;
            }
        }
    }
    det
}

// ===========================================================================
// WASM exports — Decompositions
// ===========================================================================

/// QR decomposition. Returns Q (m*m) ‖ R (m*n) packed.
#[wasm_bindgen]
pub fn qr_decompose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let (q, r) = qr_decompose_internal(data, rows, cols);
    let mut result = Vec::with_capacity(q.len() + r.len());
    result.extend_from_slice(&q);
    result.extend_from_slice(&r);
    result
}

/// Cholesky decomposition: A = L·L^T. Returns L (n*n flat).
#[wasm_bindgen]
pub fn cholesky(data: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j { sum += l[j * n + k] * l[j * n + k]; }
        let diag = data[j * n + j] - sum;
        assert!(diag > 0.0, "Matrix is not positive-definite");
        l[j * n + j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j { s += l[i * n + k] * l[j * n + k]; }
            l[i * n + j] = (data[i * n + j] - s) / l[j * n + j];
        }
    }
    l
}

/// Back substitution: Rx = b (upper triangular).
#[wasm_bindgen]
pub fn solve_triangular(r: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum -= r[i * n + j] * x[j]; }
        x[i] = sum / r[i * n + i];
    }
    x
}

/// Forward substitution: Lx = b (lower triangular).
#[wasm_bindgen]
pub fn forward_substitution(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i { sum -= l[i * n + j] * x[j]; }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// SVD via one-sided Jacobi. Returns U (m*k) ‖ S (k) ‖ V (n*k), k = min(m,n).
#[wasm_bindgen]
pub fn svd(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let m = rows;
    let n = cols;
    let k = m.min(n);

    let mut w = data.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }

    for _iter in 0..100 {
        let mut converged = true;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let (mut app, mut aqq, mut apq) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    app += wp * wp;
                    aqq += wq * wq;
                    apq += wp * wq;
                }
                if apq.abs() < 1e-12 * (app * aqq).sqrt() { continue; }
                converged = false;

                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for i in 0..m {
                    let wp = w[i * n + p]; let wq = w[i * n + q];
                    w[i * n + p] = c * wp - s * wq;
                    w[i * n + q] = s * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[i * n + p]; let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if converged { break; }
    }

    let mut svals = vec![0.0; k];
    let mut u = vec![0.0; m * k];
    for j in 0..k {
        let mut norm = 0.0;
        for i in 0..m { norm += w[i * n + j] * w[i * n + j]; }
        norm = norm.sqrt();
        svals[j] = norm;
        if norm > 1e-15 {
            for i in 0..m { u[i * k + j] = w[i * n + j] / norm; }
        }
    }

    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|a, b| svals[*b].partial_cmp(&svals[*a]).unwrap());

    let mut result = Vec::with_capacity(m * k + k + n * k);
    for i in 0..m { for j in &idx { result.push(u[i * k + j]); } }
    for j in &idx { result.push(svals[*j]); }
    for i in 0..n { for j in &idx { result.push(v[i * n + j]); } }
    result
}

/// Eigenvalues of a symmetric matrix (QR + Wilkinson shift). Sorted descending.
#[wasm_bindgen]
pub fn eigenvalues(data: &[f64], n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    if n == 1 { return vec![data[0]]; }

    let mut t = data.to_vec();

    for _iter in 0..200 {
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j { max_off = max_off.max(t[i * n + j].abs()); }
            }
        }
        if max_off < 1e-12 { break; }

        let a = t[(n - 2) * n + (n - 2)];
        let b = t[(n - 2) * n + (n - 1)];
        let c = t[(n - 1) * n + (n - 1)];
        let delta = (a - c) / 2.0;
        let mu = if b.abs() < 1e-15 { c } else {
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            c - (sign * b * b) / (delta.abs() + (delta * delta + b * b).sqrt())
        };

        let mut shifted = t.clone();
        for i in 0..n { shifted[i * n + i] -= mu; }

        let (q, r) = qr_decompose_internal(&shifted, n, n);
        t = matrix_multiply_internal(&r, n, n, &q, n, n);
        for i in 0..n { t[i * n + i] += mu; }
    }

    let mut eigs: Vec<f64> = (0..n).map(|i| t[i * n + i]).collect();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigs
}

// ===========================================================================
// WASM exports — Model algorithms
// ===========================================================================

/// Lasso / Elastic Net coordinate descent.
/// Input: flat X (n×p row-major), y (n), params.
/// Returns [intercept, coeff_0, …, coeff_{p-1}].
#[wasm_bindgen]
pub fn coordinate_descent(
    x: &[f64], y: &[f64],
    alpha: f64, l1_ratio: f64,
    max_iter: usize, tolerance: f64,
    n: usize, p: usize,
    fit_intercept: bool,
) -> Vec<f64> {
    let nf = n as f64;

    // Standardize X
    let mut x_means = vec![0.0; p];
    let mut x_stds = vec![0.0; p];
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n { sum += x[i * p + j]; }
        x_means[j] = sum / nf;
    }
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            let d = x[i * p + j] - x_means[j];
            sum += d * d;
        }
        x_stds[j] = (sum / nf).sqrt();
    }

    let mut x_std = vec![0.0; n * p];
    for i in 0..n {
        for j in 0..p {
            x_std[i * p + j] = if x_stds[j] > 1e-15 {
                (x[i * p + j] - x_means[j]) / x_stds[j]
            } else { 0.0 };
        }
    }

    // Center y
    let y_mean: f64 = y.iter().sum::<f64>() / nf;
    let y_centered: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();

    // Column norms squared
    let mut col_norms_sq = vec![0.0; p];
    for j in 0..p {
        for i in 0..n {
            col_norms_sq[j] += x_std[i * p + j] * x_std[i * p + j];
        }
    }

    let mut beta = vec![0.0; p];
    let mut residual = y_centered;

    for _iter in 0..max_iter {
        let mut max_change = 0.0_f64;

        for j in 0..p {
            let old = beta[j];
            let mut rho = 0.0;
            for i in 0..n {
                rho += x_std[i * p + j] * (residual[i] + old * x_std[i * p + j]);
            }

            let l1 = nf * alpha * l1_ratio;
            let l2 = nf * alpha * (1.0 - l1_ratio);
            beta[j] = soft_threshold(rho, l1) / (col_norms_sq[j] + l2);

            let change = beta[j] - old;
            if change != 0.0 {
                for i in 0..n { residual[i] -= change * x_std[i * p + j]; }
            }
            max_change = max_change.max(change.abs());
        }

        if max_change < tolerance { break; }
    }

    // Un-standardize → [intercept, coeffs...]
    let mut result = vec![0.0; p + 1];
    let mut intercept = if fit_intercept { y_mean } else { 0.0 };
    for j in 0..p {
        let coeff = if x_stds[j] > 1e-15 { beta[j] / x_stds[j] } else { 0.0 };
        result[j + 1] = coeff;
        if fit_intercept { intercept -= coeff * x_means[j]; }
    }
    result[0] = intercept;
    result
}

/// Row-wise softmax: (rows × cols) → probabilities.
#[wasm_bindgen]
pub fn softmax_rows(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows * cols];
    for i in 0..rows {
        let off = i * cols;
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..cols {
            if data[off + j] > max_val { max_val = data[off + j]; }
        }
        let mut sum = 0.0;
        for j in 0..cols {
            result[off + j] = (data[off + j] - max_val).exp();
            sum += result[off + j];
        }
        for j in 0..cols { result[off + j] /= sum; }
    }
    result
}

/// Euclidean distance matrix: test (n_test × dim) vs train (n_train × dim).
/// Returns flat n_test × n_train matrix.
#[wasm_bindgen]
pub fn euclidean_distances(
    train: &[f64], test: &[f64],
    n_train: usize, n_test: usize, dim: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; n_test * n_train];
    for i in 0..n_test {
        for j in 0..n_train {
            let mut sum = 0.0;
            for d in 0..dim {
                let diff = test[i * dim + d] - train[j * dim + d];
                sum += diff * diff;
            }
            result[i * n_train + j] = sum.sqrt();
        }
    }
    result
}

/// Manhattan distance matrix.
#[wasm_bindgen]
pub fn manhattan_distances(
    train: &[f64], test: &[f64],
    n_train: usize, n_test: usize, dim: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; n_test * n_train];
    for i in 0..n_test {
        for j in 0..n_train {
            let mut sum = 0.0;
            for d in 0..dim {
                sum += (test[i * dim + d] - train[j * dim + d]).abs();
            }
            result[i * n_train + j] = sum;
        }
    }
    result
}

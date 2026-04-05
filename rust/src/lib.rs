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

/// QR solve Ax=b without forming Q: apply Householder reflections to b directly.
/// Returns None if singular. O(mn) instead of O(m²).
fn qr_solve_internal(a_data: &[f64], b_data: &[f64], m: usize, n: usize) -> Option<Vec<f64>> {
    if m < n { return None; }

    let mut r = a_data.to_vec();
    let mut qtb = b_data.to_vec();

    let min_mn = m.min(n);
    for k in 0..min_mn {
        let len = m - k;

        let mut x_norm_sq = 0.0;
        for i in 0..len {
            let val = r[(k + i) * n + k];
            x_norm_sq += val * val;
        }
        let x_norm = x_norm_sq.sqrt();
        if x_norm < 1e-15 { continue; }

        let alpha = if r[k * n + k] >= 0.0 { -x_norm } else { x_norm };

        let mut v = vec![0.0; len];
        v[0] = r[k * n + k] - alpha;
        for i in 1..len {
            v[i] = r[(k + i) * n + k];
        }

        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-15 { continue; }
        for vi in v.iter_mut() { *vi /= v_norm; }

        // Apply reflection to R[k:, k:]
        for j in k..n {
            let dot: f64 = (0..len).map(|i| v[i] * r[(k + i) * n + j]).sum();
            for i in 0..len {
                r[(k + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        // Apply reflection to b[k:]
        let dot: f64 = (0..len).map(|i| v[i] * qtb[k + i]).sum();
        for i in 0..len {
            qtb[k + i] -= 2.0 * v[i] * dot;
        }
    }

    // Back substitution: R[0:n, 0:n] * x = qtb[0:n]
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = qtb[i];
        for j in (i + 1)..n {
            sum -= r[i * n + j] * x[j];
        }
        let diag = r[i * n + i];
        if diag.abs() < 1e-15 { return None; }
        x[i] = sum / diag;
    }

    Some(x)
}

fn soft_threshold(rho: f64, lambda: f64) -> f64 {
    if rho < -lambda { rho + lambda }
    else if rho > lambda { rho - lambda }
    else { 0.0 }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Householder reduction of symmetric matrix to tridiagonal form.
/// Returns (diagonal, subdiagonal) where subdiagonal has length n-1.
fn tridiagonalize(data: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mat = data.to_vec();
    let mut d = vec![0.0; n];
    let sub_len = if n > 1 { n - 1 } else { 0 };
    let mut e = vec![0.0; sub_len];

    if n <= 2 {
        d[0] = mat[0];
        if n == 2 {
            d[1] = mat[3];
            e[0] = mat[1];
        }
        return (d, e);
    }

    for k in 0..(n - 2) {
        // Compute ||x||² where x = column k below row k+1
        let mut sigma = 0.0;
        for i in (k + 1)..n {
            sigma += mat[i * n + k] * mat[i * n + k];
        }

        if sigma < 1e-30 {
            e[k] = mat[(k + 1) * n + k];
            continue;
        }

        let f = mat[(k + 1) * n + k];
        let alpha = if f >= 0.0 { -sigma.sqrt() } else { sigma.sqrt() };
        e[k] = alpha;

        // h = ||v||² / 2 = (sigma - f * alpha) for Householder vector v
        let h = sigma - f * alpha;
        mat[(k + 1) * n + k] = f - alpha;

        // Compute p = A * v / h (reusing mat[k+1..n, k] as v)
        let mut p = vec![0.0; n];
        for i in (k + 1)..n {
            let mut sum = 0.0;
            for j in (k + 1)..n {
                sum += mat[i * n + j] * mat[j * n + k];
            }
            p[i] = sum / h;
        }

        // K = v^T p / (2h)
        let mut k_val = 0.0;
        for i in (k + 1)..n {
            k_val += mat[i * n + k] * p[i];
        }
        k_val /= 2.0 * h;

        // q = p - K*v (store in p)
        for i in (k + 1)..n {
            p[i] -= k_val * mat[i * n + k];
        }

        // A -= v*q^T + q*v^T (symmetric update)
        for i in (k + 1)..n {
            for j in (k + 1)..n {
                mat[i * n + j] -= mat[i * n + k] * p[j] + p[i] * mat[j * n + k];
            }
        }
    }

    // Extract diagonal and last subdiagonal element
    for i in 0..n {
        d[i] = mat[i * n + i];
    }
    if n >= 2 {
        e[n - 2] = mat[(n - 1) * n + (n - 2)];
    }

    (d, e)
}

/// QL algorithm with implicit Wilkinson shift for symmetric tridiagonal eigenvalues.
/// Modifies d in-place. O(n) per iteration. Based on Numerical Recipes tqli.
fn tqli(d: &mut [f64], e: &mut [f64], n: usize) {
    if n <= 1 { return; }

    for l in 0..n {
        let mut niter = 0;
        loop {
            // Find small subdiagonal element
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if dd + e[m].abs() == dd { break; }
                m += 1;
            }
            if m == l { break; }

            niter += 1;
            if niter > 30 { break; }

            // Wilkinson shift from 2×2 block at position (l, l+1)
            let g_init = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r_init = (g_init * g_init + 1.0).sqrt();
            let mut g = d[m] - d[l] + e[l] / (g_init + r_init.copysign(g_init));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            let mut i = m as isize - 1;
            let mut broke = false;
            while i >= l as isize {
                let ii = i as usize;
                let f = s * e[ii];
                let b = c * e[ii];
                let r = (f * f + g * g).sqrt();
                e[ii + 1] = r;
                if r < 1e-30 {
                    d[ii + 1] -= p;
                    e[m] = 0.0;
                    broke = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[ii + 1] - p;
                let r2 = (d[ii] - g) * s + 2.0 * c * b;
                p = s * r2;
                d[ii + 1] = g + p;
                g = c * r2 - b;
                i -= 1;
            }
            if broke { continue; }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
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
        if diag <= 0.0 {
            panic!("Cholesky: matrix is not positive-definite");
        }
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

    // Extract ALL n column norms (not just k) — Jacobi may place
    // non-zero singular values in any column for wide matrices (m < n).
    let mut svals = vec![0.0; n];
    for j in 0..n {
        let mut norm = 0.0;
        for i in 0..m { norm += w[i * n + j] * w[i * n + j]; }
        svals[j] = norm.sqrt();
    }

    // Sort ALL n columns descending by singular value, take top k
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| svals[*b].partial_cmp(&svals[*a]).unwrap());

    let mut result = Vec::with_capacity(m * k + k + n * k);
    // U (m × k): normalized columns of W for top k indices
    for i in 0..m {
        for ji in 0..k {
            let j = idx[ji];
            let s = svals[j];
            result.push(if s > 1e-15 { w[i * n + j] / s } else { 0.0 });
        }
    }
    // S (k): top k singular values
    for ji in 0..k { result.push(svals[idx[ji]]); }
    // V (n × k): corresponding columns of V
    for i in 0..n {
        for ji in 0..k {
            result.push(v[i * n + idx[ji]]);
        }
    }
    result
}

/// Eigenvalues of a symmetric matrix via Householder tridiagonalization + QL iteration.
/// O(n³) tridiag + O(n·iter) QL, much faster than the old O(n³·iter) full QR approach.
/// Sorted descending.
#[wasm_bindgen]
pub fn eigenvalues(data: &[f64], n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    if n == 1 { return vec![data[0]]; }
    if n == 2 {
        let a = data[0]; let d = data[3];
        let b = data[1];
        let tr = a + d;
        let det = a * d - b * data[2];
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let mut eigs = vec![(tr + disc) / 2.0, (tr - disc) / 2.0];
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        return eigs;
    }

    let (mut diag, mut subdiag) = tridiagonalize(data, n);
    tqli(&mut diag, &mut subdiag, n);
    diag.sort_by(|a, b| b.partial_cmp(a).unwrap());
    diag
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

    let y_mean: f64 = y.iter().sum::<f64>() / nf;
    let y_centered: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();

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

/// Pearson correlation matrix from flat row-major X (n × p). Returns flat p × p.
#[wasm_bindgen]
pub fn correlation_matrix(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut means = vec![0.0; p];
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n { sum += x[i * p + j]; }
        means[j] = sum / nf;
    }
    let mut stds = vec![0.0; p];
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            let d = x[i * p + j] - means[j];
            sum += d * d;
        }
        stds[j] = (sum / nf).sqrt();
    }

    let mut result = vec![0.0; p * p];
    for j1 in 0..p {
        result[j1 * p + j1] = 1.0;
        for j2 in (j1 + 1)..p {
            let s1 = if stds[j1] > 1e-15 { stds[j1] } else { 1.0 };
            let s2 = if stds[j2] > 1e-15 { stds[j2] } else { 1.0 };
            let mut sum = 0.0;
            for i in 0..n {
                sum += ((x[i * p + j1] - means[j1]) / s1)
                     * ((x[i * p + j2] - means[j2]) / s2);
            }
            let corr = sum / nf;
            result[j1 * p + j2] = corr;
            result[j2 * p + j1] = corr;
        }
    }
    result
}

/// Bootstrap OLS: run n_bootstrap resamples via efficient QR solve (no Q formation).
/// Returns flat (n_bootstrap × n_params). NaN rows = singular sample.
#[wasm_bindgen]
pub fn bootstrap_ols(
    x: &[f64], y: &[f64],
    n: usize, p: usize,
    fit_intercept: bool,
    n_bootstrap: usize,
    seed: u32,
) -> Vec<f64> {
    let k = if fit_intercept { p + 1 } else { p };
    let mut result = vec![f64::NAN; n_bootstrap * k];

    let mut rng_state = seed as u64;
    let lcg_next = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n
    };

    let mut x_design = vec![0.0; n * k];
    let mut y_buf = vec![0.0; n];

    for b in 0..n_bootstrap {
        for i in 0..n {
            let idx = lcg_next(&mut rng_state);
            if fit_intercept {
                x_design[i * k] = 1.0;
                for j in 0..p {
                    x_design[i * k + j + 1] = x[idx * p + j];
                }
            } else {
                for j in 0..p {
                    x_design[i * k + j] = x[idx * p + j];
                }
            }
            y_buf[i] = y[idx];
        }

        if let Some(beta) = qr_solve_internal(&x_design, &y_buf, n, k) {
            for j in 0..k {
                result[b * k + j] = beta[j];
            }
        }
    }

    result
}

/// Compute VIF from X (n × p) via correlation matrix inverse diagonal.
/// VIF_j = (C⁻¹)_{jj} where C is the Pearson correlation matrix.
#[wasm_bindgen]
pub fn vif(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    if p < 2 { return vec![1.0]; }
    let nf = n as f64;

    // Compute correlation matrix
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n { sum += x[i * p + j]; }
        means[j] = sum / nf;
    }
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            let d = x[i * p + j] - means[j];
            sum += d * d;
        }
        stds[j] = (sum / nf).sqrt();
    }

    let mut corr = vec![0.0; p * p];
    for j1 in 0..p {
        corr[j1 * p + j1] = 1.0;
        for j2 in (j1 + 1)..p {
            let s1 = if stds[j1] > 1e-15 { stds[j1] } else { 1.0 };
            let s2 = if stds[j2] > 1e-15 { stds[j2] } else { 1.0 };
            let mut sum = 0.0;
            for i in 0..n {
                sum += ((x[i * p + j1] - means[j1]) / s1)
                     * ((x[i * p + j2] - means[j2]) / s2);
            }
            let c = sum / nf;
            corr[j1 * p + j2] = c;
            corr[j2 * p + j1] = c;
        }
    }

    // Cholesky decomposition of correlation matrix
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut sum = 0.0;
        for kk in 0..j { sum += l[j * p + kk] * l[j * p + kk]; }
        let diag = corr[j * p + j] - sum;
        if diag <= 1e-15 {
            return vec![f64::INFINITY; p]; // not positive definite
        }
        l[j * p + j] = diag.sqrt();
        for i in (j + 1)..p {
            let mut s = 0.0;
            for kk in 0..j { s += l[i * p + kk] * l[j * p + kk]; }
            l[i * p + j] = (corr[i * p + j] - s) / l[j * p + j];
        }
    }

    // Compute L⁻¹ column by column via forward substitution
    let mut l_inv = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            let rhs = if i == j { 1.0 } else { 0.0 };
            let mut sum = rhs;
            for kk in 0..i { sum -= l[i * p + kk] * l_inv[kk * p + j]; }
            l_inv[i * p + j] = sum / l[i * p + i];
        }
    }

    // VIF_j = (C⁻¹)_{jj} = Σ_k (L⁻¹[k,j])²
    let mut vif_result = vec![0.0; p];
    for j in 0..p {
        let mut sum = 0.0;
        for kk in 0..p {
            sum += l_inv[kk * p + j] * l_inv[kk * p + j];
        }
        vif_result[j] = sum;
    }

    vif_result
}

/// Full IRLS (Newton-Raphson) for binary logistic regression.
/// x: design matrix (n × k, row-major, INCLUDING intercept column if needed)
/// y: binary labels (0/1), length n
/// Returns: beta coefficients (length k)
#[wasm_bindgen]
pub fn irls_logistic(
    x: &[f64], y: &[f64],
    n: usize, k: usize,
    max_iter: usize, tolerance: f64,
) -> Vec<f64> {
    let mut beta = vec![0.0; k];

    let mut eta = vec![0.0; n];
    let mut xw = vec![0.0; n * k];
    let mut zw = vec![0.0; n];

    for _iter in 0..max_iter {
        // Compute linear predictor and probabilities
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..k {
                sum += x[i * k + j] * beta[j];
            }
            eta[i] = sum;
        }

        // Build weighted X and z
        for i in 0..n {
            let pi = sigmoid(eta[i]).max(1e-15).min(1.0 - 1e-15);
            let wi = pi * (1.0 - pi);
            let sqrt_w = wi.max(1e-10).sqrt();
            let z = eta[i] + (y[i] - pi) / wi;

            for j in 0..k {
                xw[i * k + j] = x[i * k + j] * sqrt_w;
            }
            zw[i] = z * sqrt_w;
        }

        // Solve WLS via efficient QR
        let beta_new = match qr_solve_internal(&xw, &zw, n, k) {
            Some(b) => b,
            None => break,
        };

        // Check convergence
        let mut max_change = 0.0_f64;
        for j in 0..k {
            max_change = max_change.max((beta_new[j] - beta[j]).abs());
            beta[j] = beta_new[j];
        }

        if max_change < tolerance { break; }
    }

    beta
}

use wasm_bindgen::prelude::*;

/// Matrix multiplication: flat row-major arrays.
/// Returns a flat array of size a_rows * b_cols.
#[wasm_bindgen]
pub fn matrix_multiply(
    a: &[f64], a_rows: usize, a_cols: usize,
    b: &[f64], b_rows: usize, b_cols: usize,
) -> Vec<f64> {
    assert_eq!(a_cols, b_rows, "Incompatible matrix dimensions");
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

/// QR decomposition via Householder reflections.
/// Returns Q (m*m) followed by R (m*n) as a single flat array.
#[wasm_bindgen]
pub fn qr_decompose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let m = rows;
    let n = cols;

    // R = copy of input
    let mut r = data.to_vec();
    // Q = identity
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    let min_mn = m.min(n);
    for k in 0..min_mn {
        // Extract column k from row k downward
        let len = m - k;
        let mut x = vec![0.0; len];
        for i in 0..len {
            x[i] = r[(k + i) * n + k];
        }

        // Compute norm
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if x_norm < 1e-15 { continue; }

        let alpha = if x[0] >= 0.0 { -x_norm } else { x_norm };

        // v = x - alpha * e1
        let mut v = x.clone();
        v[0] -= alpha;

        let v_norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if v_norm < 1e-15 { continue; }
        for vi in v.iter_mut() { *vi /= v_norm; }

        // Apply to R
        for j in k..n {
            let dot: f64 = (0..len).map(|i| v[i] * r[(k + i) * n + j]).sum();
            for i in 0..len {
                r[(k + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        // Apply to Q
        for i in 0..m {
            let dot: f64 = (0..len).map(|j| q[i * m + (k + j)] * v[j]).sum();
            for j in 0..len {
                q[i * m + (k + j)] -= 2.0 * dot * v[j];
            }
        }
    }

    // Return Q followed by R
    let mut result = Vec::with_capacity(m * m + m * n);
    result.extend_from_slice(&q);
    result.extend_from_slice(&r);
    result
}

/// Cholesky decomposition: returns L (lower triangular, n*n flat).
#[wasm_bindgen]
pub fn cholesky(data: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = data[j * n + j] - sum;
        assert!(diag > 0.0, "Matrix is not positive-definite");
        l[j * n + j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (data[i * n + j] - s) / l[j * n + j];
        }
    }
    l
}

/// Back substitution for upper triangular Rx = b.
#[wasm_bindgen]
pub fn solve_triangular(r: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= r[i * n + j] * x[j];
        }
        x[i] = sum / r[i * n + i];
    }
    x
}

/// Forward substitution for lower triangular Lx = b.
#[wasm_bindgen]
pub fn forward_substitution(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * x[j];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Matrix transpose: (rows x cols) -> (cols x rows).
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

/// SVD via one-sided Jacobi rotations.
/// Returns U (m*k) | S (k) | V (n*k) as a single flat array, where k = min(m,n).
#[wasm_bindgen]
pub fn svd(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let m = rows;
    let n = cols;
    let k = m.min(n);

    // W = copy of A (we'll orthogonalize columns)
    let mut w = data.to_vec();
    // V = identity(n)
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100;
    let tol = 1e-12;

    for _iter in 0..max_iter {
        let mut converged = true;

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                // Compute 2x2 sub-problem
                let mut app = 0.0;
                let mut aqq = 0.0;
                let mut apq = 0.0;
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    app += wp * wp;
                    aqq += wq * wq;
                    apq += wp * wq;
                }

                if apq.abs() < tol * (app * aqq).sqrt() {
                    continue;
                }
                converged = false;

                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Rotate W columns
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    w[i * n + p] = c * wp - s * wq;
                    w[i * n + q] = s * wp + c * wq;
                }

                // Rotate V columns
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values and U
    let mut singular_values = vec![0.0; k];
    let mut u = vec![0.0; m * k];
    for j in 0..k {
        let mut norm = 0.0;
        for i in 0..m {
            norm += w[i * n + j] * w[i * n + j];
        }
        norm = norm.sqrt();
        singular_values[j] = norm;
        if norm > 1e-15 {
            for i in 0..m {
                u[i * k + j] = w[i * n + j] / norm;
            }
        }
    }

    // Sort descending by singular value
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|a, b| singular_values[*b].partial_cmp(&singular_values[*a]).unwrap());

    // Pack result: U (m*k) | S (k) | V (n*k)
    let mut result = Vec::with_capacity(m * k + k + n * k);

    // Sorted U
    for i in 0..m {
        for j_idx in &indices {
            result.push(u[i * k + j_idx]);
        }
    }
    // Sorted S
    for j_idx in &indices {
        result.push(singular_values[*j_idx]);
    }
    // Sorted V (extract k columns from n*n)
    for i in 0..n {
        for j_idx in &indices {
            result.push(v[i * n + j_idx]);
        }
    }

    result
}

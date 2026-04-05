/* @ts-self-types="./regressio_wasm.d.ts" */

import * as wasm from "./regressio_wasm_bg.wasm";
import { __wbg_set_wasm } from "./regressio_wasm_bg.js";
__wbg_set_wasm(wasm);
wasm.__wbindgen_start();
export {
    bootstrap_ols, cholesky, coordinate_descent, correlation_matrix, determinant, eigenvalues, euclidean_distances, forward_substitution, frobenius_norm, irls_logistic, manhattan_distances, matrix_add, matrix_multiply, matrix_scale, matrix_subtract, matrix_transpose, qr_decompose, softmax_rows, solve_triangular, svd, vector_dot, vif
} from "./regressio_wasm_bg.js";

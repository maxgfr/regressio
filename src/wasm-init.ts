/**
 * Custom WASM loader that works in Node.js, Bun, and browsers.
 *
 * The wasm-pack `--target bundler` output relies on `import * as wasm from "./foo.wasm"`
 * which only works with webpack/vite. This loader manually reads the .wasm binary,
 * instantiates it via WebAssembly.instantiate(), and wires up the wasm-bindgen bindings.
 */

// @ts-nocheck — bg.js is wasm-pack generated JS without TS types
import {
  __wbg_set_wasm,
  __wbindgen_init_externref_table,
  bootstrap_ols,
  cholesky,
  coordinate_descent,
  correlation_matrix,
  determinant,
  eigenvalues,
  euclidean_distances,
  forward_substitution,
  frobenius_norm,
  irls_logistic,
  manhattan_distances,
  matrix_add,
  matrix_multiply,
  matrix_scale,
  matrix_subtract,
  matrix_transpose,
  qr_decompose,
  softmax_rows,
  solve_triangular,
  svd,
  vector_dot,
  vif,
} from "../pkg/regressio_wasm_bg.js";

// bunup replaces this with a string path to the .wasm file in dist/
// @ts-expect-error — .wasm import resolved by bundler
import wasmPath from "../pkg/regressio_wasm_bg.wasm";

async function loadWasmBytes(): Promise<BufferSource> {
  const resolved = typeof wasmPath === "string" ? wasmPath : null;
  if (!resolved) throw new Error("WASM path not available");

  // Node.js / Bun — read from filesystem
  if (typeof globalThis.process !== "undefined" || typeof globalThis.Bun !== "undefined") {
    const { readFile } = await import("node:fs/promises");
    const { resolve, dirname } = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    const dir = dirname(fileURLToPath(import.meta.url));
    return readFile(resolve(dir, resolved));
  }

  // Browser — fetch
  const response = await fetch(new URL(resolved, import.meta.url));
  return response.arrayBuffer();
}

/** Initialize WASM module. Must be called before using exported functions. */
export async function initWasm(): Promise<void> {
  const bytes = await loadWasmBytes();
  const imports = {
    "./regressio_wasm_bg.js": { __wbindgen_init_externref_table },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  __wbg_set_wasm(instance.exports);
}

// Re-export all WASM functions (usable only after initWasm resolves)
export {
  bootstrap_ols,
  cholesky,
  coordinate_descent,
  correlation_matrix,
  determinant,
  eigenvalues,
  euclidean_distances,
  forward_substitution,
  frobenius_norm,
  irls_logistic,
  manhattan_distances,
  matrix_add,
  matrix_multiply,
  matrix_scale,
  matrix_subtract,
  matrix_transpose,
  qr_decompose,
  softmax_rows,
  solve_triangular,
  svd,
  vector_dot,
  vif,
};

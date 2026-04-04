/**
 * Computation engine abstraction.
 * Default: pure TypeScript. Call useWasmEngine() to switch to WASM backend.
 *
 * The WASM engine accelerates: matrix multiply, QR decomposition, Cholesky, back-substitution.
 * All other operations remain in TypeScript.
 */

export interface WasmModule {
  matrix_multiply(
    a: Float64Array,
    a_rows: number,
    a_cols: number,
    b: Float64Array,
    b_rows: number,
    b_cols: number,
  ): Float64Array;
  qr_decompose(data: Float64Array, rows: number, cols: number): Float64Array;
  cholesky(data: Float64Array, n: number): Float64Array;
  solve_triangular(r: Float64Array, b: Float64Array, n: number): Float64Array;
}

export interface ComputeEngine {
  name: "typescript" | "wasm";
  wasm?: WasmModule;
}

let currentEngine: ComputeEngine = { name: "typescript" };

/** Get the current computation engine. */
export function getEngine(): ComputeEngine {
  return currentEngine;
}

/** Check if the WASM engine is active. */
export function isWasmActive(): boolean {
  return currentEngine.name === "wasm" && currentEngine.wasm != null;
}

/**
 * Switch to the WASM computation engine for faster matrix operations.
 * Requires the regressio WASM package to be built:
 *   cd rust && wasm-pack build --target bundler --out-dir ../pkg
 */
export async function useWasmEngine(): Promise<void> {
  try {
    const wasm = await import("../../pkg");
    currentEngine = { name: "wasm", wasm: wasm as unknown as WasmModule };
  } catch {
    throw new Error(
      "WASM engine not available. Build it with: cd rust && wasm-pack build --target bundler --out-dir ../pkg",
    );
  }
}

/**
 * Load WASM engine from a pre-loaded module (useful for custom bundler setups).
 */
export function useWasmModule(wasmModule: WasmModule): void {
  currentEngine = { name: "wasm", wasm: wasmModule };
}

/** Reset to the default TypeScript engine. */
export function useTypescriptEngine(): void {
  currentEngine = { name: "typescript" };
}

// ---------------------------------------------------------------------------
// Engine-dispatched operations (used by Matrix and decompositions)
// ---------------------------------------------------------------------------

/** Matrix multiply: dispatches to WASM if available, else TypeScript. */
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
  // TypeScript fallback (ikj loop)
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

/** QR decomposition: dispatches to WASM if available. Returns { Q, R } as flat arrays. */
export function engineQR(
  data: Float64Array,
  rows: number,
  cols: number,
): { Q: Float64Array; R: Float64Array } {
  if (currentEngine.wasm) {
    const result = currentEngine.wasm.qr_decompose(data, rows, cols);
    const qSize = rows * rows;
    const Q = result.slice(0, qSize);
    const R = result.slice(qSize);
    return { Q, R };
  }
  // Return null to signal "use TypeScript implementation"
  return null as unknown as { Q: Float64Array; R: Float64Array };
}

/** Cholesky: dispatches to WASM if available. Returns L as flat array, or null. */
export function engineCholesky(data: Float64Array, n: number): Float64Array | null {
  if (currentEngine.wasm) {
    return currentEngine.wasm.cholesky(data, n);
  }
  return null;
}

/** Back-substitution: dispatches to WASM if available. */
export function engineSolveTriangular(
  r: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array | null {
  if (currentEngine.wasm) {
    return currentEngine.wasm.solve_triangular(r, b, n);
  }
  return null;
}

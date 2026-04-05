import { engineBootstrapOLS } from "../core/engine";
import { LinearRegression } from "../models/linear-regression";
import type { BootstrapResult, DataMatrix, DataVector } from "../types";

/**
 * Bootstrap confidence intervals on regression coefficients.
 * Resamples with replacement, refits the model nBootstrap times.
 * Returns mean coefficients, empirical CIs (percentile method), and bootstrap SEs.
 */
export function bootstrapCoefficients(
  X: DataMatrix,
  y: DataVector,
  nBootstrap = 1000,
  alpha = 0.05,
  fitIntercept = true,
): BootstrapResult {
  const n = X.length;
  const p = X[0]!.length;
  const nParams = p + (fitIntercept ? 1 : 0);

  // WASM fast path — all bootstrap iterations run in Rust
  const xFlat = new Float64Array(n * p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) xFlat[i * p + j] = X[i]![j]!;
  const wasmResult = engineBootstrapOLS(
    xFlat,
    new Float64Array(y),
    n,
    p,
    fitIntercept,
    nBootstrap,
    (Math.random() * 0xffffffff) >>> 0,
  );
  if (wasmResult) {
    return aggregateBootstrap(wasmResult, nBootstrap, nParams, alpha);
  }

  // TypeScript fallback
  const allCoeffs: number[][] = Array.from({ length: nParams }, () => []);

  for (let b = 0; b < nBootstrap; b++) {
    const indices = Array.from({ length: n }, () => Math.floor(Math.random() * n));
    const Xb = indices.map((i) => X[i]!);
    const yb = indices.map((i) => y[i]!);

    try {
      const model = new LinearRegression({ fitIntercept });
      model.fit(Xb, yb);

      const coeffs = fitIntercept
        ? [model.intercept, ...model.coefficients]
        : [...model.coefficients];

      for (let j = 0; j < nParams; j++) {
        allCoeffs[j]!.push(coeffs[j]!);
      }
    } catch {}
  }

  return aggregateBootstrapArrays(allCoeffs, nParams, alpha);
}

/** Aggregate WASM bootstrap results (flat nBootstrap × nParams, NaN = singular). */
function aggregateBootstrap(
  flat: Float64Array,
  nBootstrap: number,
  nParams: number,
  alpha: number,
): BootstrapResult {
  const allCoeffs: number[][] = Array.from({ length: nParams }, () => []);

  for (let b = 0; b < nBootstrap; b++) {
    const val = flat[b * nParams]!;
    if (Number.isNaN(val)) continue; // singular sample
    for (let j = 0; j < nParams; j++) {
      allCoeffs[j]!.push(flat[b * nParams + j]!);
    }
  }

  return aggregateBootstrapArrays(allCoeffs, nParams, alpha);
}

/** Shared aggregation: mean, SE, percentile CI. */
function aggregateBootstrapArrays(
  allCoeffs: number[][],
  nParams: number,
  alpha: number,
): BootstrapResult {
  const coefficients: number[] = [];
  const confidenceIntervals: [number, number][] = [];
  const standardErrors: number[] = [];

  for (let j = 0; j < nParams; j++) {
    const samples = allCoeffs[j]!;
    if (samples.length === 0) {
      coefficients.push(0);
      confidenceIntervals.push([0, 0]);
      standardErrors.push(0);
      continue;
    }

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    coefficients.push(mean);

    const variance = samples.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (samples.length - 1);
    standardErrors.push(Math.sqrt(variance));

    const sorted = [...samples].sort((a, b) => a - b);
    const loIdx = Math.floor((alpha / 2) * sorted.length);
    const hiIdx = Math.floor((1 - alpha / 2) * sorted.length);
    confidenceIntervals.push([sorted[loIdx]!, sorted[Math.min(hiIdx, sorted.length - 1)]!]);
  }

  return { coefficients, confidenceIntervals, standardErrors };
}

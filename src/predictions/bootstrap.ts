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

  // Collect bootstrap samples
  const allCoeffs: number[][] = Array.from({ length: nParams }, () => []);

  for (let b = 0; b < nBootstrap; b++) {
    // Resample with replacement
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

  // Compute results
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

    // Mean
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    coefficients.push(mean);

    // Standard error
    const variance = samples.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (samples.length - 1);
    standardErrors.push(Math.sqrt(variance));

    // Percentile CI
    const sorted = [...samples].sort((a, b) => a - b);
    const loIdx = Math.floor((alpha / 2) * sorted.length);
    const hiIdx = Math.floor((1 - alpha / 2) * sorted.length);
    confidenceIntervals.push([sorted[loIdx]!, sorted[Math.min(hiIdx, sorted.length - 1)]!]);
  }

  return { coefficients, confidenceIntervals, standardErrors };
}

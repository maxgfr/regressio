import type { DataMatrix, NormalizationParams, ScalingParams } from "../types";

/** Standardize (z-score): mean=0, std=1. Returns transformed data + params. */
export function standardize(X: DataMatrix): { transformed: DataMatrix } & ScalingParams {
  const n = X.length;
  const p = X[0]!.length;
  const means: number[] = [];
  const stds: number[] = [];

  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i]![j]!;
    const mean = sum / n;
    means.push(mean);

    let sqSum = 0;
    for (let i = 0; i < n; i++) sqSum += (X[i]![j]! - mean) ** 2;
    stds.push(Math.sqrt(sqSum / n));
  }

  const transformed = X.map((row) =>
    row.map((val, j) => {
      const std = stds[j]!;
      return std > 1e-15 ? (val - means[j]!) / std : 0;
    }),
  );

  return { transformed, means, stds };
}

/** Inverse of standardize: restore original scale. */
export function unstandardize(X: DataMatrix, params: ScalingParams): DataMatrix {
  return X.map((row) => row.map((val, j) => val * params.stds[j]! + params.means[j]!));
}

/** Normalize (min-max) to [0,1]. Returns transformed data + params. */
export function normalize(X: DataMatrix): { transformed: DataMatrix } & NormalizationParams {
  const n = X.length;
  const p = X[0]!.length;
  const mins: number[] = [];
  const maxs: number[] = [];

  for (let j = 0; j < p; j++) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const val = X[i]![j]!;
      if (val < min) min = val;
      if (val > max) max = val;
    }
    mins.push(min);
    maxs.push(max);
  }

  const transformed = X.map((row) =>
    row.map((val, j) => {
      const range = maxs[j]! - mins[j]!;
      return range > 1e-15 ? (val - mins[j]!) / range : 0;
    }),
  );

  return { transformed, mins, maxs };
}

/** Inverse of normalize: restore original scale. */
export function unnormalize(X: DataMatrix, params: NormalizationParams): DataMatrix {
  return X.map((row) =>
    row.map((val, j) => val * (params.maxs[j]! - params.mins[j]!) + params.mins[j]!),
  );
}

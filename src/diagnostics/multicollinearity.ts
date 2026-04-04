import { svd } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import { LinearRegression } from "../models/linear-regression";
import type { DataMatrix } from "../types";

/**
 * Compute Variance Inflation Factor for each feature.
 * VIF_j = 1 / (1 - R²_j) where R²_j is from regressing x_j on all other x's.
 * VIF > 10 signals multicollinearity.
 */
export function vif(X: DataMatrix): number[] {
  const p = X[0]!.length;
  if (p < 2) return [1];

  const result: number[] = [];

  for (let j = 0; j < p; j++) {
    // Build y = column j, X_other = all other columns
    const yCol = X.map((row) => row[j]!);
    const XOther = X.map((row) => row.filter((_, idx) => idx !== j));

    const model = new LinearRegression();
    model.fit(XOther, yCol);
    const stats = model.statistics();
    const r2 = stats.rSquared;

    result.push(1 / Math.max(1 - r2, 1e-15));
  }

  return result;
}

/**
 * Compute pairwise Pearson correlation matrix.
 * Returns p×p matrix of correlations.
 */
export function correlationMatrix(X: DataMatrix): number[][] {
  const n = X.length;
  const p = X[0]!.length;

  // Compute means
  const means: number[] = [];
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i]![j]!;
    means.push(sum / n);
  }

  // Compute stds
  const stds: number[] = [];
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const diff = X[i]![j]! - means[j]!;
      sum += diff * diff;
    }
    stds.push(Math.sqrt(sum / n));
  }

  // Compute correlation matrix
  const corr: number[][] = [];
  for (let j1 = 0; j1 < p; j1++) {
    const row: number[] = [];
    for (let j2 = 0; j2 < p; j2++) {
      if (j1 === j2) {
        row.push(1);
      } else {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          sum +=
            ((X[i]![j1]! - means[j1]!) / Math.max(stds[j1]!, 1e-15)) *
            ((X[i]![j2]! - means[j2]!) / Math.max(stds[j2]!, 1e-15));
        }
        row.push(sum / n);
      }
    }
    corr.push(row);
  }

  return corr;
}

/**
 * Compute the condition number of X: κ = σ_max / σ_min.
 * κ > 30 signals potential numerical instability from multicollinearity.
 */
export function conditionNumber(X: DataMatrix): number {
  const A = Matrix.fromArray(X);
  const { S } = svd(A);
  const sMax = S[0] ?? 1;
  const sMin = S[S.length - 1] ?? 0;
  return sMin > 1e-15 ? sMax / sMin : Infinity;
}

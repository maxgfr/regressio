import { qrDecomposition } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import type { DataMatrix, DataVector, ResidualDiagnostics } from "../types";

/**
 * Compute residual diagnostics for a fitted linear model.
 * @param X Feature matrix (without intercept column)
 * @param y Response vector
 * @param yHat Predicted values
 * @param fitIntercept Whether the model included an intercept
 */
export function residualDiagnostics(
  X: DataMatrix,
  y: DataVector,
  yHat: DataVector,
  fitIntercept = true,
): ResidualDiagnostics {
  const n = y.length;
  const raw = y.map((yi, i) => yi - yHat[i]!);

  // Build design matrix
  const Xdesign = fitIntercept ? X.map((row) => [1, ...row]) : X;
  const A = Matrix.fromArray(Xdesign);
  const k = A.cols;

  // Hat matrix diagonal: h_ii = X_i (X^T X)^{-1} X_i^T
  // Compute via QR: H = Q Q^T, so h_ii = sum(Q[i,j]^2) for j=0..k-1
  const { Q } = qrDecomposition(A);
  const leverageValues: number[] = [];
  for (let i = 0; i < n; i++) {
    let h = 0;
    for (let j = 0; j < k; j++) {
      h += Q.get(i, j) * Q.get(i, j);
    }
    leverageValues.push(h);
  }

  // Residual standard error (leave-one-out)
  let rss = 0;
  for (const r of raw) rss += r * r;
  const s = Math.sqrt(rss / (n - k));

  // Studentized residuals: e_i* = e_i / (s * sqrt(1 - h_ii))
  const studentized = raw.map((r, i) => {
    const h = leverageValues[i]!;
    const denom = s * Math.sqrt(Math.max(1 - h, 1e-15));
    return r / denom;
  });

  // Cook's distance: D_i = (e_i*^2 * h_ii) / (k * (1 - h_ii))
  const cooksDistance = studentized.map((eStar, i) => {
    const h = leverageValues[i]!;
    return (eStar * eStar * h) / (k * Math.max(1 - h, 1e-15));
  });

  return {
    raw,
    studentized,
    cooksDistance,
    leverage: leverageValues,
  };
}

/** Compute just the leverage (hat matrix diagonal) values. */
export function leverage(X: DataMatrix, fitIntercept = true): number[] {
  const Xdesign = fitIntercept ? X.map((row) => [1, ...row]) : X;
  const A = Matrix.fromArray(Xdesign);
  const k = A.cols;
  const { Q } = qrDecomposition(A);
  const result: number[] = [];
  for (let i = 0; i < A.rows; i++) {
    let h = 0;
    for (let j = 0; j < k; j++) {
      h += Q.get(i, j) * Q.get(i, j);
    }
    result.push(h);
  }
  return result;
}

/** Compute Cook's distance for each observation. */
export function cooksDistance(
  X: DataMatrix,
  y: DataVector,
  yHat: DataVector,
  fitIntercept = true,
): number[] {
  return residualDiagnostics(X, y, yHat, fitIntercept).cooksDistance;
}

/** Compute studentized residuals. */
export function studentizedResiduals(
  X: DataMatrix,
  y: DataVector,
  yHat: DataVector,
  fitIntercept = true,
): number[] {
  return residualDiagnostics(X, y, yHat, fitIntercept).studentized;
}

import { backSubstitution, qrDecomposition } from "../core/decompositions";
import { tInverseCDF } from "../core/distributions";
import { Matrix } from "../core/matrix";
import type { DataMatrix, DataVector, PredictionInterval } from "../types";

/**
 * Compute confidence intervals on ŷ (uncertainty on the mean prediction).
 * CI: ŷ ± t_{α/2,df} × s × √(x₀ᵀ (XᵀX)⁻¹ x₀)
 */
export function confidenceInterval(
  X: DataMatrix,
  y: DataVector,
  yHat: DataVector,
  newX: DataMatrix,
  newYHat: DataVector,
  fitIntercept = true,
  alpha = 0.05,
): PredictionInterval[] {
  const { s, XtXinv, df } = computeBase(X, y, yHat, fitIntercept);
  const tCrit = tInverseCDF(1 - alpha / 2, df);

  return newX.map((row, idx) => {
    const x0 = fitIntercept ? [1, ...row] : row;
    const x0vec = Matrix.columnVector(x0);
    const se = s * Math.sqrt(quadForm(XtXinv, x0vec));
    const predicted = newYHat[idx]!;
    return {
      predicted,
      lower: predicted - tCrit * se,
      upper: predicted + tCrit * se,
    };
  });
}

/**
 * Compute prediction intervals on ŷ (uncertainty on a new individual observation).
 * PI: ŷ ± t_{α/2,df} × √(s² + s² × x₀ᵀ (XᵀX)⁻¹ x₀)
 * Always wider than confidence interval.
 */
export function predictionInterval(
  X: DataMatrix,
  y: DataVector,
  yHat: DataVector,
  newX: DataMatrix,
  newYHat: DataVector,
  fitIntercept = true,
  alpha = 0.05,
): PredictionInterval[] {
  const { s, XtXinv, df } = computeBase(X, y, yHat, fitIntercept);
  const tCrit = tInverseCDF(1 - alpha / 2, df);

  return newX.map((row, idx) => {
    const x0 = fitIntercept ? [1, ...row] : row;
    const x0vec = Matrix.columnVector(x0);
    const se = s * Math.sqrt(1 + quadForm(XtXinv, x0vec));
    const predicted = newYHat[idx]!;
    return {
      predicted,
      lower: predicted - tCrit * se,
      upper: predicted + tCrit * se,
    };
  });
}

// Shared computation
function computeBase(X: DataMatrix, y: DataVector, yHat: DataVector, fitIntercept: boolean) {
  const n = y.length;
  const Xdesign = fitIntercept ? X.map((row) => [1, ...row]) : X;
  const A = Matrix.fromArray(Xdesign);
  const k = A.cols;
  const df = n - k;

  // Residual standard error
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i]! - yHat[i]!;
    rss += r * r;
  }
  const s = Math.sqrt(rss / df);

  // (X^T X)^{-1} via QR
  const { R } = qrDecomposition(A);
  const Rsq = R.submatrix(0, k, 0, k);
  const Rinv = Matrix.zeros(k, k);
  for (let j = 0; j < k; j++) {
    const ej = Matrix.zeros(k, 1);
    ej.set(j, 0, 1);
    const col = backSubstitution(Rsq, ej);
    for (let i = 0; i < k; i++) {
      Rinv.set(i, j, col.get(i, 0));
    }
  }
  const XtXinv = Rinv.multiply(Rinv.transpose());

  return { s, XtXinv, k, df };
}

function quadForm(M: Matrix, v: Matrix): number {
  const Mv = M.multiply(v);
  return v.dot(Mv);
}

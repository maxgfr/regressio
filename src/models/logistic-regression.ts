import { qrDecomposition, solveQR } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import type {
  ClassificationStatistics,
  DataInput,
  DataMatrix,
  DataVector,
  LogisticOptions,
} from "../types";

export class LogisticRegression {
  private _coefficients: number[] = [];
  private _intercept = 0;
  private _fitted = false;
  private _fitIntercept: boolean;
  private _maxIterations: number;
  private _tolerance: number;
  private _y: DataVector = [];
  private _probabilities: DataVector = [];

  constructor(options: LogisticOptions = {}) {
    this._fitIntercept = options.fitIntercept ?? true;
    this._maxIterations = options.maxIterations ?? 100;
    this._tolerance = options.tolerance ?? 1e-6;
  }

  get coefficients(): number[] {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    return this._coefficients;
  }

  get intercept(): number {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    return this._intercept;
  }

  private sigmoid(z: number): number {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const expZ = Math.exp(z);
    return expZ / (1 + expZ);
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    if (Xmat.length !== y.length) {
      throw new Error(`X has ${Xmat.length} rows but y has ${y.length} elements`);
    }

    // Validate y is binary
    for (const yi of y) {
      if (yi !== 0 && yi !== 1) {
        throw new Error("Logistic regression requires binary y (0 or 1)");
      }
    }

    this._X = Xmat;
    this._y = y;

    const n = Xmat.length;
    const Xdesign = this._fitIntercept ? this.addInterceptColumn(Xmat) : Xmat;
    const k = Xdesign[0]!.length;

    // Initialize beta to zeros
    const beta = new Float64Array(k);

    // IRLS (Newton-Raphson)
    for (let iter = 0; iter < this._maxIterations; iter++) {
      // Compute linear predictor and probabilities
      const eta = new Float64Array(n);
      const p = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < k; j++) {
          sum += Xdesign[i]![j]! * beta[j]!;
        }
        eta[i] = sum;
        p[i] = this.sigmoid(sum);
      }

      // Compute weights W = p(1-p) and working response z
      const sqrtW = new Float64Array(n);
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const pi = Math.max(1e-15, Math.min(1 - 1e-15, p[i]!));
        const wi = pi * (1 - pi);
        sqrtW[i] = Math.sqrt(Math.max(wi, 1e-10));
        z[i] = eta[i]! + (y[i]! - pi) / wi;
      }

      // Weighted X and z
      const XW = Matrix.zeros(n, k);
      const zW = Matrix.zeros(n, 1);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
          XW.set(i, j, Xdesign[i]![j]! * sqrtW[i]!);
        }
        zW.set(i, 0, z[i]! * sqrtW[i]!);
      }

      // Solve weighted least squares
      const { Q, R } = qrDecomposition(XW);
      const betaNew = solveQR(Q, R, zW);

      // Check convergence
      let maxChange = 0;
      for (let j = 0; j < k; j++) {
        maxChange = Math.max(maxChange, Math.abs(betaNew.get(j, 0) - beta[j]!));
        beta[j] = betaNew.get(j, 0);
      }

      if (maxChange < this._tolerance) break;
    }

    // Extract coefficients
    const betaArray = Array.from(beta);
    if (this._fitIntercept) {
      this._intercept = betaArray[0]!;
      this._coefficients = betaArray.slice(1);
    } else {
      this._intercept = 0;
      this._coefficients = betaArray;
    }

    // Store probabilities
    this._probabilities = this.predictProbability(Xmat);
    this._fitted = true;
    return this;
  }

  predict(X: DataInput): DataVector {
    return this.predictProbability(X).map((p) => (p >= 0.5 ? 1 : 0));
  }

  predictProbability(X: DataInput): DataVector {
    const Xmat = this.normalizeInput(X);
    return Xmat.map((row) => {
      let sum = this._intercept;
      for (let j = 0; j < this._coefficients.length; j++) {
        sum += row[j]! * this._coefficients[j]!;
      }
      return this.sigmoid(sum);
    });
  }

  statistics(): ClassificationStatistics {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");

    const n = this._y.length;
    const predicted = this._probabilities.map((p) => (p >= 0.5 ? 1 : 0));

    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < n; i++) {
      if (this._y[i] === 1 && predicted[i] === 1) tp++;
      else if (this._y[i] === 0 && predicted[i] === 0) tn++;
      else if (this._y[i] === 0 && predicted[i] === 1) fp++;
      else fn++;
    }

    const accuracy = (tp + tn) / n;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    // Log-likelihood
    let logLik = 0;
    for (let i = 0; i < n; i++) {
      const pi = Math.max(1e-15, Math.min(1 - 1e-15, this._probabilities[i]!));
      logLik += this._y[i]! * Math.log(pi) + (1 - this._y[i]!) * Math.log(1 - pi);
    }

    // Null log-likelihood
    let yMean = 0;
    for (const yi of this._y) yMean += yi;
    yMean /= n;
    const pBar = Math.max(1e-15, Math.min(1 - 1e-15, yMean));
    const nullLogLik = n * (pBar * Math.log(pBar) + (1 - pBar) * Math.log(1 - pBar));

    const pseudoRSquared = 1 - logLik / nullLogLik;
    const k = this._coefficients.length + (this._fitIntercept ? 1 : 0);
    const aic = -2 * logLik + 2 * k;
    const bic = -2 * logLik + Math.log(n) * k;

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix: {
        truePositives: tp,
        trueNegatives: tn,
        falsePositives: fp,
        falseNegatives: fn,
      },
      pseudoRSquared,
      logLikelihood: logLik,
      aic,
      bic,
    };
  }

  private normalizeInput(X: DataInput): DataMatrix {
    if (X.length === 0) throw new Error("Input data cannot be empty");
    if (typeof X[0] === "number") {
      return (X as number[]).map((v) => [v]);
    }
    return X as DataMatrix;
  }

  private addInterceptColumn(X: DataMatrix): DataMatrix {
    return X.map((row) => [1, ...row]);
  }
}

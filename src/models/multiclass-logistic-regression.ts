import { engineSoftmaxRows } from "../core/engine";
import { Matrix } from "../core/matrix";
import type { DataInput, DataMatrix, DataVector } from "../types";

export interface MulticlassLogisticOptions {
  /** Whether to fit an intercept term (default: true). */
  fitIntercept?: boolean;
  /** Maximum iterations for gradient descent (default: 200). */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-6). */
  tolerance?: number;
  /** Learning rate (default: 0.1). */
  learningRate?: number;
}

export interface MulticlassStatistics {
  accuracy: number;
  /** Per-class precision. */
  precision: number[];
  /** Per-class recall. */
  recall: number[];
  /** Number of classes. */
  nClasses: number;
  /** Log-likelihood. */
  logLikelihood: number;
}

/**
 * Multiclass Logistic Regression (multinomial) via softmax + gradient descent.
 * Supports any number of classes (labels must be integers 0, 1, ..., K-1).
 */
export class MulticlassLogisticRegression {
  private _weights: Matrix = Matrix.zeros(0, 0); // (k x nClasses) weight matrix
  private _fitted = false;
  private _fitIntercept: boolean;
  private _maxIterations: number;
  private _tolerance: number;
  private _learningRate: number;
  private _nClasses = 0;
  private _classes: number[] = [];

  private _X: DataMatrix = [];
  private _y: DataVector = [];

  constructor(options: MulticlassLogisticOptions = {}) {
    this._fitIntercept = options.fitIntercept ?? true;
    this._maxIterations = options.maxIterations ?? 200;
    this._tolerance = options.tolerance ?? 1e-6;
    this._learningRate = options.learningRate ?? 0.1;
  }

  get weights(): Matrix {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    return this._weights;
  }

  get classes(): number[] {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    return this._classes;
  }

  /** Softmax: converts raw scores to probabilities. */
  private softmax(logits: number[]): number[] {
    const max = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    if (Xmat.length !== y.length) {
      throw new Error(`X has ${Xmat.length} rows but y has ${y.length} elements`);
    }

    this._X = Xmat;
    this._y = y;

    // Detect classes
    this._classes = [...new Set(y)].sort((a, b) => a - b);
    this._nClasses = this._classes.length;

    // Validate labels are 0..K-1
    for (const yi of y) {
      if (!this._classes.includes(yi)) {
        throw new Error(`Unexpected label ${yi}`);
      }
    }

    const n = Xmat.length;
    const Xdesign = this._fitIntercept ? Xmat.map((row) => [1, ...row]) : Xmat;
    const k = Xdesign[0]!.length;
    const K = this._nClasses;

    // One-hot encode y: (n x K)
    const Y = Matrix.zeros(n, K);
    for (let i = 0; i < n; i++) {
      const classIdx = this._classes.indexOf(y[i]!);
      Y.set(i, classIdx, 1);
    }

    // Initialize weights: (k x K)
    this._weights = Matrix.zeros(k, K);

    const XMat = Matrix.fromArray(Xdesign);

    // Gradient descent
    for (let iter = 0; iter < this._maxIterations; iter++) {
      // Compute probabilities: P = softmax(X * W), shape (n x K)
      const scores = XMat.multiply(this._weights); // (n x K)
      const wasmP = engineSoftmaxRows(scores.data, n, K);
      let P: Matrix;
      if (wasmP) {
        P = new Matrix(n, K, wasmP);
      } else {
        P = Matrix.zeros(n, K);
        for (let i = 0; i < n; i++) {
          const logits: number[] = [];
          for (let c = 0; c < K; c++) logits.push(scores.get(i, c));
          const probs = this.softmax(logits);
          for (let c = 0; c < K; c++) P.set(i, c, probs[c]!);
        }
      }

      // Gradient: X^T (P - Y) / n
      const diff = P.subtract(Y); // (n x K)
      const grad = XMat.transpose()
        .multiply(diff)
        .scale(1 / n); // (k x K)

      // Update weights
      let maxChange = 0;
      for (let i = 0; i < grad.data.length; i++) {
        const delta = this._learningRate * grad.data[i]!;
        this._weights.data[i] -= delta;
        maxChange = Math.max(maxChange, Math.abs(delta));
      }

      if (maxChange < this._tolerance) break;
    }

    this._fitted = true;
    return this;
  }

  /** Predict class labels. */
  predict(X: DataInput): DataVector {
    const probs = this.predictProbability(X);
    return probs.map((row) => {
      let maxIdx = 0;
      let maxVal = row[0]!;
      for (let c = 1; c < row.length; c++) {
        if (row[c]! > maxVal) {
          maxVal = row[c]!;
          maxIdx = c;
        }
      }
      return this._classes[maxIdx]!;
    });
  }

  /** Predict class probabilities: returns array of probability vectors. */
  predictProbability(X: DataInput): number[][] {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    const Xmat = this.normalizeInput(X);
    const Xdesign = this._fitIntercept ? Xmat.map((row) => [1, ...row]) : Xmat;
    const XMat = Matrix.fromArray(Xdesign);
    const scores = XMat.multiply(this._weights);
    const K = this._nClasses;

    const result: number[][] = [];
    for (let i = 0; i < Xmat.length; i++) {
      const logits: number[] = [];
      for (let c = 0; c < K; c++) logits.push(scores.get(i, c));
      result.push(this.softmax(logits));
    }
    return result;
  }

  statistics(): MulticlassStatistics {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");

    const predicted = this.predict(this._X);
    const n = this._y.length;
    const K = this._nClasses;

    // Accuracy
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (predicted[i] === this._y[i]) correct++;
    }

    // Per-class precision and recall
    const precision: number[] = [];
    const recall: number[] = [];
    for (let c = 0; c < K; c++) {
      const cls = this._classes[c]!;
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (let i = 0; i < n; i++) {
        if (predicted[i] === cls && this._y[i] === cls) tp++;
        else if (predicted[i] === cls && this._y[i] !== cls) fp++;
        else if (predicted[i] !== cls && this._y[i] === cls) fn++;
      }
      precision.push(tp + fp > 0 ? tp / (tp + fp) : 0);
      recall.push(tp + fn > 0 ? tp / (tp + fn) : 0);
    }

    // Log-likelihood
    const probs = this.predictProbability(this._X);
    let logLik = 0;
    for (let i = 0; i < n; i++) {
      const classIdx = this._classes.indexOf(this._y[i]!);
      logLik += Math.log(Math.max(probs[i]![classIdx]!, 1e-15));
    }

    return {
      accuracy: correct / n,
      precision,
      recall,
      nClasses: K,
      logLikelihood: logLik,
    };
  }

  private normalizeInput(X: DataInput): DataMatrix {
    if (X.length === 0) throw new Error("Input data cannot be empty");
    if (typeof X[0] === "number") {
      return (X as number[]).map((v) => [v]);
    }
    return X as DataMatrix;
  }
}

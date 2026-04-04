import { engineEuclideanDistances, engineManhattanDistances } from "../core/engine";
import type { DataInput, DataMatrix, DataVector } from "../types";

export interface KNNOptions {
  /** Number of neighbors (default: 5). */
  k?: number;
  /** Distance metric (default: "euclidean"). */
  distance?: "euclidean" | "manhattan";
  /** Mode: "classification" predicts class labels, "regression" predicts mean of neighbors (default: "classification"). */
  mode?: "classification" | "regression";
}

/**
 * K-Nearest Neighbors for classification and regression.
 * Stores training data and computes distances at prediction time.
 */
export class KNearestNeighbors {
  private _k: number;
  private _distance: "euclidean" | "manhattan";
  private _mode: "classification" | "regression";
  private _fitted = false;
  private _X: DataMatrix = [];
  private _y: DataVector = [];

  constructor(options: KNNOptions = {}) {
    this._k = options.k ?? 5;
    this._distance = options.distance ?? "euclidean";
    this._mode = options.mode ?? "classification";
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    if (Xmat.length !== y.length) {
      throw new Error(`X has ${Xmat.length} rows but y has ${y.length} elements`);
    }
    if (Xmat.length < this._k) {
      throw new Error(`Need at least k=${this._k} samples, got ${Xmat.length}`);
    }
    this._X = Xmat;
    this._y = y;
    this._fitted = true;
    return this;
  }

  predict(X: DataInput): DataVector {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    const Xmat = this.normalizeInput(X);

    // WASM batch path — compute all distances at once
    const dim = this._X[0]!.length;
    const nTrain = this._X.length;
    const nTest = Xmat.length;

    const trainFlat = new Float64Array(nTrain * dim);
    for (let i = 0; i < nTrain; i++)
      for (let j = 0; j < dim; j++) trainFlat[i * dim + j] = this._X[i]![j]!;

    const testFlat = new Float64Array(nTest * dim);
    for (let i = 0; i < nTest; i++)
      for (let j = 0; j < dim; j++) testFlat[i * dim + j] = Xmat[i]![j]!;

    const distMatrix =
      this._distance === "manhattan"
        ? engineManhattanDistances(trainFlat, testFlat, nTrain, nTest, dim)
        : engineEuclideanDistances(trainFlat, testFlat, nTrain, nTest, dim);

    if (distMatrix) {
      return Array.from({ length: nTest }, (_, i) => {
        // Find k nearest from this row of the distance matrix
        const indices = Array.from({ length: nTrain }, (_, j) => j)
          .sort((a, b) => distMatrix[i * nTrain + a]! - distMatrix[i * nTrain + b]!)
          .slice(0, this._k);
        return this.voteOrMean(indices);
      });
    }

    // TypeScript fallback
    return Xmat.map((row) => this.predictOne(row));
  }

  /** Return the k nearest neighbor indices for a single point. */
  neighbors(point: number[]): number[] {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    return this.findNeighbors(point).map((n) => n.index);
  }

  private voteOrMean(indices: number[]): number {
    if (this._mode === "regression") {
      let sum = 0;
      for (const idx of indices) sum += this._y[idx]!;
      return sum / indices.length;
    }
    const votes = new Map<number, number>();
    for (const idx of indices) {
      const label = this._y[idx]!;
      votes.set(label, (votes.get(label) ?? 0) + 1);
    }
    let bestLabel = 0;
    let bestCount = 0;
    for (const [label, count] of votes) {
      if (count > bestCount) {
        bestCount = count;
        bestLabel = label;
      }
    }
    return bestLabel;
  }

  private predictOne(point: number[]): number {
    const nearest = this.findNeighbors(point);
    return this.voteOrMean(nearest.map((n) => n.index));
  }

  private findNeighbors(point: number[]): { index: number; distance: number }[] {
    const distances: { index: number; distance: number }[] = [];

    for (let i = 0; i < this._X.length; i++) {
      const d = this.computeDistance(point, this._X[i]!);
      distances.push({ index: i, distance: d });
    }

    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, this._k);
  }

  private computeDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i]! - b[i]!;
      if (this._distance === "manhattan") {
        sum += Math.abs(diff);
      } else {
        sum += diff * diff;
      }
    }
    return this._distance === "manhattan" ? sum : Math.sqrt(sum);
  }

  private normalizeInput(X: DataInput): DataMatrix {
    if (X.length === 0) throw new Error("Input data cannot be empty");
    if (typeof X[0] === "number") {
      return (X as number[]).map((v) => [v]);
    }
    return X as DataMatrix;
  }
}

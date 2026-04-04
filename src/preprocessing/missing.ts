import type { DataMatrix, DataVector } from "../types";

/** Remove rows containing NaN/null/undefined. */
export function dropMissing(X: DataMatrix, y?: DataVector): { X: DataMatrix; y?: DataVector } {
  const validIndices: number[] = [];
  for (let i = 0; i < X.length; i++) {
    const rowValid = X[i]!.every((v) => v != null && !Number.isNaN(v));
    const yValid = y ? y[i] != null && !Number.isNaN(y[i]) : true;
    if (rowValid && yValid) validIndices.push(i);
  }

  return {
    X: validIndices.map((i) => X[i]!),
    y: y ? validIndices.map((i) => y[i]!) : undefined,
  };
}

/** Replace NaN values with column means. */
export function imputeMean(X: DataMatrix): DataMatrix {
  const n = X.length;
  const p = X[0]!.length;
  const means: number[] = [];

  for (let j = 0; j < p; j++) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const val = X[i]![j]!;
      if (!Number.isNaN(val)) {
        sum += val;
        count++;
      }
    }
    means.push(count > 0 ? sum / count : 0);
  }

  return X.map((row) => row.map((val, j) => (Number.isNaN(val) ? means[j]! : val)));
}

/** Replace NaN values with column medians. */
export function imputeMedian(X: DataMatrix): DataMatrix {
  const n = X.length;
  const p = X[0]!.length;
  const medians: number[] = [];

  for (let j = 0; j < p; j++) {
    const values = [];
    for (let i = 0; i < n; i++) {
      const val = X[i]![j]!;
      if (!Number.isNaN(val)) values.push(val);
    }
    values.sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    medians.push(
      values.length === 0
        ? 0
        : values.length % 2 === 0
          ? (values[mid - 1]! + values[mid]!) / 2
          : values[mid]!,
    );
  }

  return X.map((row) => row.map((val, j) => (Number.isNaN(val) ? medians[j]! : val)));
}

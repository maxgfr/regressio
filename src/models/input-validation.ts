import type { DataInput, DataMatrix, DataVector } from "../types";

export function normalizeModelInput(X: DataInput): DataMatrix {
  if (X.length === 0) throw new Error("Input data cannot be empty");

  const matrix =
    typeof X[0] === "number" ? (X as number[]).map((value) => [value]) : (X as DataMatrix);
  const columns = matrix[0]?.length ?? 0;

  if (columns === 0) throw new Error("Input data must contain at least one feature");

  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i]!;
    if (row.length !== columns) {
      throw new Error(`Row ${i} has ${row.length} columns, expected ${columns}`);
    }
    if (!row.every(Number.isFinite)) {
      throw new Error("Model inputs must contain only finite numbers");
    }
  }

  return matrix;
}

export function validateTargets(y: DataVector, expectedRows: number): void {
  if (y.length !== expectedRows) {
    throw new Error(`X has ${expectedRows} rows but y has ${y.length} elements`);
  }
  if (!y.every(Number.isFinite)) {
    throw new Error("Model targets must contain only finite numbers");
  }
}

export function validateFeatureCount(X: DataMatrix, expectedFeatures: number): void {
  const actualFeatures = X[0]!.length;
  if (actualFeatures !== expectedFeatures) {
    throw new Error(`Prediction data has ${actualFeatures} features, expected ${expectedFeatures}`);
  }
}

export function validateFeatureVector(point: number[], expectedFeatures: number): void {
  if (point.length !== expectedFeatures) {
    throw new Error(`Prediction point has ${point.length} features, expected ${expectedFeatures}`);
  }
  if (!point.every(Number.isFinite)) {
    throw new Error("Prediction point must contain only finite numbers");
  }
}

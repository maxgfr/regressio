import type { DataMatrix } from "../types";

/**
 * Generate polynomial features: for each column x, add x², x³, ..., x^degree.
 * The original columns are preserved.
 */
export function polynomialFeatures(X: DataMatrix, degree: number): DataMatrix {
  return X.map((row) => {
    const expanded: number[] = [];
    for (const x of row) {
      for (let d = 1; d <= degree; d++) {
        expanded.push(x ** d);
      }
    }
    return expanded;
  });
}

/**
 * Generate interaction features: x_i * x_j for all pairs (or specified pairs).
 * Original columns are NOT included in the output — only the interactions.
 */
export function interactionFeatures(X: DataMatrix, pairs?: [number, number][]): DataMatrix {
  const p = X[0]!.length;

  // Default: all unique pairs
  const allPairs =
    pairs ??
    (() => {
      const result: [number, number][] = [];
      for (let i = 0; i < p; i++) {
        for (let j = i + 1; j < p; j++) {
          result.push([i, j]);
        }
      }
      return result;
    })();

  return X.map((row) => allPairs.map(([i, j]) => row[i]! * row[j]!));
}

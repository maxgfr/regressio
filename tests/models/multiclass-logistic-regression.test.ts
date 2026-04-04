import { describe, expect, test } from "bun:test";
import { MulticlassLogisticRegression } from "../../src/models/multiclass-logistic-regression";

describe("MulticlassLogisticRegression", () => {
  test("classifies 3 linearly separable clusters", () => {
    // 3 clusters: class 0 near origin, class 1 near (5,5), class 2 near (10,0)
    const X = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [5, 5],
      [6, 5],
      [5, 6],
      [6, 6],
      [10, 0],
      [11, 0],
      [10, 1],
      [11, 1],
    ];
    const y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    const model = new MulticlassLogisticRegression({
      learningRate: 0.05,
      maxIterations: 500,
    });
    model.fit(X, y);

    const _predictions = model.predict(X);
    const stats = model.statistics();
    expect(stats.accuracy).toBeGreaterThan(0.8);
    expect(stats.nClasses).toBe(3);
  });

  test("predictProbability sums to 1 for each sample", () => {
    const X = [
      [0, 0],
      [5, 5],
      [10, 0],
    ];
    const _y = [0, 1, 2];

    const model = new MulticlassLogisticRegression({ maxIterations: 100 });
    model.fit(
      [
        [0, 0],
        [1, 0],
        [5, 5],
        [6, 5],
        [10, 0],
        [11, 0],
      ],
      [0, 0, 1, 1, 2, 2],
    );

    const probs = model.predictProbability(X);
    for (const row of probs) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 6);
    }
  });

  test("probabilities between 0 and 1", () => {
    const X = [
      [0, 0],
      [1, 0],
      [5, 5],
      [6, 5],
    ];
    const y = [0, 0, 1, 1];

    const model = new MulticlassLogisticRegression({ maxIterations: 100 });
    model.fit(X, y);

    const probs = model.predictProbability(X);
    for (const row of probs) {
      for (const p of row) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    }
  });

  test("statistics returns per-class precision and recall", () => {
    const X = [
      [0, 0],
      [1, 0],
      [0, 1],
      [5, 5],
      [6, 5],
      [5, 6],
    ];
    const y = [0, 0, 0, 1, 1, 1];

    const model = new MulticlassLogisticRegression({ maxIterations: 300 });
    model.fit(X, y);
    const stats = model.statistics();
    expect(stats.precision.length).toBe(2);
    expect(stats.recall.length).toBe(2);
  });

  test("throws if not fitted", () => {
    const model = new MulticlassLogisticRegression();
    expect(() => model.predict([[1, 2]])).toThrow("not been fitted");
  });
});

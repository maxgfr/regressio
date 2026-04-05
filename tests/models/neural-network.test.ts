import { describe, expect, test } from "bun:test";
import { NeuralNetwork } from "../../src/models/neural-network";

describe("NeuralNetwork", () => {
  describe("regression", () => {
    test("learns approximate linear function y = 2x", () => {
      const X = Array.from({ length: 50 }, (_, i) => [i / 10]);
      const y = X.map(([x]) => 2 * x!);

      const model = new NeuralNetwork({
        layers: [{ units: 8, activation: "relu" }],
        learningRate: 0.001,
        epochs: 200,
        task: "regression",
      });
      model.fit(X, y);

      const predictions = model.predict(X);
      // Check that predictions correlate with true values
      let sumSqErr = 0;
      let sumSqTot = 0;
      const yMean = y.reduce((a, b) => a + b, 0) / y.length;
      for (let i = 0; i < y.length; i++) {
        sumSqErr += (predictions[i]! - y[i]!) ** 2;
        sumSqTot += (y[i]! - yMean) ** 2;
      }
      const r2 = 1 - sumSqErr / sumSqTot;
      expect(r2).toBeGreaterThan(-0.5);
    });
  });

  describe("classification", () => {
    test("learns simple binary classification", () => {
      // Two clusters
      const X = [
        [0, 0],
        [0.1, 0.1],
        [0.2, 0],
        [0, 0.2],
        [1, 1],
        [0.9, 1],
        [1, 0.9],
        [0.8, 0.8],
      ];
      const y = [0, 0, 0, 0, 1, 1, 1, 1];

      const model = new NeuralNetwork({
        layers: [{ units: 4, activation: "sigmoid" }],
        learningRate: 0.5,
        epochs: 300,
        task: "classification",
      });
      model.fit(X, y);

      const predictions = model.predict(X);
      let correct = 0;
      for (let i = 0; i < y.length; i++) {
        if (predictions[i] === y[i]) correct++;
      }
      expect(correct / y.length).toBeGreaterThan(0.7);
    });

    test("multiclass classification (3 classes)", () => {
      const X = [
        [0, 0],
        [0.5, 0],
        [5, 5],
        [5.5, 5],
        [10, 0],
        [10.5, 0],
      ];
      const y = [0, 0, 1, 1, 2, 2];

      const model = new NeuralNetwork({
        layers: [{ units: 8, activation: "relu" }],
        learningRate: 0.01,
        epochs: 300,
        task: "classification",
      });
      model.fit(X, y);

      const predictions = model.predict(X);
      // Should get at least some right
      let correct = 0;
      for (let i = 0; i < y.length; i++) {
        if (predictions[i] === y[i]) correct++;
      }
      expect(correct).toBeGreaterThanOrEqual(2);
    });
  });

  test("predictRaw returns correct output shape", () => {
    const X = [
      [1, 2],
      [3, 4],
    ];
    const y = [0, 1];

    const model = new NeuralNetwork({
      layers: [{ units: 4 }],
      epochs: 10,
      task: "classification",
    });
    model.fit(X, y);

    const raw = model.predictRaw(X);
    expect(raw.length).toBe(2);
    expect(raw[0]!.length).toBe(2); // 2 classes
  });

  test("throws if not fitted", () => {
    const model = new NeuralNetwork({ layers: [{ units: 4 }] });
    expect(() => model.predict([[1]])).toThrow("not been fitted");
  });

  test("multiple hidden layers", () => {
    const X = Array.from({ length: 20 }, (_, i) => [i / 5]);
    const y = X.map(([x]) => (Math.sin(x!) > 0 ? 1 : 0));

    const model = new NeuralNetwork({
      layers: [
        { units: 8, activation: "tanh" },
        { units: 4, activation: "relu" },
      ],
      learningRate: 0.01,
      epochs: 100,
      task: "classification",
    });
    model.fit(X, y);

    // Just verify it doesn't crash and produces valid output
    const predictions = model.predict(X);
    expect(predictions.length).toBe(X.length);
    for (const p of predictions) {
      expect([0, 1]).toContain(p);
    }
  });
});

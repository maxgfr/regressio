import { describe, expect, test } from "bun:test";
import type { DataInput } from "../../src";
import {
  ElasticNet,
  KNearestNeighbors,
  LassoRegression,
  LinearRegression,
  LogisticRegression,
  MulticlassLogisticRegression,
  NeuralNetwork,
  PolynomialRegression,
  RidgeRegression,
  RobustRegression,
  WeightedRegression,
} from "../../src";

interface Predictor {
  predict(X: DataInput): number[];
}

const unfittedModels: { name: string; model: Predictor; input: DataInput }[] = [
  { name: "LinearRegression", model: new LinearRegression(), input: [[1, 2]] },
  { name: "PolynomialRegression", model: new PolynomialRegression(), input: [1] },
  { name: "RidgeRegression", model: new RidgeRegression(), input: [[1, 2]] },
  { name: "LassoRegression", model: new LassoRegression(), input: [[1, 2]] },
  { name: "ElasticNet", model: new ElasticNet(), input: [[1, 2]] },
  { name: "WeightedRegression", model: new WeightedRegression(), input: [[1, 2]] },
  { name: "RobustRegression", model: new RobustRegression(), input: [[1, 2]] },
  { name: "LogisticRegression", model: new LogisticRegression(), input: [[1, 2]] },
  {
    name: "MulticlassLogisticRegression",
    model: new MulticlassLogisticRegression(),
    input: [[1, 2]],
  },
  { name: "KNearestNeighbors", model: new KNearestNeighbors({ k: 1 }), input: [[1, 2]] },
  {
    name: "NeuralNetwork",
    model: new NeuralNetwork({ layers: [{ units: 2 }] }),
    input: [[1, 2]],
  },
];

const X2 = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
  [2, 1],
  [1, 2],
];
const yRegression = [0, 1, 1, 2, 3, 3];
const yBinary = [0, 0, 0, 1, 1, 1];

const fittedModels: { name: string; create: () => Predictor; invalidInput: DataInput }[] = [
  {
    name: "LinearRegression",
    create: () => new LinearRegression().fit(X2, yRegression),
    invalidInput: [[1]],
  },
  {
    name: "PolynomialRegression",
    create: () => new PolynomialRegression({ degree: 2 }).fit([0, 1, 2, 3], [0, 1, 4, 9]),
    invalidInput: [[1, 2]],
  },
  {
    name: "RidgeRegression",
    create: () => new RidgeRegression().fit(X2, yRegression),
    invalidInput: [[1]],
  },
  {
    name: "LassoRegression",
    create: () => new LassoRegression({ alpha: 0.01 }).fit(X2, yRegression),
    invalidInput: [[1]],
  },
  {
    name: "ElasticNet",
    create: () => new ElasticNet({ alpha: 0.01 }).fit(X2, yRegression),
    invalidInput: [[1]],
  },
  {
    name: "WeightedRegression",
    create: () => new WeightedRegression().fit(X2, yRegression, [1, 1, 1, 1, 1, 1]),
    invalidInput: [[1]],
  },
  {
    name: "RobustRegression",
    create: () => new RobustRegression().fit(X2, yRegression),
    invalidInput: [[1]],
  },
  {
    name: "LogisticRegression",
    create: () => new LogisticRegression().fit(X2, yBinary),
    invalidInput: [[1]],
  },
  {
    name: "MulticlassLogisticRegression",
    create: () =>
      new MulticlassLogisticRegression().fit(
        [
          [0, 0],
          [0, 1],
          [3, 3],
          [3, 4],
          [6, 0],
          [6, 1],
        ],
        [0, 0, 1, 1, 2, 2],
      ),
    invalidInput: [[1]],
  },
  {
    name: "KNearestNeighbors",
    create: () => new KNearestNeighbors({ k: 1 }).fit(X2, yBinary),
    invalidInput: [[1]],
  },
  {
    name: "NeuralNetwork",
    create: () => new NeuralNetwork({ layers: [{ units: 2 }], epochs: 1 }).fit(X2, yRegression),
    invalidInput: [[1]],
  },
];

describe("model state guards", () => {
  for (const { name, model, input } of unfittedModels) {
    test(`${name} rejects prediction before fit`, () => {
      expect(() => model.predict(input)).toThrow("Model has not been fitted");
    });
  }
});

describe("prediction feature guards", () => {
  for (const { name, create, invalidInput } of fittedModels) {
    test(`${name} rejects a different feature count`, () => {
      expect(() => create().predict(invalidInput)).toThrow("features");
    });
  }
});

describe("training data guards", () => {
  test("regression rejects ragged matrices", () => {
    expect(() => new LinearRegression().fit([[1, 2], [3]], [1, 2])).toThrow("columns");
  });

  test("standalone models reject non-finite training values", () => {
    expect(() => new LogisticRegression().fit([[0], [Number.NaN]], [0, 1])).toThrow(
      "finite numbers",
    );
    expect(() =>
      new KNearestNeighbors({ k: 1 }).fit([[0], [Number.POSITIVE_INFINITY]], [0, 1]),
    ).toThrow("finite numbers");
    expect(() =>
      new NeuralNetwork({ layers: [{ units: 2 }] }).fit([[0], [Number.NaN]], [0, 1]),
    ).toThrow("finite numbers");
  });
});

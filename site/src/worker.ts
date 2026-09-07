/// <reference lib="webworker" />

import {
  bootstrapCoefficients,
  breuschPagan,
  conditionNumber,
  confidenceInterval,
  cooksDistance,
  correlationMatrix,
  dropMissing,
  durbinWatson,
  ElasticNet,
  imputeMean,
  imputeMedian,
  interactionFeatures,
  isWasmActive,
  KNearestNeighbors,
  LassoRegression,
  LinearRegression,
  LogisticRegression,
  leverage,
  Matrix,
  MulticlassLogisticRegression,
  NeuralNetwork,
  normalize,
  oneHotEncode,
  PolynomialRegression,
  polynomialFeatures,
  predictionInterval,
  RidgeRegression,
  RobustRegression,
  residualDiagnostics,
  shapiroWilk,
  standardize,
  studentizedResiduals,
  unnormalize,
  unstandardize,
  vif,
  WeightedRegression,
} from "../../dist/index.js";
import { API_FAMILIES } from "./coverage";
import type { ApiCheck, PlotPoint, SuiteResult, WorkerResponse } from "./types";

const scope = self as unknown as DedicatedWorkerGlobalScope;

const EXPORT_NAMES = Object.keys(API_FAMILIES);

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function finite(values: number[], label: string): void {
  assert(values.length > 0, `${label} returned no values`);
  assert(values.every(Number.isFinite), `${label} returned a non-finite value`);
}

function close(actual: number, expected: number, tolerance = 1e-6): void {
  assert(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

async function waitForWasm(): Promise<boolean> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (isWasmActive()) return true;
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
  return isWasmActive();
}

function createDataset(): { X: number[][]; y: number[]; yPoly: number[]; weights: number[] } {
  const X = Array.from({ length: 36 }, (_, index) => [(index - 17.5) / 1.75]);
  const noise = [0.4, -1.1, 1.8, -0.6, 1, -1.5];
  const y = X.map((row, index) => 2.5 * row[0]! - 1 + noise[index % noise.length]!);
  const yPoly = X.map((row, index) => {
    const x = row[0]!;
    return -0.03 * x * x + 2.5 * x - 1 + noise[index % noise.length]!;
  });
  const weights = X.map((_, index) => (index === 31 ? 0.2 : 1));
  return { X, y, yPoly, weights };
}

async function runSuite(): Promise<SuiteResult> {
  const started = performance.now();
  const startedAt = new Date().toISOString();
  const checks: ApiCheck[] = [];
  const { X, y, yPoly, weights } = createDataset();
  const linear = new LinearRegression().fit(X, y);
  const robust = new RobustRegression().fit(X, y);
  const polynomial = new PolynomialRegression({ degree: 2 }).fit(X, yPoly);
  const yHat = linear.predict(X);

  const run = async (name: string, test: () => unknown | Promise<unknown>): Promise<void> => {
    const begin = performance.now();
    let status: ApiCheck["status"] = "pass";
    let value = "verified";
    let error: string | undefined;
    try {
      const result = await test();
      if (typeof result === "number")
        value = Number.isFinite(result) ? result.toFixed(4) : String(result);
      else if (typeof result === "string") value = result;
      else if (Array.isArray(result)) value = `${result.length} values`;
      else if (result && typeof result === "object") value = "structured result";
    } catch (cause) {
      status = "fail";
      error = cause instanceof Error ? cause.message : String(cause);
      value = "failed";
    }
    checks.push({
      name,
      family: API_FAMILIES[name as keyof typeof API_FAMILIES] ?? "Other",
      status,
      value,
      duration: performance.now() - begin,
      ...(error ? { error } : {}),
    });
    scope.postMessage({
      type: "progress",
      completed: checks.length,
      total: EXPORT_NAMES.length,
      name,
    } satisfies WorkerResponse);
  };

  const exerciseRegression = (
    model: {
      fit: (features: number[][], target: number[], weights?: number[]) => unknown;
      predict: (features: number[][]) => number[];
      readonly coefficients: number[];
      readonly intercept: number;
      residuals: () => number[];
      statistics: () => { rSquared: number };
      summary: () => string;
    },
    target = y,
    fitWeights?: number[],
  ): number => {
    model.fit(X, target, fitWeights);
    const prediction = model.predict(X);
    finite(prediction, "prediction");
    finite(model.coefficients, "coefficients");
    assert(Number.isFinite(model.intercept), "intercept is not finite");
    finite(model.residuals(), "residuals");
    const stats = model.statistics();
    assert(Number.isFinite(stats.rSquared), "R² is not finite");
    assert(model.summary().includes("Coefficients:"), "summary is missing coefficients");
    return stats.rSquared;
  };

  await run("LinearRegression", () => exerciseRegression(new LinearRegression()));
  await run("PolynomialRegression", () =>
    exerciseRegression(new PolynomialRegression({ degree: 2 }), yPoly),
  );
  await run("RidgeRegression", () => exerciseRegression(new RidgeRegression({ alpha: 0.15 })));
  await run("LassoRegression", () => exerciseRegression(new LassoRegression({ alpha: 0.01 })));
  await run("ElasticNet", () => exerciseRegression(new ElasticNet({ alpha: 0.01, l1Ratio: 0.5 })));
  await run("WeightedRegression", () => exerciseRegression(new WeightedRegression({ weights })));
  await run("RobustRegression", () => exerciseRegression(new RobustRegression()));

  await run("LogisticRegression", () => {
    const features = X.map((row) => [row[0]! / 4]);
    const target = features.map((row) => (row[0]! > 0 ? 1 : 0));
    const model = new LogisticRegression().fit(features, target);
    finite(model.coefficients, "logistic coefficients");
    finite(model.predictProbability(features), "probabilities");
    finite(model.predict(features), "classes");
    assert(model.statistics().accuracy > 0.9, "binary accuracy is below 90%");
    return model.statistics().accuracy;
  });

  await run("MulticlassLogisticRegression", () => {
    const features = [
      [-3, -2],
      [-2, -1],
      [-2, -3],
      [0, 3],
      [1, 2],
      [-1, 2],
      [3, -1],
      [2, 0],
      [3, 1],
    ];
    const target = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const model = new MulticlassLogisticRegression({ maxIterations: 500, learningRate: 0.25 }).fit(
      features,
      target,
    );
    assert(model.classes.length === 3, "expected three classes");
    assert(model.weights.rows > 0, "weights are empty");
    assert(
      model
        .predictProbability(features)
        .every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-6),
      "probabilities do not sum to 1",
    );
    assert(model.statistics().accuracy > 0.65, "multiclass accuracy is too low");
    return model.statistics().accuracy;
  });

  await run("KNearestNeighbors", () => {
    const train = [[0], [1], [2], [8], [9], [10]];
    const classifier = new KNearestNeighbors({ k: 3 }).fit(train, [0, 0, 0, 1, 1, 1]);
    const regressor = new KNearestNeighbors({
      k: 2,
      mode: "regression",
      distance: "manhattan",
    }).fit(train, [0, 1, 2, 8, 9, 10]);
    assert(classifier.predict([[1.5], [9]])[0] === 0, "classification vote is incorrect");
    finite(regressor.predict([[4]]), "KNN regression");
    assert(classifier.neighbors([1.5]).length === 3, "neighbors() returned the wrong count");
    return "classification + regression";
  });

  await run("NeuralNetwork", () => {
    const regression = new NeuralNetwork({
      layers: [{ units: 4, activation: "tanh" }],
      epochs: 80,
      learningRate: 0.01,
    }).fit([[0], [0.25], [0.5], [0.75], [1]], [0, 0.5, 1, 1.5, 2]);
    finite(regression.predict([[0.4]]), "neural regression");
    finite(regression.predictRaw([[0.4]])[0]!, "neural raw output");
    const classifier = new NeuralNetwork({
      layers: [{ units: 4, activation: "tanh" }],
      task: "classification",
      epochs: 100,
      learningRate: 0.05,
    }).fit([[-2], [-1], [1], [2]], [0, 0, 1, 1]);
    finite(classifier.predict([[-1.5], [1.5]]), "neural classification");
    finite(classifier.predictRaw([[1.5]])[0]!, "neural probabilities");
    return "regression + classification";
  });

  await run("Matrix", () => {
    const direct = new Matrix(2, 2, [1, 2, 3, 4]);
    const matrix = Matrix.fromArray([
      [1, 2],
      [3, 4],
    ]);
    const zeros = Matrix.zeros(2, 2);
    const ones = Matrix.ones(2, 2);
    const identity = Matrix.identity(2);
    const column = Matrix.columnVector([1, 2]);
    const row = Matrix.rowVector([1, 2]);
    const diagonal = Matrix.diagonal([2, 3]);
    zeros.set(0, 0, 2);
    close(zeros.get(0, 0), 2);
    close(matrix.getColumn(1).get(0, 0), 2);
    close(matrix.getRow(1).get(0, 0), 3);
    zeros.setColumn(1, column);
    close(matrix.transpose().get(0, 1), 3);
    close(matrix.multiply(identity).get(1, 1), 4);
    close(matrix.add(ones).get(0, 0), 2);
    close(matrix.subtract(ones).get(0, 0), 0);
    close(matrix.scale(2).get(0, 0), 2);
    const mutable = matrix.clone();
    mutable.addInPlace(ones);
    mutable.subtractInPlace(ones);
    mutable.scaleInPlace(2);
    close(mutable.get(1, 1), 8);
    close(matrix.norm(), Math.sqrt(30));
    close(matrix.trace(), 5);
    close(matrix.determinant(), -2);
    close(matrix.submatrix(0, 1, 0, 2).get(0, 1), 2);
    assert(matrix.toArray()[1]![1] === 4, "toArray failed");
    assert(matrix.toFlatArray().length === 4, "toFlatArray failed");
    close(column.dot(column), 5);
    assert(
      direct.rows === 2 && row.cols === 2 && diagonal.trace() === 5,
      "factory or constructor failed",
    );
    return matrix.determinant();
  });

  const diagnostics = residualDiagnostics(X, y, yHat);
  await run("residualDiagnostics", () => {
    finite(diagnostics.raw, "raw residuals");
    return diagnostics.raw.length;
  });
  await run("studentizedResiduals", () => {
    const value = studentizedResiduals(X, y, yHat);
    finite(value, "studentized residuals");
    return Math.max(...value.map(Math.abs));
  });
  await run("cooksDistance", () => {
    const value = cooksDistance(X, y, yHat);
    finite(value, "Cook's distance");
    return Math.max(...value);
  });
  await run("leverage", () => {
    const value = leverage(X);
    finite(value, "leverage");
    return Math.max(...value);
  });
  await run("durbinWatson", () => durbinWatson(linear.residuals()).statistic);
  await run("breuschPagan", () => breuschPagan(X, linear.residuals()).statistic);
  await run("shapiroWilk", () => shapiroWilk(linear.residuals()).statistic);

  const multivariate = X.map((row, index) => [
    row[0]!,
    Math.sin(index * 0.7) + row[0]! * 0.05,
    Math.cos(index * 0.3),
  ]);
  await run("vif", () => {
    const value = vif(multivariate);
    finite(value, "VIF");
    return Math.max(...value);
  });
  await run("correlationMatrix", () => {
    const value = correlationMatrix(multivariate);
    finite(value.flat(), "correlation matrix");
    return value[0]![1]!;
  });
  await run("conditionNumber", () => conditionNumber(multivariate));

  const newX = [[-4], [0], [4]];
  const newYHat = linear.predict(newX);
  await run("confidenceInterval", () => {
    const value = confidenceInterval(X, y, yHat, newX, newYHat);
    assert(
      value.every((item) => item.lower <= item.predicted && item.predicted <= item.upper),
      "prediction outside confidence interval",
    );
    return value.length;
  });
  await run("predictionInterval", () => {
    const confidence = confidenceInterval(X, y, yHat, newX, newYHat);
    const value = predictionInterval(X, y, yHat, newX, newYHat);
    assert(
      value[0]!.upper - value[0]!.lower > confidence[0]!.upper - confidence[0]!.lower,
      "prediction interval should be wider",
    );
    return value.length;
  });
  await run("bootstrapCoefficients", () => {
    const value = bootstrapCoefficients(X, y, 120);
    finite(value.coefficients, "bootstrap coefficients");
    return value.coefficients.length;
  });

  const sample = [
    [1, 10],
    [2, 20],
    [3, 30],
  ];
  await run("standardize", () => {
    const value = standardize(sample);
    close(unstandardize(value.transformed, value)[2]![1]!, 30);
    return value.transformed.length;
  });
  await run("unstandardize", () => {
    const value = standardize(sample);
    return unstandardize(value.transformed, value)[0]![0]!;
  });
  await run("normalize", () => {
    const value = normalize(sample);
    close(value.transformed[0]![0]!, 0);
    return value.transformed.length;
  });
  await run("unnormalize", () => {
    const value = normalize(sample);
    close(unnormalize(value.transformed, value)[2]![1]!, 30);
    return value.transformed.length;
  });
  await run("oneHotEncode", () => {
    const value = oneHotEncode(["red", "blue", "red"]);
    assert(value[0]!.reduce((a, b) => a + b, 0) === 1, "one-hot row is invalid");
    return value.length;
  });
  await run("polynomialFeatures", () => {
    const value = polynomialFeatures([[2, 3]], 3);
    assert(value[0]!.length === 6, "polynomial feature count is wrong");
    return value[0]!.length;
  });
  await run("interactionFeatures", () => {
    const value = interactionFeatures([[2, 3, 4]]);
    assert(value[0]!.join(",") === "6,8,12", "interaction features are wrong");
    return value[0]!.length;
  });
  await run("dropMissing", () => {
    const value = dropMissing(
      [
        [1, 2],
        [Number.NaN, 3],
        [4, 5],
      ],
      [1, 2, 3],
    );
    assert(value.X.length === 2 && value.y?.length === 2, "missing rows were not removed");
    return value.X.length;
  });
  await run("imputeMean", () => {
    const value = imputeMean([
      [1, 10],
      [Number.NaN, 20],
      [3, Number.NaN],
    ]);
    close(value[1]![0]!, 2);
    return value[1]![0]!;
  });
  await run("imputeMedian", () => {
    const value = imputeMedian([
      [1, 10],
      [Number.NaN, 20],
      [9, Number.NaN],
    ]);
    close(value[1]![0]!, 5);
    return value[1]![0]!;
  });
  const wasmCheckStarted = performance.now();
  checks.push({
    name: "isWasmActive",
    family: API_FAMILIES.isWasmActive,
    status: "pass",
    value: isWasmActive() ? "active" : "fallback",
    duration: performance.now() - wasmCheckStarted,
  });
  scope.postMessage({
    type: "progress",
    completed: checks.length,
    total: EXPORT_NAMES.length,
    name: "isWasmActive",
  } satisfies WorkerResponse);

  const plot: PlotPoint[] = X.map((row, index) => ({
    x: row[0]!,
    y: y[index]!,
    predicted: yHat[index]!,
    robust: robust.predict([[row[0]!]])[0]!,
    polynomial: polynomial.predict([[row[0]!]])[0]!,
  }));

  return {
    checks,
    startedAt,
    duration: performance.now() - started,
    wasm: isWasmActive(),
    plot,
    coefficients: {
      ols: [linear.intercept, linear.coefficients[0]!],
      robust: [robust.intercept, robust.coefficients[0]!],
      polynomial: [polynomial.intercept, ...polynomial.coefficients],
    },
    diagnostics: {
      rSquared: linear.statistics().rSquared,
      adjustedRSquared: linear.statistics().adjustedRSquared,
      durbinWatson: durbinWatson(linear.residuals()).statistic,
      maxCook: Math.max(...diagnostics.cooksDistance),
      maxLeverage: Math.max(...diagnostics.leverage),
      residualStdError: linear.statistics().residualStandardError,
    },
    examples: {
      Regression: `const model = new LinearRegression();\nmodel.fit(X, y);\nconst yHat = model.predict(X);\nconsole.log(model.statistics());`,
      Classification: `const model = new LogisticRegression();\nmodel.fit(X, labels);\nconst probabilities = model.predictProbability(X);`,
      Diagnostics: `const model = new LinearRegression().fit(X, y);\nconst result = residualDiagnostics(X, y, model.predict(X));`,
      Preprocessing: `const scaled = standardize(X);\nconst original = unstandardize(scaled.transformed, scaled);`,
      Intervals: `const intervals = predictionInterval(\n  X, y, model.predict(X), newX, model.predict(newX)\n);`,
      Matrix: `const A = Matrix.fromArray([[1, 2], [3, 4]]);\nconst determinant = A.determinant();\nconst product = A.multiply(Matrix.identity(2));`,
    },
  };
}

scope.addEventListener("message", async (event: MessageEvent<{ type: "run" }>) => {
  if (event.data.type !== "run") return;
  try {
    scope.postMessage({
      type: "progress",
      completed: 0,
      total: EXPORT_NAMES.length,
      name: "initializing WASM",
    } satisfies WorkerResponse);
    await waitForWasm();
    const result = await runSuite();
    scope.postMessage({ type: "complete", result } satisfies WorkerResponse);
  } catch (cause) {
    scope.postMessage({
      type: "fatal",
      error: cause instanceof Error ? cause.message : String(cause),
    } satisfies WorkerResponse);
  }
});

import { describe, expect, test } from "bun:test";
import { LinearRegression } from "../../src/models/linear-regression";
import { confidenceInterval, predictionInterval } from "../../src/predictions/intervals";

describe("confidenceInterval", () => {
  const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1];
  const model = new LinearRegression();
  model.fit(X, y);
  const yHat = model.predict(X);

  test("returns correct number of intervals", () => {
    const newX = [[5], [6]];
    const newYHat = model.predict(newX);
    const ci = confidenceInterval(X, y, yHat, newX, newYHat);
    expect(ci.length).toBe(2);
  });

  test("predicted value is inside the interval", () => {
    const newX = [[5]];
    const newYHat = model.predict(newX);
    const ci = confidenceInterval(X, y, yHat, newX, newYHat);
    expect(ci[0]!.predicted).toBeCloseTo(newYHat[0]!, 10);
    expect(ci[0]!.lower).toBeLessThan(ci[0]!.predicted);
    expect(ci[0]!.upper).toBeGreaterThan(ci[0]!.predicted);
  });

  test("lower < upper", () => {
    const newX = [[3], [7]];
    const newYHat = model.predict(newX);
    const ci = confidenceInterval(X, y, yHat, newX, newYHat);
    for (const interval of ci) {
      expect(interval.lower).toBeLessThan(interval.upper);
    }
  });

  test("narrower at mean of X", () => {
    // CI should be narrower near the center of the data
    const nearCenter = [[5.5]];
    const nearEdge = [[1]];
    const yHatCenter = model.predict(nearCenter);
    const yHatEdge = model.predict(nearEdge);
    const ciCenter = confidenceInterval(X, y, yHat, nearCenter, yHatCenter);
    const ciEdge = confidenceInterval(X, y, yHat, nearEdge, yHatEdge);
    const widthCenter = ciCenter[0]!.upper - ciCenter[0]!.lower;
    const widthEdge = ciEdge[0]!.upper - ciEdge[0]!.lower;
    expect(widthCenter).toBeLessThan(widthEdge);
  });
});

describe("predictionInterval", () => {
  const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1];
  const model = new LinearRegression();
  model.fit(X, y);
  const yHat = model.predict(X);

  test("prediction interval wider than confidence interval", () => {
    const newX = [[5]];
    const newYHat = model.predict(newX);
    const ci = confidenceInterval(X, y, yHat, newX, newYHat);
    const pi = predictionInterval(X, y, yHat, newX, newYHat);
    const ciWidth = ci[0]!.upper - ci[0]!.lower;
    const piWidth = pi[0]!.upper - pi[0]!.lower;
    expect(piWidth).toBeGreaterThan(ciWidth);
  });

  test("lower < predicted < upper", () => {
    const newX = [[5]];
    const newYHat = model.predict(newX);
    const pi = predictionInterval(X, y, yHat, newX, newYHat);
    expect(pi[0]!.lower).toBeLessThan(pi[0]!.predicted);
    expect(pi[0]!.upper).toBeGreaterThan(pi[0]!.predicted);
  });
});

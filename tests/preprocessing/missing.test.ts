import { describe, expect, test } from "bun:test";
import { dropMissing, imputeMean, imputeMedian } from "../../src/preprocessing/missing";

describe("dropMissing", () => {
  test("removes rows with NaN", () => {
    const X = [
      [1, 2],
      [NaN, 4],
      [5, 6],
    ];
    const y = [1, 2, 3];
    const result = dropMissing(X, y);
    expect(result.X).toEqual([
      [1, 2],
      [5, 6],
    ]);
    expect(result.y).toEqual([1, 3]);
  });

  test("removes rows where y is NaN", () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const y = [1, NaN, 3];
    const result = dropMissing(X, y);
    expect(result.X).toEqual([
      [1, 2],
      [5, 6],
    ]);
    expect(result.y).toEqual([1, 3]);
  });

  test("works without y", () => {
    const X = [
      [1, 2],
      [NaN, 4],
      [5, 6],
    ];
    const result = dropMissing(X);
    expect(result.X).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });
});

describe("imputeMean", () => {
  test("replaces NaN with column mean", () => {
    const X = [
      [1, 10],
      [NaN, 20],
      [3, NaN],
    ];
    const result = imputeMean(X);
    expect(result[1]![0]).toBeCloseTo(2, 10); // mean of 1, 3
    expect(result[2]![1]).toBeCloseTo(15, 10); // mean of 10, 20
  });

  test("leaves non-NaN values unchanged", () => {
    const X = [
      [1, 2],
      [3, 4],
    ];
    const result = imputeMean(X);
    expect(result).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});

describe("imputeMedian", () => {
  test("replaces NaN with column median", () => {
    const X = [
      [1, 10],
      [NaN, 20],
      [5, NaN],
      [3, 40],
    ];
    const result = imputeMedian(X);
    expect(result[1]![0]).toBeCloseTo(3, 10); // median of [1, 3, 5] = 3
    expect(result[2]![1]).toBeCloseTo(20, 10); // median of [10, 20, 40] = 20
  });
});

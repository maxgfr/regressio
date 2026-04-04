import { describe, expect, test } from "bun:test";
import { KNearestNeighbors } from "../../src/models/knn";

describe("KNearestNeighbors", () => {
  describe("classification", () => {
    test("classifies simple 2D data", () => {
      const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [10, 10],
        [11, 10],
        [10, 11],
      ];
      const y = [0, 0, 0, 1, 1, 1];

      const model = new KNearestNeighbors({ k: 3 });
      model.fit(X, y);

      expect(model.predict([[0.5, 0.5]])[0]).toBe(0);
      expect(model.predict([[10.5, 10.5]])[0]).toBe(1);
    });

    test("1D classification", () => {
      const X = [1, 2, 3, 8, 9, 10];
      const y = [0, 0, 0, 1, 1, 1];

      const model = new KNearestNeighbors({ k: 3 });
      model.fit(X, y);

      expect(model.predict([2])[0]).toBe(0);
      expect(model.predict([9])[0]).toBe(1);
    });

    test("manhattan distance works", () => {
      const X = [
        [0, 0],
        [1, 1],
        [10, 10],
        [11, 11],
      ];
      const y = [0, 0, 1, 1];

      const model = new KNearestNeighbors({ k: 2, distance: "manhattan" });
      model.fit(X, y);
      expect(model.predict([[0, 1]])[0]).toBe(0);
    });
  });

  describe("regression", () => {
    test("predicts mean of neighbors", () => {
      const X = [1, 2, 3, 4, 5];
      const y = [10, 20, 30, 40, 50];

      const model = new KNearestNeighbors({ k: 3, mode: "regression" });
      model.fit(X, y);

      // Nearest to 2: neighbors are 1,2,3 → mean of 10,20,30 = 20
      const pred = model.predict([2]);
      expect(pred[0]).toBeCloseTo(20, 6);
    });

    test("k=1 returns exact neighbor value", () => {
      const X = [1, 5, 10];
      const y = [100, 200, 300];

      const model = new KNearestNeighbors({ k: 1, mode: "regression" });
      model.fit(X, y);

      expect(model.predict([1])[0]).toBe(100);
      expect(model.predict([5])[0]).toBe(200);
    });
  });

  test("neighbors returns k indices", () => {
    const X = [
      [0, 0],
      [1, 0],
      [100, 100],
    ];
    const y = [0, 0, 1];

    const model = new KNearestNeighbors({ k: 2 });
    model.fit(X, y);

    const nbrs = model.neighbors([0.5, 0]);
    expect(nbrs.length).toBe(2);
    expect(nbrs).toContain(0);
    expect(nbrs).toContain(1);
  });

  test("throws if not fitted", () => {
    const model = new KNearestNeighbors();
    expect(() => model.predict([[1]])).toThrow("not been fitted");
  });

  test("throws if k > n", () => {
    const model = new KNearestNeighbors({ k: 10 });
    expect(() => model.fit([1, 2, 3], [0, 0, 1])).toThrow("at least k=10");
  });
});

import { describe, expect, test } from "bun:test";
import { oneHotEncode } from "../../src/preprocessing/encoding";

describe("oneHotEncode", () => {
  test("basic encoding", () => {
    const result = oneHotEncode(["a", "b", "c", "a"], ["a", "b", "c"]);
    expect(result).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ]);
  });

  test("auto-detects categories", () => {
    const result = oneHotEncode(["x", "y", "x"]);
    expect(result.length).toBe(3);
    expect(result[0]!.length).toBe(2);
  });

  test("dropFirst avoids multicollinearity trap", () => {
    const result = oneHotEncode(["a", "b", "c"], ["a", "b", "c"], true);
    expect(result[0]!.length).toBe(2); // dropped "a"
    expect(result).toEqual([
      [0, 0], // a → both 0
      [1, 0], // b
      [0, 1], // c
    ]);
  });

  test("numeric categories", () => {
    const result = oneHotEncode([1, 2, 3, 1], [1, 2, 3]);
    expect(result).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ]);
  });
});

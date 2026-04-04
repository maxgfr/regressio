/**
 * One-hot encode a categorical column.
 * @param column Array of category labels (strings or numbers)
 * @param categories Optional explicit category list. Auto-detected if omitted.
 * @param dropFirst If true, drop the first category to avoid multicollinearity trap (default: false).
 * @returns 2D array where each row is a binary vector.
 */
export function oneHotEncode(
  column: (string | number)[],
  categories?: (string | number)[],
  dropFirst = false,
): number[][] {
  const cats = categories ?? [...new Set(column)].sort();
  const startIdx = dropFirst ? 1 : 0;
  const resultCats = cats.slice(startIdx);

  return column.map((val) => resultCats.map((cat) => (val === cat ? 1 : 0)));
}

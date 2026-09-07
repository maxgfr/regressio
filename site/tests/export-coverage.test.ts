import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";
import * as regressio from "../../dist/index.js";
import { API_FAMILIES, METHOD_COVERAGE } from "../src/coverage";

describe("browser proof coverage", () => {
  test("tracks every runtime export", () => {
    expect(Object.keys(API_FAMILIES).sort()).toEqual(Object.keys(regressio).sort());
  });

  test("tracks every declared public class member", async () => {
    const declarations = await readFile(
      resolve(import.meta.dirname, "../../dist/index.d.ts"),
      "utf8",
    );
    const source = ts.createSourceFile("index.d.ts", declarations, ts.ScriptTarget.Latest, true);

    const declared = new Map<string, { base?: string; members: string[] }>();
    source.forEachChild((node) => {
      if (!ts.isClassDeclaration(node) || !node.name) return;
      const members = node.members
        .filter((member) => {
          const modifiers = ts.canHaveModifiers(member) ? (ts.getModifiers(member) ?? []) : [];
          return !modifiers.some(
            (modifier) =>
              modifier.kind === ts.SyntaxKind.PrivateKeyword ||
              modifier.kind === ts.SyntaxKind.ProtectedKeyword ||
              modifier.kind === ts.SyntaxKind.ReadonlyKeyword,
          );
        })
        .filter((member) => !ts.isConstructorDeclaration(member))
        .map((member) => member.name?.getText(source))
        .filter((name): name is string => Boolean(name));
      const extendsClause = node.heritageClauses?.find(
        (clause) => clause.token === ts.SyntaxKind.ExtendsKeyword,
      );
      const base = extendsClause?.types[0]?.expression.getText(source);
      declared.set(node.name.text, { ...(base ? { base } : {}), members });
    });

    const publicMembers = (className: string, seen = new Set<string>()): string[] => {
      if (seen.has(className)) throw new Error(`Circular class hierarchy at ${className}`);
      seen.add(className);
      const classInfo = declared.get(className);
      if (!classInfo) return [];
      const inherited = classInfo.base ? publicMembers(classInfo.base, seen) : [];
      return [...new Set([...inherited, ...classInfo.members])].sort();
    };

    for (const [className, expected] of Object.entries(METHOD_COVERAGE)) {
      expect(publicMembers(className)).toEqual([...expected].sort());
    }
  });
});

import { describe, expect, test } from "bun:test"
import { buildSearchArgs } from "./kamibiki"

describe("buildSearchArgs", () => {
	test("builds a basic search with the default result count", () => {
		expect(buildSearchArgs({ query: "auth flow" })).toEqual([
			"search",
			".",
			"auth flow",
			"-n",
			"10",
		])
	})

	test("clamps the result count into the 1..50 range", () => {
		expect(buildSearchArgs({ query: "x", top: 999 })).toContain("50")
		expect(buildSearchArgs({ query: "x", top: 0 })).toContain("1")
		expect(buildSearchArgs({ query: "x", top: 7 })).toContain("7")
	})

	test("trims the query", () => {
		expect(buildSearchArgs({ query: "  spaced  " })[2]).toBe("spaced")
	})

	test("appends each dir as a repeatable --dir flag", () => {
		const args = buildSearchArgs({
			query: "q",
			dirs: ["src/search", "src/index"],
		})
		expect(args).toEqual([
			"search",
			".",
			"q",
			"-n",
			"10",
			"--dir",
			"src/search",
			"--dir",
			"src/index",
		])
	})

	test("appends each exclude_dir as a repeatable --exclude-dir flag", () => {
		const args = buildSearchArgs({
			query: "q",
			exclude_dirs: ["target", "node_modules"],
		})
		expect(args.slice(5)).toEqual([
			"--exclude-dir",
			"target",
			"--exclude-dir",
			"node_modules",
		])
	})

	test("emits --dir before --exclude-dir when both are provided", () => {
		const args = buildSearchArgs({
			query: "q",
			dirs: ["src"],
			exclude_dirs: ["src/generated"],
		})
		expect(args).toEqual([
			"search",
			".",
			"q",
			"-n",
			"10",
			"--dir",
			"src",
			"--exclude-dir",
			"src/generated",
		])
	})

	test("ignores empty, whitespace-only, and non-string dir entries", () => {
		const args = buildSearchArgs({
			query: "q",
			// deliberately malformed input to exercise normalization
			dirs: ["  ", "", "src", 42 as unknown as string, "  api  "],
		})
		expect(args.filter((a) => a === "--dir").length).toBe(2)
		expect(args).toContain("src")
		expect(args).toContain("api")
	})

	test("treats missing/undefined dir arrays as no directory scoping", () => {
		expect(buildSearchArgs({ query: "q", dirs: undefined })).toEqual([
			"search",
			".",
			"q",
			"-n",
			"10",
		])
	})

	test("preserves repo-relative, cwd-relative, and absolute paths verbatim for the CLI to relativize", () => {
		const args = buildSearchArgs({
			query: "q",
			dirs: ["src", "../sibling", "/abs/path"],
		})
		const dirValues = args.filter((_, i) => args[i - 1] === "--dir")
		expect(dirValues).toEqual(["src", "../sibling", "/abs/path"])
	})
})

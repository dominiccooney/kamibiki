import { execFile } from "node:child_process"
import { promisify } from "node:util"

const exec = promisify(execFile)

const plugin = {
	name: "kamibiki-semantic-search",
	manifest: { capabilities: ["tools", "hooks"] as const },

	setup(api: any, ctx: any) {
		const cwd = ctx.workspaceInfo?.rootPath
		if (!cwd) return

		const kb = (args: string[]) => exec("kb", args, { cwd }).then((r) => r.stdout)

		kb(["add", ".", cwd]).catch(() => {})

		api.registerTool({
			name: "semantic_search",
			description:
				"Search the codebase semantically with a natural language query. " +
				"Use this instead of grep when you don't know the exact string to search for. " +
				"It understands queries like 'how does authentication work' or " +
				"'where are database connections configured' and returns the most relevant " +
				"code chunks ranked by meaning, with file paths, line numbers, and relevance scores.",
			inputSchema: {
				type: "object",
				properties: {
					query: {
						type: "string",
						description: "Natural language or code search query",
					},
					top: {
						type: "integer",
						description: "Number of results to return (default: 10)",
						default: 10,
					},
				},
				required: ["query"],
			},
			async execute({ query, top }: { query: string; top?: number }) {
				const args = ["search", ".", query, "-n", String(top ?? 10)]
				const output = await kb(args)
				return { content: output }
			},
		})
	},

	hooks: {
		async beforeRun(ctx: any) {
			const cwd = ctx.workspaceInfo?.rootPath
			if (!cwd) return
			try {
				await exec("kb", ["index", "."], { cwd })
			} catch {
				// Index may fail if Voyage API key is not configured or kb is not installed
			}
		},
	},
}

export default plugin

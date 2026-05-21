import { execFile } from "node:child_process"
import { createHash } from "node:crypto"
import { basename } from "node:path"
import { promisify } from "node:util"
import { type AgentPlugin, type AgentToolContext, createTool } from "@cline/core"

const exec = promisify(execFile)

const DEFAULT_TOP = 10
const MAX_TOP = 50
const DEFAULT_SESSION_KEY = "__default__"

interface WorkspaceState {
	cwd: string
	repoName: string
	registration?: Promise<void>
}

interface SemanticSearchInput {
	query: string
	top?: number
}

interface CommandError {
	message: string
	stdout?: string
	stderr?: string
	code?: string | number
	signal?: string
}

interface CommandResult {
	content: string
	repoName: string
}

interface CommandErrorResult {
	error: true
	message: string
	repoName?: string
	stdout?: string
	stderr?: string
	code?: string | number
	signal?: string
}

type PluginSetupApi = Parameters<NonNullable<AgentPlugin["setup"]>>[0]
type PluginSetupContext = Parameters<NonNullable<AgentPlugin["setup"]>>[1]
type PluginHooks = NonNullable<AgentPlugin["hooks"]>
type BeforeRunContext = Parameters<NonNullable<PluginHooks["beforeRun"]>>[0]

const stateBySession = new Map<string, WorkspaceState>()
let fallbackState: WorkspaceState | undefined

function sessionKey(sessionId: string | undefined): string {
	return sessionId ?? DEFAULT_SESSION_KEY
}

function workspaceSlug(cwd: string): string {
	const name = basename(cwd).toLowerCase().replace(/[^a-z0-9_-]+/g, "-")
	const hash = createHash("sha256").update(cwd).digest("hex").slice(0, 12)
	return `cline-${name || "workspace"}-${hash}`
}

function getState(sessionId: string | undefined): WorkspaceState | undefined {
	return stateBySession.get(sessionKey(sessionId)) ?? fallbackState
}

function normalizeTop(top: number | undefined): number {
	if (top === undefined) return DEFAULT_TOP
	if (!Number.isFinite(top)) return DEFAULT_TOP
	return Math.min(Math.max(Math.trunc(top), 1), MAX_TOP)
}

function stringify(value: unknown): string | undefined {
	if (typeof value === "string") return value.trim()
	if (Buffer.isBuffer(value)) return value.toString("utf8").trim()
	return undefined
}

function commandError(error: unknown): CommandError {
	if (typeof error !== "object" || error === null) {
		return { message: String(error) }
	}

	const record = error as Record<string, unknown>
	return {
		message: error instanceof Error ? error.message : "Command failed",
		stdout: stringify(record.stdout),
		stderr: stringify(record.stderr),
		code:
			typeof record.code === "string" || typeof record.code === "number"
				? record.code
				: undefined,
		signal: typeof record.signal === "string" ? record.signal : undefined,
	}
}

function errorResult(error: unknown, repoName?: string): CommandErrorResult {
	const details = commandError(error)
	return {
		error: true,
		message: details.message,
		repoName,
		stdout: details.stdout,
		stderr: details.stderr,
		code: details.code,
		signal: details.signal,
	}
}

async function kb(args: string[], cwd: string): Promise<string> {
	const result = await exec("kb", args, {
		cwd,
		maxBuffer: 5 * 1024 * 1024,
	})
	return result.stdout
}

async function ensureRegistered(cwd: string, repoName: string): Promise<void> {
	try {
		await kb(["status", "."], cwd)
		return
	} catch {
		await kb(["add", repoName, cwd], cwd)
	}
}

function ensureStateRegistered(state: WorkspaceState): Promise<void> {
	if (!state.registration) {
		state.registration = ensureRegistered(state.cwd, state.repoName).catch(
			(error: unknown) => {
				state.registration = undefined
				throw error
			},
		)
	}
	return state.registration
}

const plugin: AgentPlugin = {
	name: "kamibiki-semantic-search",
	manifest: { capabilities: ["tools", "hooks"] },

	setup(api: PluginSetupApi, ctx: PluginSetupContext) {
		const cwd = ctx.workspaceInfo?.rootPath
		if (!cwd) return

		const repoName = workspaceSlug(cwd)
		const state: WorkspaceState = {
			cwd,
			repoName,
		}
		ensureStateRegistered(state).catch(() => undefined)
		stateBySession.set(sessionKey(ctx.session?.sessionId), state)
		fallbackState = state

		api.registerTool(
			createTool<SemanticSearchInput, CommandResult | CommandErrorResult>({
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
							minLength: 1,
						},
						top: {
							type: "integer",
							description: `Number of results to return, from 1 to ${MAX_TOP}`,
							default: DEFAULT_TOP,
							minimum: 1,
							maximum: MAX_TOP,
						},
					},
					required: ["query"],
					additionalProperties: false,
				},
				timeoutMs: 120_000,
				async execute(input: SemanticSearchInput, context: AgentToolContext) {
					const { query, top } = input
					const currentState = getState(context.sessionId)
					if (!currentState) {
						return {
							error: true,
							message: "No workspace root was provided by the Cline host.",
						}
					}

					const trimmedQuery = query.trim()
					if (!trimmedQuery) {
						return {
							error: true,
							message: "Query must not be empty.",
							repoName: currentState.repoName,
						}
					}

					try {
						await ensureStateRegistered(currentState)
						const args = [
							"search",
							".",
							trimmedQuery,
							"-n",
							String(normalizeTop(top)),
						]
						const output = await kb(args, currentState.cwd)
						return { content: output, repoName: currentState.repoName }
					} catch (error) {
						return errorResult(error, currentState.repoName)
					}
				},
			}),
		)
	},

	hooks: {
		async beforeRun({ snapshot }: BeforeRunContext) {
			const currentState = getState(snapshot.sessionId)
			if (!currentState) return
			try {
				await ensureStateRegistered(currentState)
				await kb(["index", "."], currentState.cwd)
			} catch {
				// The search tool returns structured errors if kb is unavailable or unconfigured.
			}
		},
	},
}

export default plugin

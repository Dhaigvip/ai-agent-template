# ai-agent-template

Most agent demos call a tool and print the result. They don't show what
happens when a tool call needs a human to say yes first and the server
restarts mid-decision, when the token bill for describing a few dozen tools
on every turn quietly eats your margin, or when a mutation gets buried three
layers inside a sandboxed script where nothing can review it before it runs.

`ai-agent-template` is a LangGraph agent on Bedrock built around those
specific failure modes: human-in-the-loop approval that survives a restart
because the interrupt lives in the checkpoint, not the process; prompt
caching that's actually measured instead of just claimed; tools sourced
entirely from a companion MCP server instead of hand-registered one by one;
and two different code-execution tools with two different, deliberately
different, safety postures — one that can call this agent's own tools but
only the read-only ones, one that can do real math but can't touch this
agent's tools at all. It's meant to be read and adapted, not just run —
every non-obvious decision is a comment where it bites, not a design doc
you have to trust.

## What this is

- **Middleware engine, not a hand-rolled graph** — lifecycle hooks (read
  memory before the turn, write a summary after), per-turn context
  injection (memory/documents, appended ephemerally and never written back
  into checkpointed state), Bedrock prompt-cache settings, and a reactive
  context-overflow safety net that trims and retries once if a model call
  comes back too large.
- **Human-in-the-loop done properly** — an approval policy (mutating tool
  calls require explicit `approve`/`reject`), a risk classifier, and a
  separate input guardrail that runs once per turn rather than once per
  model call.
- **Tools come from MCP** — connects to the companion
  `mcp-server-template` over HTTP (`MCP_SERVER_URL`) rather than defining
  tools locally. Run that repo's `MOCK_BACKEND=1` quickstart alongside this
  one to get a fully working agent + tools setup with no external
  dependencies.
- **Prompt caching that's actually measured** — normalised token and
  tool-call accounting, because a telemetry counter that silently reads
  zero is worse than no counter at all.
- **Optional modules, every one off by default** — AgentCore-backed memory
  and persistence, Bedrock Knowledge Base retrieval, ADOT tracing/metrics,
  CloudWatch log shipping, an input guardrail, and two distinct code-running
  tools: `run_orchestration` (local, calls this agent's own tools, no math)
  and `run_python` (remote AgentCore sandbox, math/charts, no access to this
  agent's own tools — see `ARCHITECTURE.md` for the full distinction). The
  agent completes a full turn with all of them unset; each is a single env
  var away from being enabled once you've configured the underlying AWS
  resource.
- **WebSocket API** at `/api/agent`, plus `/health` and
  `/api/agent/download/{file_id}` (charts/files `run_python` produces).

## Current provider scope

**Bedrock only, currently** — not the three-provider (Bedrock/OpenAI/
Anthropic-direct) design this was originally scoped for. The provider
interface, conformance-test-suite, and OpenAI/Anthropic-direct adapters
are not yet built; `pyproject.toml` and `.env.example` both reflect
Bedrock as the only supported provider today. Multi-provider support is
the natural next step for this template, not a design dead-end — the
middleware chain above doesn't assume Bedrock specifically anywhere except
the prompt-cache middleware and the model client itself.

## Quickstart

This walks through everything needed to get a real turn working, in order —
each step says what breaks if you skip it.

### 1. Prerequisites

- **Python 3.12+**
- An **AWS account with Bedrock model access enabled** for whichever Claude
  model you plan to use, in whichever region you plan to use. This is a
  one-time per-account/per-region step done in the AWS Console (Bedrock >
  Model access) — nothing in this template can do it for you, and every
  call fails until it's done.
- A way to authenticate to AWS: an `AWS_PROFILE` (local named profile), a
  `BEDROCK_API_KEY` (bearer token), or explicit
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` (useful for
  CI runners, containers, or temporary STS credentials with no local
  profile). There's no offline fallback for the model call — the agent
  needs real credentials to start a turn.
  **`BEDROCK_API_KEY` only covers plain Bedrock model calls** — it does
  NOT authenticate AgentCore APIs (memory, code interpreter). If you plan
  to enable `AGENTCORE_MEMORY_ID` or `AGENTCORE_CODE_INTERPRETER_ID` in
  step 5, use `AWS_PROFILE` or the explicit access-key triple instead.

### 2. Install and configure

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
```

Edit `.env`:
- `AWS_REGION` — required, no default (this template deliberately never
  guesses a region for you)
- One of the credential options from step 1
- `AGENT_MODEL_ID` — a Bedrock model id. **In most regions other than
  `us-east-1`/`us-west-2` this needs to be a cross-region inference profile
  id, not the bare model id** (e.g. `eu.anthropic.claude-sonnet-4-6`, not
  `anthropic.claude-sonnet-4-6`) — see the detailed comment above
  `AGENT_MODEL_ID` in `.env.example` for how to find the right one and the
  exact error you'll hit if this is wrong (`ValidationException: The
  provided model identifier is invalid`)

Everything else in `.env.example` is optional and commented out by default
— the agent runs a complete turn with all of it unset. Leave it alone for
now; step 5 covers what each optional module needs.

### 3. Run the agent + the companion MCP server

Two processes, two terminals:

```powershell
# Terminal 1 — this repo
python -m ai_agent_template
```

```powershell
# Terminal 2 — the companion mcp-server-template, with its bundled mock
# backend so you don't need a real GraphQL API to try this
cd ..\mcp-server-template
$env:MOCK_BACKEND=1
python -m mcp_server_template
```

The agent listens on `HOST`/`PORT` (default `0.0.0.0:3002`) and needs the
MCP server reachable at `MCP_SERVER_URL` (default
`http://localhost:3001/mcp`) for any tool call to resolve — without it, the
agent still starts, but every turn that needs a tool fails.

**Prefer one click over two terminals?** `templates.code-workspace` (one
level up, alongside this repo) is a multi-root VS Code workspace with a
debug compound that launches the MCP server, this agent, and the companion
UI together — open it in VS Code, then Run and Debug → "🟢 Local — MCP +
Agent + UI".

### 4. Talk to it

The agent speaks a WebSocket protocol at `WS /api/agent` — point the
companion `agent-chat-ui-template` at it, or any WebSocket client. See
[`DEMO-QUERIES.md`](DEMO-QUERIES.md) for one ready-to-paste example query
per feature (plain MCP tool call, HITL approval, orchestration, memory,
code execution) — useful both as a manual smoke test and as a demo script.

### 5. Optional modules

AgentCore memory, Bedrock Knowledge Base retrieval, observability,
CloudWatch, and the `run_python` code-execution tool are all disabled until
their env vars are set — see `.env.example` for the full list. The two
AgentCore-backed ones (long-term memory, code execution) need you to
**provision a real AWS resource first** (not something this template
creates) — `.env.example`'s comments above `AGENTCORE_MEMORY_ID` and
`AGENTCORE_CODE_INTERPRETER_ID` walk through where to create each one and
what to paste back. Skip this section entirely for a first run — nothing
here is required to see a working turn.

## Status

Core agent loop, middleware chain, HITL, MCP tool sourcing, and the
optional modules above are implemented. Not yet built: the
provider-agnostic interface (OpenAI/Anthropic-direct support) and its
conformance test suite, and the automated test suite generally. See
`docs/adr/` for design decisions as they're made.

## License

MIT — see `LICENSE`.

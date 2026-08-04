# ai-agent-template

A production-shaped LangGraph AI agent: a middleware execution engine,
human-in-the-loop with stateless resume, tools sourced from an MCP server,
and a set of optional modules that default off until you configure them.

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
  CloudWatch log shipping, an input guardrail. The agent completes a full
  turn with all of them unset; each is a single env var away from being
  enabled once you've configured the underlying AWS resource.
- **WebSocket API** at `/api/agent`, plus a `/health` endpoint.

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

Requires **Python 3.12+**.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
# edit .env: set AWS_REGION and either AWS_PROFILE or BEDROCK_API_KEY
python -m ai_agent_template
```

Server listens on `HOST`/`PORT` (default `0.0.0.0:3002`). For tools to
resolve, also run the companion MCP server in another terminal:

```powershell
cd ..\mcp-server-template
$env:MOCK_BACKEND=1
python -m mcp_server_template
```

Every optional module (AgentCore memory, KB retrieval, observability,
CloudWatch) is disabled until its env vars are set — see `.env.example`
for the full list and what each one gates.

## Status

Core agent loop, middleware chain, HITL, MCP tool sourcing, and the
optional modules above are implemented. Not yet built: the
provider-agnostic interface (OpenAI/Anthropic-direct support) and its
conformance test suite, and the automated test suite generally. See
`docs/adr/` for design decisions as they're made.

## License

MIT — see `LICENSE`.

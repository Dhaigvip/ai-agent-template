# ai-agent-template

A production-shaped LangGraph AI agent template: a middleware execution
engine, human-in-the-loop with stateless resume, and three first-class model
providers behind one interface.

## What this is

- **Middleware engine, not a hand-rolled graph** — state + lifecycle
  middleware, dynamic tool binding, context editing and summarization
  instead of destructive message trimming.
- **A parity harness pattern** — how to replace an agent's execution engine
  behind a feature flag without breaking production behaviour, verified by
  scripted offline scenarios rather than hope.
- **Human-in-the-loop done properly** — an approval policy, runtime gating,
  and *stateless* resume through a gateway (the interrupt survives a
  restart).
- **Three first-class providers from day one** — Amazon Bedrock, OpenAI, and
  Anthropic direct, behind one provider interface, with a conformance test
  suite so all three are held to the same behaviour. Bedrock is the
  reference implementation; provider-specific extras (e.g. Bedrock
  AgentCore memory/code-interpreter) stay behind flags as optional AWS-only
  modules.
- **Prompt caching that's actually measured** — normalised token and
  tool-call accounting across providers, because a telemetry counter that
  silently reads zero is worse than no counter at all.
- **Streaming**: event streams with live token deltas.
- Connects to MCP servers for tools — see the companion
  `mcp-server-template`.

## Status

Scaffolding only. Implementation in progress — see `docs/adr/` for design
decisions as they're made.

## License

MIT — see `LICENSE`.

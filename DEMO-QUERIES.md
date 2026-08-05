# Demo queries — one per feature

Queries chosen to hit exactly one feature cleanly, so each is useful both as
a manual smoke test and as a demo/recording script. All reference data from
`mcp-server-template`'s bundled mock backend (`MOCK_BACKEND=1`) — projects
"Website Redesign" / "API Migration", members Alice Johnson / Bob Smith /
Carol White, milestones "Beta Launch" / "v2.0 Release", tasks like "Design
homepage mockup" / "Deploy to staging".

Run order: start `mcp-server-template` with `MOCK_BACKEND=1`, then
`ai-agent-template`, then talk to it over the WebSocket (`agent-chat-ui-template`,
or any WS client hitting `/api/agent`). See `templates.code-workspace` for a
one-click compound launch of all three.

---

## 1. Plain MCP tool call (read-only, no approval)

**Prereqs:** none beyond the base setup — this is the default path.

> "What tasks are in the Website Redesign project?"

**What happens:** the model calls `get_tasks` (or `get_projects` first to
resolve the project, then `get_tasks` filtered by it) directly against the
MCP server — a single straightforward tool call, auto-approved because
`risk_classifier.classify_risk()` treats `get_*` as read-only. No HITL
prompt, no sandbox. Good baseline query to show the wire protocol streaming
a tool call and its result before the final answer.

---

## 2. Human-in-the-loop (mutation requiring approval)

**Prereqs:** none.

> "Mark the 'Deploy to staging' task as complete."

**What happens:** the model resolves the task (a `get_tasks` read, still
auto-approved) then calls `update_task` — a mutation. The turn **pauses**
with a `hitl_request` wire event describing the pending call
(`toolName: "update_task"`, the args, `allowed: ["approve", "reject"]` or
`["approve", "edit", "reject"]`). Nothing executes until you send a decision
back. Good demo of three branches from the same query:
- **approve** → tool runs, turn resumes, agent confirms
- **reject** (optionally with a message, e.g. "not yet, still in QA") →
  tool does NOT run, agent adapts and continues
- **edit** the args before approving (e.g. change the target task) → the
  *edited* call runs, not the original

This is also the cleanest way to demonstrate **stateless resume**: kill and
restart the agent process between the pause and your decision (in-memory
checkpointer only survives if the process stays up — with `AGENTCORE_MEMORY_ID`
+ `AGENT_AGENTCORE_CHECKPOINTER_ENABLED=true` it survives a real restart).

---

## 3. Orchestration tool (`run_orchestration`)

**Prereqs:** `AGENT_ORCHESTRATION_ENABLED=true`. No AWS resource needed.

> "Which projects have no milestone set, and how many open tasks does each one have?"

**What happens:** answering this requires a **dependent** pair of reads —
you can't know which projects to check tasks for until you've seen which
ones lack a milestone (reshape + branch, per `run_orchestration`'s own
docstring example). Instead of two round trips (list projects → wait for
result → list tasks), the model writes a short Python script and calls
`run_orchestration` once. The wire event shows one `run_orchestration` tool
call instead of two separate `get_projects`/`get_tasks` calls.

**Good follow-up to show the HITL boundary is enforced:** ask it to also fix
what it finds in the same breath —

> "...and assign a milestone to any project missing one."

The model will run the read-only orchestration for the analysis, then issue
`update_project` calls **outside** the sandbox as normal tool calls — each
one still pausing for approval like query #2. Confirms `run_orchestration`
cannot be used to smuggle mutations past HITL (see
`agent/orchestration/tool_bridge.py`'s policy) — a `tools.update_project(...)`
call *inside* the script would instead come back as
`"Orchestration failed: 'update_project' changes data and requires human approval..."`,
which you can also demo directly by asking it to do the update via
orchestration explicitly if you want to show the refusal message itself.

---

## 4. Long-term memory across turns/sessions

**Prereqs:** real AgentCore resources — `AGENTCORE_MEMORY_ID` and
`AGENTCORE_SEMANTIC_STRATEGY_ID` both set to real values, with working AWS
credentials (see README Quickstart / blog2's prerequisite note). Without
both, `agent/memory.py`'s functions are no-ops and nothing will be
recalled — this is the one feature here that cannot be demoed against the
mock backend alone.

**Turn 1** (plant a preference):
> "By default, assign new tasks to Alice Johnson unless I say otherwise."

**Wait ~2 minutes** — `save_turn`'s docstring notes AgentCore's semantic
extraction runs asynchronously after the turn, not synchronously within it.
Recalling immediately in the same turn will not find anything yet.

**Turn 2, same or a new session** (recall):
> "Create a task called 'Fix nav overlap' in Website Redesign."

**What happens:** `EntryMiddleware` searches semantic memory before the
model runs and — once extraction has caught up — injects a
`<user_memory>` block noting the Alice Johnson preference into this turn's
context (never written into `messages`, so it doesn't compound the prompt
cache). The model should pick Alice as the assignee without being told
again, and the resulting `create_task` mutation still pauses for HITL
approval exactly like any other mutation — memory informs *what* the model
proposes, it never bypasses approval.

Good way to also demonstrate the **degradation path**: unset
`AGENTCORE_MEMORY_ID` mid-demo (or point it at a bogus id) and rerun turn 2
— the turn should complete normally with one `warning` wire event
("knowledge base"/memory unavailable) instead of failing.

---

## 5. Code interpreter (`run_python`)

**Prereqs:** `AGENTCORE_CODE_INTERPRETER_ID` set to a real AgentCore Code
Interpreter resource, with working AWS credentials (same requirement as
memory in #4 above). No `FeatureFlags` entry to flip — id presence is the
sole gate, same pattern as memory/KB.

> "Plot the number of tasks per project as a bar chart."

**What happens:** the model first calls `get_tasks`/`get_projects` (plain
MCP reads) to get the data, then passes those rows as a Python literal into
`run_python` — one self-contained call that builds a pandas DataFrame,
aggregates, and saves a Plotly figure as `chart.plotly.json`. The server
starts a fresh AgentCore sandbox, runs the code, downloads the produced
file, and stops the sandbox immediately (self-contained lifecycle — nothing
lingers or keeps billing). The tool result contains ready-to-paste markdown
(`![chart.plotly.json](/api/agent/download/<id>)`); the model pastes it into
its final answer and the chart renders inline.

**Good contrast query, same turn or a follow-up**, to demonstrate the tool
is NOT a general-purpose data lookup:

> "Which tasks are overdue?"

This should NOT invoke `run_python` at all — it's a plain filter over
`get_tasks`, and the tool's own docstring gate ("does the answer require
MATH or a VISUAL?") tells the model to answer directly instead. If you see
`run_python` fire for a query like this, that's a docstring/prompt tuning
gap worth flagging, not expected behavior.

**Also good for showing graceful degradation:** point
`AGENTCORE_CODE_INTERPRETER_ID` at a real id but revoke/expire the AWS
credentials, then rerun the chart query — the turn should complete with a
`warning` wire event ("code execution is disabled...") instead of failing,
same pattern as the memory/KB degradation paths.

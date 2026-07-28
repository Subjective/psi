# Psi Design Doc

Status: Working draft

## Goal

Psi is a minimal terminal coding agent written in Rust, built to study speculative tool execution.

It has two goals, in priority order:

1. Answer a research question: under fixed prediction and execution budgets, when does speculatively executing predicted tool calls reduce agent latency without hurting aggregate throughput? In particular, compare branch-sampling and direct-proposal prediction under equal budgets.
2. Be a good standalone terminal agent: minimal in the spirit of Pi (github.com/earendil-works/pi, a minimal terminal agent whose tiny-core philosophy Psi follows), terminal-native, with immutable history, branching by editing past messages, and an eventually strong Vim composer.

Non-goals for now: plugins, MCP, sub-agents, plan mode, an OS sandbox, and provider breadth beyond OpenAI and vLLM.

## Data Model

The harness owns all durable state. Interfaces send commands and render events. One rule keeps the model small: **a session is an append-only tree of items.**

| Entity | Purpose |
| --- | --- |
| `Session` | One durable conversation: metadata, an append-only item log, and a `head` pointer. |
| `Item` | One typed record with `id`, `parent_id`, `turn_id`, and timestamps. |
| `WorkspaceRevision` | Counter bumped after every successful workspace mutation. Scopes cached tool results. |

Everything else is derived, not stored:

- The active conversation is the path from the root to `head`.
- A branch is any leaf. `head` may point at any item; submitting a message when `head` already has children starts a new branch, and the old path is untouched. Editing a past message is interface sugar: `set_head` to the item before it, then submit the revised text.
- A turn is the span of items sharing a `turn_id`: one user message through the assistant response that ends it. Turns exist for grouping and timing, not as records.

Item kinds:

```text
user_message
assistant_message
reasoning
tool_call
tool_result
```

Notes:

- `tool_call` carries the tool name, canonical arguments, call ID, working directory, and workspace revision.
- `tool_result` carries content, status, duration, and truncation info.
- Diffs render from `apply_patch` tool calls; they are not separate items.
- Errors are recorded on the item that failed and on the turn-finished event; they are not separate items.
- Items may carry opaque provider data (for example encrypted reasoning) so turns replay correctly. The harness never depends on its contents.
- Speculative executions never enter session history. They exist only in the speculation runtime and in traces.

### Interface protocol

Commands:

```text
create_session | load_session | list_sessions
submit_message
cancel_turn
set_head
```

Every event carries a monotonic sequence number, a timestamp, and its session ID (`sessions_listed`, which spans sessions, carries none), so trace export is an assembly step rather than a retrofit.

```text
session_created | session_loaded | sessions_listed
turn_started
item_started | item_delta | item_finished
turn_finished
```

`session_loaded` includes an item-tree snapshot. `item_finished` and `turn_finished` carry a status — `completed | cancelled | failed` — and, when failed, an error message; a turn that fails before any item starts records the error on `turn_finished` alone. `turn_finished` also carries the tokens billed across the turn's model responses, which is what the Milestone 5 baselines cost a run by. Durable items are complete records; streaming deltas exist only in the event protocol.

The in-process TUI and future external clients consume the same logical protocol.

### Model boundary

Psi uses provider-neutral request and event types: `TurnRequest` in, a stream of `ModelEvent` out (text and reasoning deltas, reasoning completion, tool-call argument deltas, completed tool calls, usage, completion, errors). The harness never depends on provider wire types; replay-critical provider data passes through as opaque item fields, carried out of the stream by reasoning completion. Dropping the stream cancels the request.

### Speculative cache

A cache key is:

```text
tool name + canonical arguments + working directory + workspace revision
```

An entry holds the in-flight future or the finished result, so an authoritative call can adopt work that has not completed yet. Entries are invalidated when the workspace revision changes and dropped when the turn ends, so the cache never outlives a turn and edits made outside Psi between turns cannot poison it. Outside edits during a turn are an accepted gap.

The revision bumps after every successful `apply_patch` and after every `exec`: its effects are unknown, so Psi conservatively assumes mutation.

### Speculation

Most of a turn's wall time is the authoritative model generating its next response. Speculation uses that time: while the model generates, Psi asks the predictor which tool call is likely to come next, executes the top few guesses, and parks each result in the speculative cache. When the real call arrives, reconciliation is an exact cache-key lookup: a hit returns the finished result immediately or awaits the in-flight future; a miss executes normally. Unused entries remain usable for later calls in the same turn until a mutation invalidates them.

A prediction is just a guessed tool call. The prediction strategy — direct proposal or branch sampling — is per-run configuration, not per-prediction metadata. Each strategy hands the runtime an ordered, deduplicated list of calls; the runtime keeps those that pass the read-only allowlist and are not already cached, and executes the first few.

Two configuration numbers bound each round of speculation (one round per authoritative model response), and they are the independent variables of the research question: the prediction budget (predictor tokens spent guessing) and the execution budget (concurrent speculative calls — the fanout). Fixing both is what makes the strategies comparable, because branch sampling spends far more predictor tokens per guess than direct proposal. The core metrics are hit rate (the fraction of authoritative calls served from the cache) and net latency change.

The replay oracle is a fake predictor used only in tests. Replaying a recorded session against a fixture workspace, it "predicts" exactly the call the recording makes next, so it is always right. It measures the ceiling — the savings available when prediction is perfect — which tells us whether speculation can pay at all before any real predictor exists.

Speculation adds no interface events. It is recorded through the speculation runtime's own trace log (predictions, executions, hits, misses, wasted work), which shares the event stream's clock and sequence space.

## Design Decisions

### Rust and Tokio

A Rust workspace with Tokio for model streams, tool execution, cancellation, and interface events. One distributable binary and precise timing control, at the cost of slower UI iteration than TypeScript.

Two crates to start: `psi-core` (the harness — items, sessions, protocol types, turn engine, tools, model backends, speculation) and `psi` (the binary — CLI, headless mode, TUI). A crate splits out only when it gains a second consumer; for example, the protocol types become their own crate when an external transport arrives.

### The harness is the source of truth

The TUI, headless mode, and future clients are projections over harness commands and events. Branching, persistence, and speculation are never TUI-only behavior.

### In-process protocol first

The first TUI and harness run in one process over typed Rust channels. The protocol is client-server-shaped so JSONL over stdio can be added later as a transport, not a second agent implementation.

### Terminal-native TUI and composer

Use Crossterm and Ratatui, rendering inline so terminal scrollback is preserved and Psi feels part of the user's shell. Keep the rendering boundary replaceable in case a custom line-diff renderer is needed; a fullscreen mode may be added later for views that need the whole viewport, such as branch navigation.

The composer is Psi's own modal state machine over a `ropey` text buffer, structured from day one around Vim's grammar — `[count] operator [count] (motion | text object)` — so later Vim features are table entries rather than rewrites. Existing widget crates are not used: they own rendering, and none models the grammar. modalkit is the semantic reference, and its core is the fallback dependency if hand-rolling stalls. The MVP needs only reliable multiline editing, history, and basic normal/insert modes; once Vim support grows past that, correctness is checked differentially against headless Neovim — identical keystrokes into both, comparing buffer, cursor, mode, and registers including register type.

### Persistence: append-only JSONL per session

One file per session, one item per line, plus the `head` pointer. History is never mutated; edits fork. This costs storage but buys free branching, trivial resume, greppable sessions, and reproducible replay. SQLite only if search or scale later demands it.

A file is a header line of session metadata followed by the only two mutations a session tree has: an item is appended, or `head` moves. Appending an item always moves `head` onto it, so an item line carries its own head move and only `set_head` needs a line of its own; on load, the last line that touches `head` wins. Items are written as they finish rather than at the end of a turn, so a crash costs at most the record being written.

A log is valid up to its first defect: a final line with no terminating newline, a line that does not parse, or a record that contradicts the tree. Loading drops everything from there on and truncates the file to that point, so the prefix a reader accepted is exactly what the next append extends, and a record that survives can never reference one that did not.

### Five tools, one profile

Default tools: `read_file`, `list_directory`, `search`, `apply_patch`, `exec`.

Structured read-only tools exist because they are the unit of speculation: canonical arguments, declared effects, cache keys, and per-tool latency stats. `exec` is the general escape hatch. The registry may hold more tools than the active profile; only the profile is advertised to the model, and the agent and predictor always share the same profile so their calls are comparable.

The v0 speculative allowlist is `read_file`, `list_directory`, `search`. A planned experiment: a shell-minimal profile (`read_file`, `apply_patch`, `exec`) to measure whether structured tools earn their schema cost through speculative coverage.

### Trusted environment and hooks

Psi follows Pi: it assumes a trusted or externally sandboxed environment and ships no OS sandbox, no permission rules, and no confirmation prompts.

- Structured file tools enforce workspace roots in-process.
- `exec` inherits Psi's process permissions; run Psi in a container for untrusted work, and Psi says so plainly.
- A small serial in-process hook seam (compiled-in Rust, registered at harness construction) runs before and after every authoritative tool call; selection also applies before-hooks to predictions, so a call a hook would block is never executed speculatively. A hook returns continue or block; a block is reported to the model as a refused call. This seam is where any future policy would attach, including interactive approval if it is ever wanted.

### Model backends: one Responses codec, explicit capabilities

The internal model boundary stays provider-neutral. Behind it, both backends speak the OpenAI Responses protocol through one shared codec:

- OpenAI requires Responses for reasoning-item replay (encrypted reasoning with `store: false`); it is also the recommended surface for cache utilization and tool calling.
- vLLM's `/v1/responses` endpoint is general-purpose and reuses the same tool and reasoning parsers as its Chat endpoint, so a second codec would buy an older event-assembly layer, not better parsing.

The two backends share a wire format, not a capability set. Each configured model target carries a capability descriptor for the differences Psi branches on — encrypted-reasoning replay (OpenAI only; vLLM rejects it) and provider-side compaction — plus per-target quirks (reasoning-effort semantics, `tool_choice` breadth, stream-terminating error events) recorded as they are found. The features vLLM's Responses path lacks (`store`, `previous_response_id`, compaction) are features Psi does not depend on: Psi owns history and sends stateless requests, and provider-side storage is never authoritative.

Two required guards: a vLLM response that returns zero tool calls when tools were supplied is a configuration error (a missing tool parser fails silently), and vLLM cancellation is a connection drop, not the cancel endpoint.

A Chat Completions fallback exists only as a predictor-side config switch, for model-parser combinations whose streaming misbehaves under Responses.

### Compaction

Session history is immutable; compaction never rewrites it. Compaction produces a derived context checkpoint (a portable summary plus a recent tail) used when building model requests. Checkpoints live in session metadata, not as items, and are invalidated when `head` leaves the compacted path. Provider-side compaction (OpenAI `/responses/compact`) is an optional experiment behind a capability flag. Out of MVP scope; scheduled in Milestone 8, before long-session benchmarks.

### Speculation is optional middleware

The baseline agent loop works without speculation. Speculation observes a turn, executes allowlisted predicted calls, and reconciles them when the authoritative call arrives. v0 speculates only read-only tools at a fixed workspace revision; writes remain authoritative.

### Tool speculation in Psi, token speculation in vLLM

Speculative tool execution depends on tool semantics and agent state, so it lives in Psi. Token-level speculative decoding lives in vLLM; Psi configures and measures it but does not implement draft-token verification.

### Open questions

- Whether Ratatui's inline rendering is sufficient or a custom line-diff renderer is needed.
- Which open models have reliable enough tool calling to serve as the predictor.
- How much provider-specific data must be retained to replay reasoning-model turns correctly.
- Whether branch navigation should temporarily enter fullscreen mode.
- Whether multi-step branch rollouts — executing a branch's guessed reads so it can predict further ahead — can pay for their compounded divergence and cost.

## Milestones

### 1. Skeleton and protocol

- Create the Rust workspace: `psi-core` and the `psi` binary.
- Define items, sessions, commands, and events; every event sequenced and timestamped from day one.
- Add a fake model and fake tools for deterministic tests.

Verified when a headless test drives a complete fake turn and asserts the exact event sequence.

### 2. Baseline model-tool loop

- Implement the Responses codec and the OpenAI backend: streaming, cancellation, timeouts.
- Implement the five tools with canonical arguments and declared effects; bound all output.
- Add the hook seam and workspace-root enforcement for structured tools.

Verified when the headless agent can inspect a fixture repository, change a file, run a test, and finish; a structured-tool access outside the workspace root is refused; and a blocking hook surfaces to the model as a refused call.

### 3. Persistence and branching

- Persist sessions as JSONL item trees; resume on restart.
- Implement `set_head` and branching by submitting under a non-leaf `head`.

Verified when restarting Psi preserves the tree and both sides of a fork can be resumed independently.

Milestones 4 and 5 are independent; order them by whichever goal is more pressing.

### 4. Standalone TUI (MVP boundary)

- Inline terminal-native rendering: streaming messages, tool activity, diffs.
- Multiline composer with history, cancellation, and basic Vim normal/insert modes.
- Edit a past message to fork, and switch between branches.
- Clean terminal restore on exit and interruption.

Verified when a user can, entirely from the TUI: start a session, stream a response, watch a tool run, view a diff, cancel a turn, edit a past message to fork, switch branches, and exit with the terminal restored.

### 5. Traces and baselines

- JSONL trace export assembled from the existing event stream.
- Deterministic benchmark tasks with configurable artificial tool latency.
- Non-speculative latency and success baselines.

Verified when a run can be reconstructed from its trace and compared across repeated trials.

### 6. Speculation runtime and replay oracle

- Normalization, allowlist filtering, selection, execution, and reconciliation of predicted calls.
- Cache in-flight futures under workspace-aware keys.
- Speculation records (predictions, executions, hits, misses, wasted work) join the trace on the event stream's clock.
- Drive it first with the replay oracle: this measures the attainable upper bound before any predictor exists.

Verified when an oracle run shows measured latency reduction against the Milestone 5 baseline with agent-visible results unchanged.

### 7. Prediction strategies

- Bring up the vLLM backend behind the shared codec: capability descriptor, the zero-tool-call parser guard, cancellation by connection drop, and an end-to-end streaming tool-call smoke test.
- Direct proposal (ask the predictor for the most likely next tool calls) and single-step branch sampling (sample short continuations, keep the tool calls in each one's first response).
- Compare under equal prediction and execution budgets; repeat under the shell-minimal profile to test whether structured tools earn their schema cost.

Verified when benchmark reports include hit rate, predictor cost, wasted work, and net latency change.

### 8. vLLM systems experiments

- Self-host both models; measure batching, prefix caching, concurrency, and contention.
- Add context checkpoint compaction, needed once benchmark sessions grow long.
- Compare fixed and adaptive speculation policies.
- Token-level speculative decoding only through vLLM configuration or a focused engine extension.

Verified when experiments reproduce from configuration and report aggregate throughput (tasks completed per GPU-hour under a concurrent workload) alongside per-task latency, identifying where speculation helps or hurts.

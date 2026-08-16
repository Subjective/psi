# Psi

## Overview

Psi is a small terminal coding agent written in Rust. I built it partly to
understand the pieces of an agentic loop from the ground up, and partly to
explore whether predicting tool calls can hide some of the time agents spend
waiting on tools.

The agent has a terminal UI, a minimal set of tools inspired by
[Pi](https://github.com/earendil-works/pi), and supports headless operation. The
benchmark harness can record and replay real agent trajectories, then compare
direct tool-call prediction with branch sampling under the same prediction and
execution budgets.

Psi assumes a trusted workspace and does not provide an OS sandbox. Use a
container or another sandbox when working with untrusted code.

## Getting started

You will need Rust 1.88 or newer and an OpenAI API key. From the repository
root, set the key and build the workspace:

```sh
export OPENAI_API_KEY="..."
cargo build --workspace
```

Run `psi` without a prompt to open the terminal UI:

```sh
cargo run -p psi
```

Pass a prompt to run a single turn without opening the UI, or use `--continue`
to resume the most recent session:

```sh
cargo run -p psi -- "Summarize this codebase"
cargo run -p psi -- --continue
```

Run the full test suite with:

```sh
cargo test --workspace
```

Run a baseline benchmark:

```sh
cargo run -p psi-core --bin psi-bench -- --trials 5
```

Compare direct prediction and branch sampling against a local vLLM endpoint:

```sh
cargo run -p psi-core --bin psi-bench -- \
  --predictor-url http://localhost:8000/v1 \
  --predictor-model MODEL \
  --predictor-chat \
  --strategy direct \
  --strategy branch \
  --prediction-budget 256 \
  --execution-budget 2 \
  --samples 4
```

Optional environment variables are `PSI_MODEL`, `PSI_BASE_URL`, and
`PSI_SESSIONS_DIR`. Run `cargo run -p psi -- --help` or
`cargo run -p psi-core --bin psi-bench -- --help` for the full command help.

## Experiment

I recorded real GPT-5.6 Luna/max coding-agent trajectories, then replayed each
trajectory with and without speculative tool execution so every comparison saw
the same model responses and tool latencies. The confirmatory suite used 18
controlled code-navigation tasks (six each) on pinned snapshots of the
Kubernetes, Rust, and VS Code repositories. Predictor models ran live through
vLLM on rented RunPod H100s; the harness and repository tools ran locally. Each
condition used two replay blocks, for 288 scheduled comparisons and 287
completed comparisons.

| Predictor | Strategy | vLLM prefix cache | Mean saved per task | Speedup | 95% CI for time saved |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | Direct | Off | 655 ms | 4.63% | 277–1,061 ms |
| Qwen3.5-4B | Branch | Off | 736 ms | 5.20% | 361–1,134 ms |
| Qwen3.5-4B | Direct | On | 726 ms | 5.14% | 328–1,140 ms |
| Qwen3.5-4B | Branch | On | 750 ms | 5.31% | 376–1,135 ms |
| Qwen3.5-35B-A3B-FP8 | Direct | Off | 881 ms | 6.23% | 425–1,360 ms |
| Qwen3.5-35B-A3B-FP8 | Branch | Off | 811 ms | 5.73% | 392–1,243 ms |
| Qwen3.5-35B-A3B-FP8 | Direct | On | 755 ms | 5.23% | 333–1,219 ms |
| Qwen3.5-35B-A3B-FP8 | Branch | On | 944 ms | 6.57% | 447–1,507 ms |

All eight conditions were faster than their paired baselines and produced the
same agent-visible results. Tool-result cache hit rates were 23–27%, with most
useful hits coming from repository search.

Psi speculates only on the structured read-only tools `search`, `read_file`, and
`list_directory`; writes remain under the main model's control. This makes the
experiment safe and repeatable, but the controlled tasks and narrow argument
space may make calls easier to predict than open-ended coding work. vLLM prefix
caching reuses predictor prompt computation; it is separate from Psi's
turn-scoped cache of speculative tool results.

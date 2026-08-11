# Psi

## Overview

Psi is a small terminal coding agent written in Rust. I built it partly to
understand the pieces of an agentic loop from the ground up, and partly to
explore whether predicting tool calls can hide some of the time agents spend
waiting on tools.

The agent has a terminal UI, a headless mode, structured file tools, and durable
branching sessions. The benchmark harness can record and replay real agent
trajectories, then compare direct tool-call prediction with branch sampling
under the same prediction and execution budgets.

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

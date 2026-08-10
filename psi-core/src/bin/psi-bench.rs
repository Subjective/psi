//! The benchmark runner (docs/design.md, Milestones 5 and 7). Runs every task
//! repeatedly against fixture workspaces with injected tool latency, writes one
//! trace per trial, and prints the aggregated report; naming strategies adds a
//! speculating run of each, reported against that task's own baseline.
//!
//! It is a binary of the library crate rather than of `psi`, so
//! `cargo install --path psi` never ships it.
//!
//! The Milestone 7 comparison — both strategies under equal budgets, over the
//! default and the shell-minimal profiles — is one invocation, because the
//! shell-minimal task runs alongside the others:
//!
//! ```text
//! cargo run -p psi-core --bin psi-bench -- \
//!     --predictor-url http://localhost:8000/v1 \
//!     --strategy direct --strategy branch \
//!     --prediction-budget 256 --execution-budget 2 --samples 4
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use psi_core::bench::{
    BenchConfig, Comparison, Speculation, Strategy, record_task, recorded_task, run_task, tasks,
};
use psi_core::openai::{OpenAiBackend, OpenAiConfig};
use psi_core::vllm::{Endpoint, VllmConfig};

const USAGE: &str = "\
usage: psi-bench [--trials N] [--dir PATH] [--recorded DIR]... [--strategy NAME]...
                 [--prediction-budget N] [--execution-budget N] [--samples N]
                 [--predictor-url URL] [--predictor-model NAME] [--predictor-chat]
       psi-bench --record OUT --fixture DIR --task-name NAME --prompt TEXT...

  --trials N            how many times to run each task (default 5)
  --dir PATH            where traces, fixture workspaces, and sessions go
                        (default target/psi-bench/<start time>)
  --recorded DIR        run this recording instead of the hand-written tasks;
                        may be repeated. A recording replays with its own
                        generation delays and tool durations, so --trials and
                        the injected latency knobs do not shape it.
  --strategy NAME       also run each task under this prediction strategy and
                        report it against the baseline; may be repeated.
                        oracle | direct | branch
  --prediction-budget N predictor tokens one round may spend guessing
                        (default 256)
  --execution-budget N  concurrent speculative calls per round (default 2)
  --samples N           continuations per round for branch sampling (default 4)
  --predictor-url URL   the vLLM base url direct and branch ask, e.g.
                        http://localhost:8000/v1
  --predictor-model N   the served model name, when the server needs one
  --predictor-chat      send prediction requests through /v1/chat/completions
                        instead of /v1/responses

recording (drives the live agent once and stores the run as a benchmark task):
  --record OUT          where the recording directory is created; must not exist
  --fixture DIR         the workspace the live run starts from
  --task-name NAME      the recorded task's name in reports and trace files
  --prompt TEXT         one user message per turn; may be repeated
                        environment: OPENAI_API_KEY (required), PSI_MODEL,
                        PSI_BASE_URL";

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = BenchConfig::default();
    let mut dir = None;
    let mut names: Vec<String> = Vec::new();
    let mut prediction_budget = 256;
    let mut execution_budget = 2;
    let mut samples = 4;
    let mut predictor = VllmConfig::default();
    let mut predictor_url = None;
    let mut recorded: Vec<PathBuf> = Vec::new();
    let mut record_out: Option<PathBuf> = None;
    let mut record_fixture: Option<PathBuf> = None;
    let mut record_name: Option<String> = None;
    let mut record_prompts: Vec<String> = Vec::new();

    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        // Every flag below takes exactly one value, so one helper reads them
        // all and a missing value is one message.
        let mut value = || rest.next().ok_or_else(|| format!("{arg} needs a value"));
        let parsed = match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--trials" => value().and_then(|n| {
                config.trials = n.parse().map_err(|_| format!("{arg} needs a number"))?;
                Ok(())
            }),
            "--dir" => value().map(|path| dir = Some(PathBuf::from(path))),
            "--strategy" => value().map(|name| names.push(name.clone())),
            "--prediction-budget" => value().and_then(|n| {
                prediction_budget = n.parse().map_err(|_| format!("{arg} needs a number"))?;
                Ok(())
            }),
            "--execution-budget" => value().and_then(|n| {
                execution_budget = n.parse().map_err(|_| format!("{arg} needs a number"))?;
                Ok(())
            }),
            "--samples" => value().and_then(|n| {
                samples = n.parse().map_err(|_| format!("{arg} needs a number"))?;
                Ok(())
            }),
            "--predictor-url" => value().map(|url| predictor_url = Some(url.clone())),
            "--predictor-model" => value().map(|model| predictor.model = model.clone()),
            "--predictor-chat" => {
                predictor.endpoint = Endpoint::ChatCompletions;
                Ok(())
            }
            "--recorded" => value().map(|path| recorded.push(PathBuf::from(path))),
            "--record" => value().map(|path| record_out = Some(PathBuf::from(path))),
            "--fixture" => value().map(|path| record_fixture = Some(PathBuf::from(path))),
            "--task-name" => value().map(|name| record_name = Some(name.clone())),
            "--prompt" => value().map(|prompt| record_prompts.push(prompt.clone())),
            other => Err(format!("unknown argument: {other}")),
        };
        if let Err(message) = parsed {
            return fail(&message);
        }
    }

    // Recording is its own mode: drive the live agent once, store the run,
    // and exit. The stored recording replays via --recorded.
    if let Some(out) = record_out {
        let (Some(fixture), Some(name)) = (record_fixture, record_name) else {
            return fail("--record needs --fixture and --task-name");
        };
        if record_prompts.is_empty() {
            return fail("--record needs at least one --prompt");
        }
        let mut model = OpenAiConfig::default();
        if let Ok(name) = std::env::var("PSI_MODEL") {
            model.model = name;
        }
        if let Ok(base_url) = std::env::var("PSI_BASE_URL") {
            model.base_url = base_url;
        }
        let backend = match OpenAiBackend::new(model) {
            Ok(backend) => backend,
            Err(err) => return fail(&err.to_string()),
        };
        return match record_task(&name, &fixture, &record_prompts, Arc::new(backend), &out).await {
            Ok(()) => {
                println!(
                    "recorded {name} at {}\nreplay it with: psi-bench --recorded {}",
                    out.display(),
                    out.display()
                );
                ExitCode::SUCCESS
            }
            Err(err) => fail(&format!("recording failed: {err}")),
        };
    }

    if let Some(url) = predictor_url {
        predictor.base_url = url;
    } else if names.iter().any(|name| name != "oracle") {
        return fail("direct and branch need --predictor-url");
    }
    let strategies: Vec<Strategy> = match names
        .iter()
        .map(|name| match name.as_str() {
            "oracle" => Ok(Strategy::Oracle),
            "direct" => Ok(Strategy::Direct {
                predictor: predictor.clone(),
            }),
            "branch" => Ok(Strategy::Branch {
                predictor: predictor.clone(),
                samples,
            }),
            other => Err(format!("unknown strategy: {other}")),
        })
        .collect()
    {
        Ok(strategies) => strategies,
        Err(message) => return fail(&message),
    };

    // Recordings replace the hand-written tasks when named: the flag says
    // what to run.
    let bench_tasks = if recorded.is_empty() {
        tasks()
    } else {
        match recorded.iter().map(|dir| recorded_task(dir)).collect() {
            Ok(tasks) => tasks,
            Err(err) => return fail(&format!("loading a recording: {err}")),
        }
    };

    let dir = dir.unwrap_or_else(default_dir);
    println!("traces: {}\n", dir.display());
    for task in &bench_tasks {
        let baseline = match run_task(task, &config, &dir.join("baseline")).await {
            Ok(report) => report,
            Err(err) => return fail(&format!("{}: {err}", task.name)),
        };
        if strategies.is_empty() {
            println!("{baseline}");
        }
        for (index, strategy) in strategies.iter().enumerate() {
            let mut speculated = config.clone();
            speculated.speculate = Some(Speculation {
                strategy: strategy.clone(),
                prediction_budget,
                execution_budget,
            });
            // One directory per strategy, so a comparison's two runs never
            // overwrite each other's traces.
            let into = dir.join(format!("{}-{index}", names[index]));
            let report = match run_task(task, &speculated, &into).await {
                Ok(report) => report,
                Err(err) => return fail(&format!("{} ({}): {err}", task.name, names[index])),
            };
            println!(
                "{}",
                Comparison {
                    baseline: baseline.clone(),
                    speculated: report,
                }
            );
        }
    }
    ExitCode::SUCCESS
}

/// Under `target/` so a benchmark's artifacts are never mistaken for the
/// user's sessions and `cargo clean` sweeps them; one directory per
/// invocation, so successive runs keep their traces apart.
fn default_dir() -> PathBuf {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis();
    PathBuf::from("target/psi-bench").join(started.to_string())
}

fn fail(message: &str) -> ExitCode {
    eprintln!("psi-bench: {message}\n{USAGE}");
    ExitCode::FAILURE
}

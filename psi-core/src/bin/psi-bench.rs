//! The baseline benchmark runner (docs/design.md, Milestone 5). Runs every
//! task repeatedly against fixture workspaces with injected tool latency,
//! writes one trace per trial, and prints the aggregated report.
//!
//! It is a binary of the library crate rather than of `psi`, so
//! `cargo install --path psi` never ships it.
//!
//! ```text
//! cargo run -p psi-core --bin psi-bench -- [--trials N] [--dir PATH]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use psi_core::bench::{BenchConfig, run_task, tasks};

const USAGE: &str = "\
usage: psi-bench [--trials N] [--dir PATH] [--speculate [N]]

  --trials N      how many times to run each task (default 5)
  --dir PATH      where traces, fixture workspaces, and sessions go
                  (default target/psi-bench/<start time>)
  --speculate [N] also run each task with the replay oracle at execution
                  budget N (default 4) and report the change against the
                  baseline";

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = BenchConfig::default();
    let mut dir = None;
    let mut speculate = None;

    let mut rest = args.iter().peekable();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--trials" => match rest.next().and_then(|n| n.parse().ok()) {
                Some(trials) => config.trials = trials,
                None => return fail("--trials needs a number"),
            },
            "--dir" => match rest.next() {
                Some(path) => dir = Some(PathBuf::from(path)),
                None => return fail("--dir needs a path"),
            },
            "--speculate" => {
                let budget = match rest.peek().and_then(|n| n.parse().ok()) {
                    Some(budget) => {
                        rest.next();
                        budget
                    }
                    None => 4,
                };
                speculate = Some(budget);
            }
            other => return fail(&format!("unknown argument: {other}")),
        }
    }

    let dir = dir.unwrap_or_else(default_dir);
    println!("traces: {}\n", dir.display());
    for task in tasks() {
        let baseline = match run_task(task, &config, &dir.join("baseline")).await {
            Ok(report) => report,
            Err(err) => return fail(&format!("{}: {err}", task.name)),
        };
        println!("{baseline}");
        // The oracle run measures the ceiling: the latency reduction available
        // when prediction is perfect (docs/design.md, "Speculation").
        if let Some(budget) = speculate {
            let mut speculated = config.clone();
            speculated.speculate = Some(budget);
            let report = match run_task(task, &speculated, &dir.join("speculated")).await {
                Ok(report) => report,
                Err(err) => return fail(&format!("{} (speculated): {err}", task.name)),
            };
            println!("{report}");
            println!(
                "  oracle ceiling: median turn {}ms -> {}ms ({}ms saved)\n",
                baseline.turn_ms.median_ms,
                report.turn_ms.median_ms,
                baseline.turn_ms.median_ms as i64 - report.turn_ms.median_ms as i64,
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

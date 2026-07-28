//! Milestone 7's verification: the two prediction strategies against a
//! scripted vLLM server — what they ask for, what they make of the answer,
//! what they do when the answer never comes — and a benchmark report that
//! carries hit rate, predictor cost, wasted work, and net latency change.
//!
//! The scripted streams are shaped the way vLLM emits them, the same shapes
//! `vllm_backend.rs` uses: a function call arrives as
//! `response.output_item.added`, then `response.output_item.done`, and
//! `response.completed` carries the usage a round is billed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use psi_core::bench::{
    BenchConfig, Comparison, Latency, Speculation, SpeculationStats, Strategy, run_task, run_trial,
    tasks,
};
use psi_core::item::{CompletionStatus, Item, ItemId, ItemPayload, TurnId};
use psi_core::model::{ModelBackend, Sampling, TurnRequest};
use psi_core::predictor::{BranchSampling, DirectProposal};
use psi_core::session::SessionId;
use psi_core::speculation::Predictor;
use psi_core::tool::ToolSpec;
use psi_core::trace::{RunTrace, TraceRecord};
use psi_core::vllm::{Endpoint, VllmBackend, VllmConfig};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A local server that answers every request with the next canned body,
/// cycling once the list runs out, and keeps what it read. Cycling is what lets
/// one scripted predictor serve a whole benchmark run, whose round count is a
/// property of the task rather than of the test.
struct Server {
    base_url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn count(&self) -> usize {
        self.requests.lock().expect("requests").len()
    }

    /// The target of the nth request's request line, which is how a test tells
    /// the two endpoints apart.
    fn path(&self, index: usize) -> String {
        let requests = self.requests.lock().expect("requests");
        requests[index]
            .split_whitespace()
            .nth(1)
            .expect("request line")
            .to_string()
    }

    fn body(&self, index: usize) -> Value {
        let requests = self.requests.lock().expect("requests");
        let body = requests[index].split("\r\n\r\n").nth(1).expect("body");
        serde_json::from_str(body).expect("json body")
    }
}

async fn serve(responses: Vec<String>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = requests.clone();
    tokio::spawn(async move {
        let mut next = 0;
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            // The body follows the headers in the same stream; every request
            // here declares a content-length, so read until it is satisfied.
            loop {
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&chunk[..read]),
                }
                if request_is_complete(&request) {
                    break;
                }
            }
            if request.is_empty() {
                continue;
            }
            recorded
                .lock()
                .expect("requests")
                .push(String::from_utf8_lossy(&request).into_owned());
            let response = responses[next % responses.len()].clone();
            next += 1;
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
            // Dropping the socket closes the connection, which ends the stream.
        }
    });
    Server { base_url, requests }
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(head_end) = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
    else {
        return false;
    };
    let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
    let length: usize = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0);
    request.len() >= head_end + length
}

fn frame(event: &Value) -> String {
    format!("data: {event}\n\n")
}

/// One Responses stream making the given calls and billing the given tokens.
fn stream(calls: &[(&str, Value)], output_tokens: u64) -> String {
    let mut body = String::new();
    for (index, (tool, arguments)) in calls.iter().enumerate() {
        let item = json!({
            "id": format!("fc_{index}"),
            "type": "function_call",
            "call_id": format!("call_{index}"),
            "name": tool,
            "arguments": arguments.to_string(),
        });
        body.push_str(&frame(&json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": item,
        })));
        body.push_str(&frame(&json!({
            "type": "response.output_item.done",
            "output_index": index,
            "item": item,
        })));
    }
    body.push_str(&frame(&json!({
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "status": "completed",
            "usage": { "input_tokens": 100, "output_tokens": output_tokens },
        },
    })));
    format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{body}")
}

/// One whole Chat Completions reply, the shape the predictor's fallback reads.
fn chat_reply(calls: &[(&str, Value)]) -> String {
    let tool_calls: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(index, (tool, arguments))| {
            json!({
                "id": format!("call_{index}"),
                "type": "function",
                "function": { "name": tool, "arguments": arguments.to_string() },
            })
        })
        .collect();
    let body = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "", "tool_calls": tool_calls },
            "finish_reason": "tool_calls",
        }],
        "usage": { "prompt_tokens": 120, "completion_tokens": 24 },
    })
    .to_string();
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn predictor_config(base_url: &str) -> VllmConfig {
    VllmConfig {
        base_url: base_url.to_string(),
        model: "Qwen/Qwen3-8B".to_string(),
        request_timeout: Duration::from_secs(5),
        idle_timeout: Duration::from_secs(5),
        ..VllmConfig::default()
    }
}

fn backend(config: VllmConfig) -> Arc<dyn ModelBackend> {
    Arc::new(VllmBackend::new(config).expect("backend"))
}

/// The tool profile both the agent and the predictor are given.
fn profile() -> Vec<ToolSpec> {
    ["read_file", "search"]
        .into_iter()
        .map(|name| ToolSpec {
            name: name.to_string(),
            description: format!("{name} something"),
            parameters: json!({ "type": "object" }),
        })
        .collect()
}

/// The authoritative request a strategy is handed: one user message and the
/// shared profile.
fn context() -> TurnRequest {
    TurnRequest {
        session_id: SessionId("s0".to_string()),
        items: vec![Item {
            id: ItemId(0),
            parent_id: None,
            turn_id: TurnId(0),
            created_at_ms: 0,
            status: CompletionStatus::Completed,
            error: None,
            payload: ItemPayload::UserMessage {
                text: "which module owns the retry budget?".to_string(),
            },
        }],
        tools: profile(),
        sampling: Sampling::default(),
    }
}

fn read(path: &str) -> (&'static str, Value) {
    ("read_file", json!({ "path": path }))
}

/// What a prediction proposed, as `tool arguments` strings in ranked order.
fn proposed(calls: &[psi_core::model::ToolCallRequest]) -> Vec<String> {
    calls
        .iter()
        .map(|call| format!("{} {}", call.tool, call.arguments))
        .collect()
}

#[tokio::test]
async fn direct_proposal_asks_once_within_its_budget() {
    let server = serve(vec![stream(
        &[read("a.txt"), read("b.txt"), read("a.txt")],
        37,
    )])
    .await;
    let predictor = DirectProposal::new(backend(predictor_config(&server.base_url)));

    let prediction = predictor.predict(&context(), 128).await;

    assert_eq!(server.count(), 1, "direct proposal asks once");
    assert_eq!(server.path(0), "/v1/responses");
    let body = server.body(0);
    // The budget is a cap on generated tokens, and the strategy asks for the
    // most likely calls rather than a sample.
    assert_eq!(body["max_output_tokens"], 128);
    assert_eq!(body["temperature"], 0.0);
    // The same profile the agent is advertised, so the calls are comparable.
    let names: Vec<&str> = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["read_file", "search"]);
    // The shared context, plus the instruction that makes it a proposal.
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(
        input[0]["content"][0]["text"],
        "which module owns the retry budget?"
    );
    let instruction = input[1]["content"][0]["text"].as_str().unwrap();
    assert_eq!(input[1]["role"], "user");
    assert!(
        instruction.contains("Predict what the assistant does next"),
        "{instruction}"
    );

    // The repeat is one guess, not two, and the proposal order is the ranking.
    assert_eq!(
        proposed(&prediction.calls),
        [
            "read_file {\"path\":\"a.txt\"}",
            "read_file {\"path\":\"b.txt\"}"
        ]
    );
    assert_eq!(prediction.usage.output_tokens, 37);
    assert_eq!(prediction.error, None);
}

#[tokio::test]
async fn branch_sampling_ranks_calls_by_agreement() {
    // Three continuations that disagree: a is proposed by all three (twice by
    // the first, which counts once), b by two, c by one.
    let server = serve(vec![
        stream(&[read("a.txt"), read("a.txt"), read("b.txt")], 10),
        stream(&[read("a.txt"), read("c.txt")], 10),
        stream(&[read("b.txt"), read("a.txt")], 10),
    ])
    .await;
    let predictor = BranchSampling::new(backend(predictor_config(&server.base_url)), 3);

    let prediction = predictor.predict(&context(), 120).await;

    assert_eq!(server.count(), 3, "one request per sample");
    for index in 0..3 {
        let body = server.body(index);
        // The budget divides across the samples, so a round of branch sampling
        // generates no more than a round of direct proposal.
        assert_eq!(body["max_output_tokens"], 40);
        assert_eq!(body["temperature"], 1.0);
        // Branch sampling asks nothing: the samples are the predictor
        // continuing the turn itself, so the context is sent untouched.
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
    }

    assert_eq!(
        proposed(&prediction.calls),
        [
            "read_file {\"path\":\"a.txt\"}",
            "read_file {\"path\":\"b.txt\"}",
            "read_file {\"path\":\"c.txt\"}",
        ]
    );
    // Every sample's usage is the round's cost.
    assert_eq!(prediction.usage.output_tokens, 30);
    assert_eq!(prediction.usage.input_tokens, 300);
    assert_eq!(prediction.error, None);
}

#[tokio::test]
async fn the_chat_fallback_sends_one_whole_chat_request() {
    let server = serve(vec![chat_reply(&[read("a.txt"), read("b.txt")])]).await;
    let mut config = predictor_config(&server.base_url);
    config.endpoint = Endpoint::ChatCompletions;
    let predictor = DirectProposal::new(backend(config));

    let prediction = predictor.predict(&context(), 64).await;

    assert_eq!(server.path(0), "/v1/chat/completions");
    let body = server.body(0);
    assert_eq!(body["stream"], false);
    assert_eq!(body["max_completion_tokens"], 64);
    // vLLM truncates a reply to its first tool call when this is explicitly
    // false, which would cap every proposal at one call.
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(
        messages[1]["content"],
        "which module owns the retry budget?"
    );
    assert!(
        messages[2]["content"]
            .as_str()
            .unwrap()
            .contains("Predict what the assistant does next"),
    );

    assert_eq!(
        proposed(&prediction.calls),
        [
            "read_file {\"path\":\"a.txt\"}",
            "read_file {\"path\":\"b.txt\"}"
        ]
    );
    assert_eq!(prediction.usage.input_tokens, 120);
    assert_eq!(prediction.usage.output_tokens, 24);
    assert_eq!(prediction.error, None);
}

/// The zero-tool-call guard latches per target, so a predictor's first empty
/// proposal is reported as the configuration error it might be. The outcome is
/// an empty prediction either way, and one proposed call settles it for good.
#[tokio::test]
async fn an_empty_first_proposal_carries_the_configuration_guard_and_then_stops() {
    let server = serve(vec![
        stream(&[], 5),
        stream(&[read("a.txt")], 9),
        stream(&[], 5),
    ])
    .await;
    let predictor = DirectProposal::new(backend(predictor_config(&server.base_url)));

    let first = predictor.predict(&context(), 32).await;
    assert!(first.calls.is_empty());
    let error = first.error.expect("the guard's message");
    assert!(error.contains("--tool-call-parser"), "{error}");

    let second = predictor.predict(&context(), 32).await;
    assert_eq!(proposed(&second.calls), ["read_file {\"path\":\"a.txt\"}"]);
    assert_eq!(second.error, None);

    // The parser is proven now, so an honestly empty round is just empty.
    let third = predictor.predict(&context(), 32).await;
    assert!(third.calls.is_empty());
    assert_eq!(third.error, None);
}

/// A predictor that never answers costs the run its speculation and nothing
/// else: the turns still complete, the task still succeeds, and the trace says
/// why every round came back empty.
#[tokio::test]
async fn a_failing_predictor_misses_rounds_instead_of_failing_turns() {
    let server = serve(vec![
        "HTTP/1.1 500 Internal Server Error\r\ncontent-length: 5\r\n\r\noops!".to_string(),
    ])
    .await;
    let task = tasks()
        .iter()
        .find(|task| task.name == "read_and_answer")
        .expect("the read-only benchmark task");
    let dir = tempfile::tempdir().unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(20),
        model_delay_ms: 50,
        speculate: Some(Speculation {
            strategy: Strategy::Direct {
                predictor: predictor_config(&server.base_url),
            },
            prediction_budget: 128,
            execution_budget: 2,
        }),
    };

    let path = run_trial(task, 0, &config, dir.path()).await.unwrap();
    let run = RunTrace::read(&path).unwrap();

    assert!(run.success, "the task still succeeds without speculation");
    let stats = SpeculationStats::of(std::slice::from_ref(&run)).expect("records");
    assert_eq!(stats.proposed, 0);
    assert_eq!(stats.executed, 0);
    assert_eq!(stats.hits, 0);
    assert!(stats.misses > 0, "every authoritative call executed itself");
    let reasons: Vec<&str> = run
        .turns
        .iter()
        .flat_map(|turn| &turn.speculation)
        .filter_map(|record| match record {
            TraceRecord::Prediction { error, .. } => error.as_deref(),
            _ => None,
        })
        .collect();
    assert!(!reasons.is_empty(), "the missed rounds record their reason");
    assert!(
        reasons.iter().all(|reason| reason.contains("500")),
        "{reasons:?}"
    );
}

#[tokio::test]
async fn a_strategy_run_reports_hit_rate_cost_waste_and_latency_change() {
    // A predictor that proposes the two calls the task's first turn makes.
    let server = serve(vec![stream(
        &[
            ("search", json!({ "pattern": "RetryBudget" })),
            ("read_file", json!({ "path": "src/budget.rs" })),
        ],
        18,
    )])
    .await;
    let task = tasks()
        .iter()
        .find(|task| task.name == "read_and_answer")
        .expect("the read-only benchmark task");
    let dir = tempfile::tempdir().unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(120),
        model_delay_ms: 300,
        speculate: None,
    };

    let baseline = run_task(task, &config, &dir.path().join("baseline"))
        .await
        .unwrap();
    let mut speculated_config = config.clone();
    speculated_config.speculate = Some(Speculation {
        strategy: Strategy::Direct {
            predictor: predictor_config(&server.base_url),
        },
        prediction_budget: 128,
        execution_budget: 2,
    });
    let speculated = run_task(task, &speculated_config, &dir.path().join("direct"))
        .await
        .unwrap();

    let stats = speculated.speculation.expect("speculation records");
    assert!(stats.hits > 0, "the predictor guessed some calls right");
    assert!(stats.wasted > 0, "and some it did not");
    assert!(
        stats.predictor_tokens.output_tokens > 0,
        "the predictor billed what it spent"
    );
    assert!(speculated.predictor_errors.is_empty());

    let comparison = Comparison {
        baseline,
        speculated,
    };
    let printed = comparison.to_string();
    for expected in ["hits (", "predictor cost:", "wasted", "net latency change:"] {
        assert!(
            printed.contains(expected),
            "{expected} missing from\n{printed}"
        );
    }
}

/// The shell-minimal profile experiment (docs/design.md, "Five tools, one
/// profile"): the same two questions answered through `exec` instead of
/// `search` is work speculation cannot cover, because `exec` is neither
/// allowlisted nor read-only. The replay oracle drives both, so prediction
/// quality is perfect in each and the only difference measured is the profile.
#[tokio::test]
async fn the_shell_minimal_profile_speculates_less_of_the_same_work() {
    let dir = tempfile::tempdir().unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(20),
        model_delay_ms: 50,
        speculate: Some(Speculation {
            strategy: Strategy::Oracle,
            prediction_budget: 0,
            execution_budget: 2,
        }),
    };

    let mut coverage = Vec::new();
    for name in ["read_and_answer", "read_and_answer_shell"] {
        let task = tasks().iter().find(|task| task.name == name).expect(name);
        let report = run_task(task, &config, &dir.path().join(name))
            .await
            .unwrap();
        assert_eq!(report.successes, 1, "{name} answered both questions");
        coverage.push(report.speculation.expect("records"));
    }

    let (structured, shell) = (coverage[0], coverage[1]);
    assert_eq!(
        structured.hit_rate(),
        1.0,
        "perfect prediction, all covered"
    );
    assert!(
        shell.hit_rate() < structured.hit_rate(),
        "shell-minimal covered {:.2}, structured {:.2}",
        shell.hit_rate(),
        structured.hit_rate(),
    );
    assert_eq!(
        shell.hits + shell.misses,
        structured.hits + structured.misses,
        "the same number of calls, differently covered",
    );
}

/// The Milestone 7 comparison against a real server: both strategies under
/// equal budgets, over the default profile and the shell-minimal one. Run with
/// `cargo test -- --ignored` and `PSI_VLLM_BASE_URL` pointing at a vLLM server
/// started with `--enable-auto-tool-choice` and a `--tool-call-parser` for its
/// model; `PSI_VLLM_MODEL` names the served model when an empty one is
/// rejected. The authoritative model stays scripted, so the only thing varying
/// between the runs is the predictor.
#[tokio::test]
#[ignore = "needs PSI_VLLM_BASE_URL and a running vLLM server"]
async fn live_strategies_compare_under_equal_budgets() {
    let base_url = std::env::var("PSI_VLLM_BASE_URL").expect("PSI_VLLM_BASE_URL");
    let mut predictor = VllmConfig {
        base_url,
        ..VllmConfig::default()
    };
    if let Ok(model) = std::env::var("PSI_VLLM_MODEL") {
        predictor.model = model;
    }
    let dir = tempfile::tempdir().unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::measured(),
        model_delay_ms: 2_000,
        speculate: None,
    };

    for task in tasks() {
        let baseline = run_task(task, &config, &dir.path().join("baseline"))
            .await
            .unwrap();
        for (name, strategy) in [
            (
                "direct",
                Strategy::Direct {
                    predictor: predictor.clone(),
                },
            ),
            (
                "branch",
                Strategy::Branch {
                    predictor: predictor.clone(),
                    samples: 4,
                },
            ),
        ] {
            let mut speculated_config = config.clone();
            speculated_config.speculate = Some(Speculation {
                strategy,
                prediction_budget: 256,
                execution_budget: 2,
            });
            let speculated = run_task(task, &speculated_config, &dir.path().join(name))
                .await
                .unwrap();
            assert!(
                speculated.speculation.is_some(),
                "{}/{name} recorded no speculation",
                task.name
            );
            println!(
                "{name}\n{}",
                Comparison {
                    baseline: baseline.clone(),
                    speculated,
                }
            );
        }
    }
}

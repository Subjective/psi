//! The benchmark tasks. A task is a fixture workspace, a scripted model, and
//! a success criterion, written as Rust values rather than a data format: a
//! data format would have to grow a language for the model script and the
//! criterion, and neither is something a benchmark run varies.
//!
//! Both tasks run against the real five-tool profile, so a task's tools really
//! read and really patch the fixture and its success criterion is a claim
//! about the world, not about the script.

use serde_json::json;

use crate::fake::FakeResponse;
use crate::model::{ModelEvent, ToolCallRequest, Usage};

/// One deterministic benchmark task.
pub struct BenchTask {
    pub name: &'static str,
    /// Written into a fresh workspace before every trial, as `(path,
    /// contents)`.
    pub fixture: &'static [(&'static str, &'static str)],
    /// One user message per turn.
    pub prompts: &'static [&'static str],
    /// The model's responses, in order, across every turn of the task.
    pub script: fn() -> Vec<FakeResponse>,
    /// Did the run end in the expected state? Takes the workspace root and
    /// each turn's final assistant message, which are the two forms an answer
    /// takes: what the agent changed and what it said.
    pub success: fn(&std::path::Path, &[String]) -> bool,
}

/// Every task a baseline covers.
pub fn tasks() -> &'static [BenchTask] {
    &TASKS
}

static TASKS: [BenchTask; 2] = [
    BenchTask {
        name: "fix_and_test",
        fixture: FIX_AND_TEST_FIXTURE,
        prompts: &["make the test pass"],
        script: fix_and_test_script,
        success: fix_and_test_success,
    },
    BenchTask {
        name: "read_and_answer",
        fixture: READ_AND_ANSWER_FIXTURE,
        prompts: &[
            "which module owns the retry budget?",
            "how many retries does it allow?",
        ],
        script: read_and_answer_script,
        success: read_and_answer_success,
    },
];

/// A shell library whose test passes only after the fix, so "did the run
/// succeed" is a question about the workspace.
static FIX_AND_TEST_FIXTURE: &[(&str, &str)] = &[
    ("src/lib.sh", "answer() {\n  echo 41\n}\n"),
    (
        "test.sh",
        ". ./src/lib.sh\n[ \"$(answer)\" = \"42\" ] || { echo 'FAIL: expected 42'; exit 1; }\necho 'PASS'\n",
    ),
];

/// The fix landed and the agent said so.
fn fix_and_test_success(workspace: &std::path::Path, answers: &[String]) -> bool {
    let patched = std::fs::read_to_string(workspace.join("src/lib.sh"))
        .is_ok_and(|text| text.contains("echo 42"));
    patched && answers.last().is_some_and(|answer| answer.contains("42"))
}

/// Both questions were answered from the fixture.
fn read_and_answer_success(_workspace: &std::path::Path, answers: &[String]) -> bool {
    answers.len() == 2 && answers[0].contains("budget.rs") && answers[1].contains("3 retries")
}

/// Inspect, edit, and test: the mixed profile, where the two calls that mutate
/// are the two Milestone 6 may not speculate on.
fn fix_and_test_script() -> Vec<FakeResponse> {
    vec![
        FakeResponse::new(vec![
            ModelEvent::ReasoningDelta {
                delta: "Look at the tree first.".into(),
            },
            call("list_directory", "call-1", json!({ "depth": 2 })),
            usage(1_400, 120),
            ModelEvent::Completed,
        ]),
        response(
            call("search", "call-2", json!({ "pattern": "echo 41" })),
            1_600,
        ),
        response(
            call("read_file", "call-3", json!({ "path": "src/lib.sh" })),
            1_800,
        ),
        response(
            call(
                "apply_patch",
                "call-4",
                json!({
                    "path": "src/lib.sh",
                    "old_text": "echo 41",
                    "new_text": "echo 42",
                }),
            ),
            2_000,
        ),
        response(
            call("exec", "call-5", json!({ "command": "sh test.sh" })),
            2_200,
        ),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "The test passes: answer now returns 42.".into(),
            },
            usage(2_400, 90),
            ModelEvent::Completed,
        ]),
    ]
}

static READ_AND_ANSWER_FIXTURE: &[(&str, &str)] = &[
    (
        "src/budget.rs",
        "pub struct RetryBudget {\n    pub max_retries: u32,\n}\n\nimpl RetryBudget {\n    pub fn new() -> Self {\n        Self { max_retries: 3 }\n    }\n}\n",
    ),
    (
        "src/client.rs",
        "use crate::budget::RetryBudget;\n\npub struct Client {\n    budget: RetryBudget,\n}\n",
    ),
    ("README.md", "A tiny client with a retry budget.\n"),
];

/// Two turns of reading and answering: every call is read-only, so this is the
/// task Milestone 6's speculative allowlist covers completely.
fn read_and_answer_script() -> Vec<FakeResponse> {
    vec![
        response(
            call("search", "call-1", json!({ "pattern": "RetryBudget" })),
            1_100,
        ),
        response(
            call("read_file", "call-2", json!({ "path": "src/budget.rs" })),
            1_300,
        ),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "src/budget.rs owns the retry budget.".into(),
            },
            usage(1_500, 80),
            ModelEvent::Completed,
        ]),
        response(
            call("read_file", "call-3", json!({ "path": "src/budget.rs" })),
            1_900,
        ),
        response(
            call("search", "call-4", json!({ "pattern": "max_retries" })),
            2_100,
        ),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "It allows 3 retries.".into(),
            },
            usage(2_300, 70),
            ModelEvent::Completed,
        ]),
    ]
}

/// One response that makes one call and reports what it cost.
fn response(call: ModelEvent, input_tokens: u64) -> FakeResponse {
    FakeResponse::new(vec![call, usage(input_tokens, 60), ModelEvent::Completed])
}

fn call(tool: &str, call_id: &str, arguments: serde_json::Value) -> ModelEvent {
    ModelEvent::ToolCallCompleted {
        call: ToolCallRequest {
            call_id: call_id.into(),
            tool: tool.into(),
            arguments,
        },
    }
}

/// Scripted token counts. They stand in for what a real response would bill,
/// so a baseline has a cost to report next to its latency; they grow across a
/// task the way a lengthening history does.
fn usage(input_tokens: u64, output_tokens: u64) -> ModelEvent {
    ModelEvent::Usage {
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    }
}

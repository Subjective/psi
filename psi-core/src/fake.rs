//! Deterministic fakes: a scripted model and canned tools, so headless tests
//! drive complete turns with no network (Milestone 1).

use std::collections::VecDeque;
use std::sync::Mutex;

use tokio::sync::mpsc;

use crate::model::{ModelBackend, ModelEvent, TurnRequest};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolOutput, ToolSpec};

/// One scripted model response. `hang` keeps the stream open after the events
/// until the engine drops it, so tests can cancel mid-response.
pub struct FakeResponse {
    pub events: Vec<ModelEvent>,
    pub hang: bool,
}

impl FakeResponse {
    pub fn new(events: Vec<ModelEvent>) -> Self {
        Self {
            events,
            hang: false,
        }
    }

    pub fn hanging(events: Vec<ModelEvent>) -> Self {
        Self { events, hang: true }
    }
}

/// Plays back a script, one response per request, in order. An exhausted
/// script answers with a model error.
pub struct FakeModel {
    script: Mutex<VecDeque<FakeResponse>>,
}

impl FakeModel {
    pub fn new(script: impl IntoIterator<Item = FakeResponse>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().collect()),
        }
    }
}

impl ModelBackend for FakeModel {
    fn stream_response(&self, _request: TurnRequest) -> mpsc::Receiver<ModelEvent> {
        let (tx, rx) = mpsc::channel(32);
        let response = self.script.lock().expect("script lock").pop_front();
        tokio::spawn(async move {
            let Some(response) = response else {
                let _ = tx
                    .send(ModelEvent::Error {
                        message: "fake model: script exhausted".to_string(),
                    })
                    .await;
                return;
            };
            for event in response.events {
                if tx.send(event).await.is_err() {
                    return;
                }
            }
            if response.hang {
                // Hold the sender until the receiver is dropped.
                tx.closed().await;
            }
        });
        rx
    }
}

/// A tool with a fixed spec and a canned responder.
pub struct FakeTool {
    spec: ToolSpec,
    effect: ToolEffect,
    respond: Box<dyn Fn(&serde_json::Value) -> ToolOutput + Send + Sync>,
}

impl FakeTool {
    pub fn new(
        name: &str,
        effect: ToolEffect,
        respond: impl Fn(&serde_json::Value) -> ToolOutput + Send + Sync + 'static,
    ) -> Self {
        Self {
            spec: ToolSpec {
                name: name.to_string(),
                description: format!("fake {name}"),
                parameters: serde_json::json!({ "type": "object" }),
            },
            effect,
            respond: Box::new(respond),
        }
    }

    /// A tool that always succeeds with fixed content.
    pub fn canned(name: &str, effect: ToolEffect, content: &str) -> Self {
        let content = content.to_string();
        Self::new(name, effect, move |_| ToolOutput {
            content: content.clone(),
            error: None,
            truncated: false,
        })
    }
}

impl Tool for FakeTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn effect(&self) -> ToolEffect {
        self.effect
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        let output = (self.respond)(&invocation.arguments);
        Box::pin(async move { output })
    }
}

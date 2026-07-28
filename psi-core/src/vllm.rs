//! The vLLM backend: the same Responses codec pointed at a self-hosted vLLM
//! server. vLLM's `/v1/responses` is built for every generate-capable model,
//! not just gpt-oss, and it converts Responses tools into the Chat format and
//! parses replies with the Chat tool parsers, so only transport and
//! capabilities differ from OpenAI (docs/design.md, "Model backends: one
//! Responses codec, explicit capabilities").
//!
//! Cancellation is dropping the returned receiver, which drops the HTTP
//! response and closes the connection. vLLM's `/v1/responses/{id}/cancel`
//! refuses anything but a background response, so there is no endpoint to
//! call and nothing here may assume one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::chat;
use crate::model::{ModelBackend, ModelEvent, TurnRequest};
use crate::openai::{BackendError, DEFAULT_INSTRUCTIONS};
use crate::responses::{Capabilities, Decoder, SseBuffer, build_request};

/// Which endpoint a target is driven through. Responses is the one Psi is built
/// on; Chat Completions exists only as a predictor-side config switch, for
/// model-parser combinations whose Responses streaming misbehaves (docs/
/// design.md, "Model backends"). The authoritative loop needs streaming and
/// never selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Endpoint {
    #[default]
    Responses,
    ChatCompletions,
}

/// Where the backend points and how long it waits.
///
/// The zero-tool-call guard lives here because this is the surface that causes
/// it: vLLM parses tool calls only when the server was started with
/// `--enable-auto-tool-choice` and a `--tool-call-parser` matching the model.
/// Without both, no parser is installed, the tools are still rendered into the
/// prompt, and the model's tool call comes back as ordinary output text — a
/// response with zero tool calls and no error anywhere. One response cannot be
/// told apart from a model that simply chose to answer, so the backend reports
/// a configuration error while the target has never yet completed a tool call,
/// and stops checking once it has. See `VllmBackend`.
///
/// A predictor target meets the same guard, and legitimately proposes nothing
/// sometimes. That is the guard's premise holding rather than misfiring: while
/// a target has never returned a parsed call, an empty reply really is
/// ambiguous, and either reading produces the same prediction — a round with
/// zero proposals, which `crate::predictor` records in the trace along with the
/// guard's message as the round's reason. The predictor's first proposed call
/// clears the flag for good; if the parser really is missing, every round
/// carries the message and the hit rate is zero, which is the diagnosis.
#[derive(Debug, Clone)]
pub struct VllmConfig {
    /// The served model name. Empty means the model the server already
    /// loaded: vLLM accepts a request that names no model, and a server
    /// serves one base model.
    pub model: String,
    pub base_url: String,
    /// The environment variable holding the API key, read by
    /// `VllmBackend::new`. vLLM usually runs keyless, so an unset variable is
    /// not an error: the request simply goes out unauthenticated.
    pub api_key_env: String,
    /// The system prompt, sent as the request's `instructions`. Psi's, not the
    /// provider's, so both backends start from the same one.
    pub instructions: String,
    /// How long the request may take to produce response headers.
    pub request_timeout: Duration,
    /// How long the stream may go quiet before the round fails. Stream silence
    /// is never success. The Chat Completions path has no stream — vLLM sends
    /// its headers only once generation is done — so there it bounds the whole
    /// exchange instead, which is the same thing measured end to end.
    pub idle_timeout: Duration,
    /// The predictor's fallback switch; the default is the Responses endpoint
    /// the rest of Psi is built on.
    pub endpoint: Endpoint,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            base_url: "http://localhost:8000/v1".to_string(),
            api_key_env: "VLLM_API_KEY".to_string(),
            instructions: DEFAULT_INSTRUCTIONS.to_string(),
            request_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(120),
            endpoint: Endpoint::Responses,
        }
    }
}

pub struct VllmBackend {
    config: VllmConfig,
    /// Absent when the server runs keyless.
    api_key: Option<String>,
    http: reqwest::Client,
    /// Whether this target still has to prove it has a tool parser. A missing
    /// parser is a property of the server process, not of one response, so the
    /// first tool call the target ever completes clears the flag for good and
    /// every later silent response is taken at face value.
    tool_parser_unproven: Arc<AtomicBool>,
}

impl VllmBackend {
    /// Reads the API key from the environment if it is set. An unset variable
    /// is the normal case for a self-hosted server.
    pub fn new(config: VllmConfig) -> Result<Self, BackendError> {
        let api_key = std::env::var(&config.api_key_env).ok();
        Ok(Self {
            http: reqwest::Client::builder()
                .connect_timeout(config.request_timeout)
                .build()?,
            config,
            api_key,
            tool_parser_unproven: Arc::new(AtomicBool::new(true)),
        })
    }
}

impl ModelBackend for VllmBackend {
    fn stream_response(&self, request: TurnRequest) -> mpsc::Receiver<ModelEvent> {
        let (events, receiver) = mpsc::channel(64);
        let guard = Guard {
            tools_advertised: !request.tools.is_empty(),
            unproven: self.tool_parser_unproven.clone(),
        };
        let base_url = self.config.base_url.trim_end_matches('/');
        let (url, body) = match self.config.endpoint {
            Endpoint::Responses => (
                format!("{base_url}/responses"),
                build_request(
                    &self.config.model,
                    &self.config.instructions,
                    Capabilities::VLLM,
                    &request,
                ),
            ),
            Endpoint::ChatCompletions => (
                format!("{base_url}/chat/completions"),
                chat::build_request(&self.config.model, &self.config.instructions, &request),
            ),
        };
        // Auth is the transport difference: a keyless server is sent no
        // `authorization` header at all rather than an empty one.
        let mut post = self.http.post(&url).json(&body);
        if self.config.endpoint == Endpoint::Responses {
            post = post.header("accept", "text/event-stream");
        }
        if let Some(api_key) = &self.api_key {
            post = post.bearer_auth(api_key);
        }
        let request_timeout = self.config.request_timeout;
        let idle_timeout = self.config.idle_timeout;
        let endpoint = self.config.endpoint;
        tokio::spawn(async move {
            match endpoint {
                Endpoint::Responses => {
                    stream(post, request_timeout, idle_timeout, guard, events).await
                }
                Endpoint::ChatCompletions => complete(post, idle_timeout, guard, events).await,
            }
        });
        receiver
    }
}

/// The zero-tool-call guard's state for one response.
struct Guard {
    tools_advertised: bool,
    unproven: Arc<AtomicBool>,
}

impl Guard {
    /// Called on every decoded event; returns the event to forward.
    fn observe(&self, event: ModelEvent) -> ModelEvent {
        match event {
            ModelEvent::ToolCallCompleted { .. } => {
                self.unproven.store(false, Ordering::Relaxed);
                event
            }
            ModelEvent::Completed
                if self.tools_advertised && self.unproven.load(Ordering::Relaxed) =>
            {
                ModelEvent::Error {
                    message: "vllm advertised tools and returned none, and this target has \
                              never returned one: start the server with --enable-auto-tool-choice \
                              and a --tool-call-parser matching the model, or tool calls arrive \
                              as plain text. If the model really did just answer, one tool call \
                              settles it for good."
                        .to_string(),
                }
            }
            event => event,
        }
    }
}

async fn stream(
    post: reqwest::RequestBuilder,
    request_timeout: Duration,
    idle_timeout: Duration,
    guard: Guard,
    events: mpsc::Sender<ModelEvent>,
) {
    let response = match timeout(request_timeout, post.send()).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => return fail(&events, format!("vllm request failed: {err}")).await,
        Err(_) => {
            return fail(
                &events,
                format!("vllm did not respond within {request_timeout:?}"),
            )
            .await;
        }
    };

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let body: String = body.chars().take(500).collect();
        return fail(&events, format!("vllm returned {status}: {body}")).await;
    }

    let mut response = response;
    let mut frames = SseBuffer::default();
    let mut decoder = Decoder::default();
    loop {
        let chunk = tokio::select! {
            // The engine drops the receiver to cancel a turn. Returning here
            // drops the response, which closes the connection — the only
            // cancellation vLLM offers for a synchronous stream.
            _ = events.closed() => return,
            chunk = timeout(idle_timeout, response.chunk()) => chunk,
        };
        let chunk = match chunk {
            Ok(Ok(Some(chunk))) => chunk,
            // The body ended without a terminal event. vLLM drops the stream
            // this way when generation fails mid-response; the engine treats a
            // silent end as a failed round.
            Ok(Ok(None)) => return,
            Ok(Err(err)) => return fail(&events, format!("vllm stream failed: {err}")).await,
            Err(_) => {
                return fail(&events, format!("vllm stream idle for {idle_timeout:?}")).await;
            }
        };
        for payload in frames.push(&chunk) {
            let Ok(event) = serde_json::from_str::<serde_json::Value>(&payload) else {
                return fail(&events, format!("vllm sent a non-JSON event: {payload}")).await;
            };
            for event in decoder.decode(&event) {
                let event = guard.observe(event);
                let terminal = matches!(event, ModelEvent::Completed | ModelEvent::Error { .. });
                if events.send(event).await.is_err() || terminal {
                    return;
                }
            }
        }
    }
}

/// Reads one whole Chat Completions reply. Cancellation is still the receiver
/// going away: returning here drops the pending response and closes the
/// connection, exactly as the streaming path does.
async fn complete(
    post: reqwest::RequestBuilder,
    reply_timeout: Duration,
    guard: Guard,
    events: mpsc::Sender<ModelEvent>,
) {
    let exchange = async {
        let response = post.send().await?;
        let status = response.status();
        Ok::<_, reqwest::Error>((status, response.text().await?))
    };
    let reply = tokio::select! {
        _ = events.closed() => return,
        reply = timeout(reply_timeout, exchange) => reply,
    };
    let (status, body) = match reply {
        Ok(Ok(reply)) => reply,
        Ok(Err(err)) => return fail(&events, format!("vllm chat request failed: {err}")).await,
        Err(_) => {
            return fail(
                &events,
                format!("vllm chat did not reply within {reply_timeout:?}"),
            )
            .await;
        }
    };
    if !status.is_success() {
        let body: String = body.chars().take(500).collect();
        return fail(&events, format!("vllm chat returned {status}: {body}")).await;
    }
    let Ok(reply) = serde_json::from_str::<serde_json::Value>(&body) else {
        let body: String = body.chars().take(500).collect();
        return fail(&events, format!("vllm chat sent a non-JSON reply: {body}")).await;
    };
    for event in chat::decode_response(&reply) {
        let event = guard.observe(event);
        let terminal = matches!(event, ModelEvent::Completed | ModelEvent::Error { .. });
        if events.send(event).await.is_err() || terminal {
            return;
        }
    }
}

async fn fail(events: &mpsc::Sender<ModelEvent>, message: String) {
    let _ = events.send(ModelEvent::Error { message }).await;
}

//! The OpenAI backend: one streaming Responses request per model round.
//! Dropping the returned receiver drops the HTTP response, which closes the
//! connection and aborts the request.

use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::model::{ModelBackend, ModelEvent, TurnRequest};
use crate::responses::{Capabilities, Decoder, SseBuffer, build_request};

pub const DEFAULT_INSTRUCTIONS: &str = "\
You are Psi, a coding agent working in a terminal. You have tools for reading, \
searching, and editing files in the user's workspace, and for running shell \
commands in it. Look at the workspace before you answer questions about it, and \
verify your edits by running the project's own checks. Keep replies short and \
concrete.";

/// Where the backend points and how long it waits. A plain struct: Psi has no
/// configuration framework.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub model: String,
    pub base_url: String,
    /// The environment variable holding the API key, read by
    /// `OpenAiBackend::new`.
    pub api_key_env: String,
    /// The system prompt, sent as the request's `instructions`.
    pub instructions: String,
    /// How long the request may take to produce response headers.
    pub request_timeout: Duration,
    /// How long the stream may go quiet before the round fails. Stream silence
    /// is never success.
    pub idle_timeout: Duration,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: "gpt-5.6".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key_env: "OPENAI_API_KEY".to_string(),
            instructions: DEFAULT_INSTRUCTIONS.to_string(),
            request_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("{0} is not set")]
    MissingApiKey(String),
    #[error("http client: {0}")]
    Http(#[from] reqwest::Error),
}

pub struct OpenAiBackend {
    config: OpenAiConfig,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiBackend {
    pub fn new(config: OpenAiConfig) -> Result<Self, BackendError> {
        let api_key = std::env::var(&config.api_key_env)
            .map_err(|_| BackendError::MissingApiKey(config.api_key_env.clone()))?;
        Self::with_api_key(config, api_key)
    }

    /// Takes the key directly, for callers that do not hold it in the
    /// environment — the codec tests point a backend at a local server.
    pub fn with_api_key(config: OpenAiConfig, api_key: String) -> Result<Self, BackendError> {
        Ok(Self {
            http: reqwest::Client::builder()
                .connect_timeout(config.request_timeout)
                .build()?,
            config,
            api_key,
        })
    }
}

impl ModelBackend for OpenAiBackend {
    fn stream_response(&self, request: TurnRequest) -> mpsc::Receiver<ModelEvent> {
        let (events, receiver) = mpsc::channel(64);
        let body = build_request(
            &self.config.model,
            &self.config.instructions,
            Capabilities::OPENAI,
            &request,
        );
        let url = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let request_timeout = self.config.request_timeout;
        let idle_timeout = self.config.idle_timeout;
        tokio::spawn(async move {
            stream(
                http,
                url,
                api_key,
                body,
                request_timeout,
                idle_timeout,
                events,
            )
            .await;
        });
        receiver
    }
}

async fn stream(
    http: reqwest::Client,
    url: String,
    api_key: String,
    body: serde_json::Value,
    request_timeout: Duration,
    idle_timeout: Duration,
    events: mpsc::Sender<ModelEvent>,
) {
    let send = http
        .post(&url)
        .bearer_auth(&api_key)
        .header("accept", "text/event-stream")
        .json(&body)
        .send();
    let response = match timeout(request_timeout, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => return fail(&events, format!("openai request failed: {err}")).await,
        Err(_) => {
            return fail(
                &events,
                format!("openai did not respond within {request_timeout:?}"),
            )
            .await;
        }
    };

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let body: String = body.chars().take(500).collect();
        return fail(&events, format!("openai returned {status}: {body}")).await;
    }

    let mut response = response;
    let mut frames = SseBuffer::default();
    let mut decoder = Decoder::default();
    loop {
        let chunk = tokio::select! {
            // The engine drops the receiver to cancel a turn. Returning here
            // drops the response, which closes the connection.
            _ = events.closed() => return,
            chunk = timeout(idle_timeout, response.chunk()) => chunk,
        };
        let chunk = match chunk {
            Ok(Ok(Some(chunk))) => chunk,
            // The body ended without a terminal event. The engine treats a
            // silent end as a failed round.
            Ok(Ok(None)) => return,
            Ok(Err(err)) => return fail(&events, format!("openai stream failed: {err}")).await,
            Err(_) => {
                return fail(&events, format!("openai stream idle for {idle_timeout:?}")).await;
            }
        };
        for payload in frames.push(&chunk) {
            let Ok(event) = serde_json::from_str::<serde_json::Value>(&payload) else {
                return fail(&events, format!("openai sent a non-JSON event: {payload}")).await;
            };
            for event in decoder.decode(&event) {
                let terminal = matches!(event, ModelEvent::Completed | ModelEvent::Error { .. });
                if events.send(event).await.is_err() || terminal {
                    return;
                }
            }
        }
    }
}

async fn fail(events: &mpsc::Sender<ModelEvent>, message: String) {
    let _ = events.send(ModelEvent::Error { message }).await;
}

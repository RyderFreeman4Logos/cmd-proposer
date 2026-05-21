//! Async HTTP client for OpenAI-compatible chat completion endpoints.

use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tracing::{debug, trace, warn};

use crate::error::{LlmError, Result};
use crate::stream::{parse_event, ParseOutcome, StreamAssembler, StreamChunk};
use crate::types::{ChatCompletionResponse, ChatRequest, ChatResponse};

/// Async client for an OpenAI-compatible chat completion endpoint.
///
/// One client serves many requests across the lifetime of a session; clone
/// is cheap (shares the underlying `reqwest::Client` connection pool).
#[derive(Debug, Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    endpoint: String,
    request_timeout: Duration,
}

/// Construction parameters for [`LlmClient::new`].
///
/// Mirrors the subset of [`cps_config::ModelConfig`] + [`cps_config::RuntimeConfig`]
/// that this crate needs, kept as a separate type so the client does not pull
/// in the full `Config` struct at construction time.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Endpoint base URL, e.g. `http://localhost:8317/v1`.
    pub base_url: String,
    /// Bearer token; MAY be empty for no-auth local endpoints (SPEC §12.2).
    pub api_key: String,
    /// Per-request timeout (`runtime.request_timeout_ms`).
    pub request_timeout_ms: u64,
}

impl ClientConfig {
    /// Build a [`ClientConfig`] from the project's [`cps_config::Config`].
    pub fn from_config(cfg: &cps_config::Config) -> Self {
        Self {
            base_url: cfg.model.base_url.clone(),
            api_key: cfg.model.api_key.clone(),
            request_timeout_ms: cfg.runtime.request_timeout_ms,
        }
    }
}

impl LlmClient {
    /// Construct a new client.
    ///
    /// Returns [`LlmError::InvalidConfig`] if `base_url` is empty or the
    /// auth header cannot be built from the given API key.
    pub fn new(cfg: &ClientConfig) -> Result<Self> {
        if cfg.base_url.is_empty() {
            return Err(LlmError::InvalidConfig("base_url is empty".into()));
        }

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !cfg.api_key.is_empty() {
            let value = HeaderValue::from_str(&format!("Bearer {}", cfg.api_key))
                .map_err(|e| LlmError::InvalidConfig(format!("invalid api_key: {e}")))?;
            headers.insert(AUTHORIZATION, value);
        }

        let http = reqwest::Client::builder()
            .default_headers(headers)
            // We enforce the per-request timeout per-call (see `complete`/
            // `complete_streaming`) so a single long stream cannot exceed
            // it. Setting it on the builder too keeps connection-level
            // hangs bounded.
            .timeout(Duration::from_millis(cfg.request_timeout_ms))
            .build()
            .map_err(|e| LlmError::InvalidConfig(format!("reqwest builder: {e}")))?;

        Ok(Self {
            http,
            endpoint: format!("{}/chat/completions", cfg.base_url.trim_end_matches('/')),
            request_timeout: Duration::from_millis(cfg.request_timeout_ms),
        })
    }

    /// Non-streaming chat completion.
    ///
    /// The `stream` flag in `request` is forced to `false`; callers that
    /// want streaming MUST use [`Self::complete_streaming`].
    pub async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut body = request.clone();
        body.stream = false;

        let response = self
            .http
            .post(&self.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_send_err(e, self.request_timeout))?;

        let status = response.status();
        if !status.is_success() {
            return Err(error_from_response(response).await);
        }

        let parsed: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LlmError::Malformed(format!("response body: {e}")))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::Malformed("response has no choices".into()))?;

        Ok(ChatResponse {
            message: choice.message,
            usage: parsed.usage,
            finish_reason: choice.finish_reason,
        })
    }

    /// Streaming chat completion.
    ///
    /// Invokes `callback` with each parsed [`StreamChunk`] as it arrives,
    /// then returns the assembled final [`ChatResponse`] once the stream
    /// closes (whether by `[DONE]` sentinel or by the underlying body
    /// ending). The total time budget is bounded by `request_timeout_ms`.
    pub async fn complete_streaming<F>(
        &self,
        request: &ChatRequest,
        mut callback: F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(StreamChunk),
    {
        let mut body = request.clone();
        body.stream = true;

        let send = self.http.post(&self.endpoint).json(&body).send();

        let response = match tokio::time::timeout(self.request_timeout, send).await {
            Err(_) => {
                return Err(LlmError::Timeout {
                    timeout_ms: self.request_timeout.as_millis() as u64,
                })
            }
            Ok(r) => r.map_err(|e| map_send_err(e, self.request_timeout))?,
        };

        let status = response.status();
        if !status.is_success() {
            return Err(error_from_response(response).await);
        }

        let mut byte_stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut assembler = StreamAssembler::default();
        let deadline = tokio::time::Instant::now() + self.request_timeout;
        let mut saw_done = false;

        loop {
            let next = tokio::time::timeout_at(deadline, byte_stream.next()).await;
            let chunk = match next {
                Err(_) => {
                    return Err(LlmError::Timeout {
                        timeout_ms: self.request_timeout.as_millis() as u64,
                    });
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => return Err(LlmError::Connection(e.to_string())),
                Ok(Some(Ok(bytes))) => bytes,
            };

            buffer.extend_from_slice(&chunk);

            while let Some((idx, delimiter_len)) = find_event_boundary(&buffer) {
                let event = std::str::from_utf8(&buffer[..idx])
                    .map_err(|e| LlmError::Malformed(format!("non-utf8 stream event: {e}")))?
                    .to_owned();
                buffer.drain(..idx + delimiter_len);

                if process_event_block(&event, &mut assembler, &mut callback)? {
                    saw_done = true;
                    break;
                }
            }

            if saw_done {
                break;
            }
        }

        // Flush any trailing event the server didn't terminate with the
        // double-newline (some lightweight servers close the connection
        // immediately after the final payload).
        if !saw_done && !is_sse_whitespace(&buffer) {
            let trailing = std::str::from_utf8(&buffer)
                .map_err(|e| LlmError::Malformed(format!("non-utf8 stream event: {e}")))?
                .to_owned();
            process_event_block(&trailing, &mut assembler, &mut callback)?;
        }

        let (message, finish_reason, usage) = assembler.finalize();
        Ok(ChatResponse {
            message,
            usage,
            finish_reason,
        })
    }
}

/// Locate the end of the next SSE event in `buf`.
///
/// SSE event delimiter is a blank line (`\n\n` or `\r\n\r\n`). Returns the
/// byte index of the first delimiter byte plus delimiter length, or `None`
/// if no complete event is buffered yet.
fn find_event_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    // Prefer the earliest of the two possible delimiters.
    let a = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|idx| (idx, 2));
    let b = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|idx| (idx, 4));
    match (a, b) {
        (Some(x), Some(y)) => Some(if x.0 <= y.0 { x } else { y }),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn is_sse_whitespace(buf: &[u8]) -> bool {
    buf.iter()
        .all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
}

/// Parse one complete SSE event block (which may contain multiple `data:`
/// lines per the spec, though OpenAI uses one) and feed it to the assembler.
fn process_event_block<F>(
    block: &str,
    assembler: &mut StreamAssembler,
    callback: &mut F,
) -> Result<bool>
where
    F: FnMut(StreamChunk),
{
    for line in block.lines() {
        let line = line.trim_end_matches('\r');
        let payload = match line.strip_prefix("data:") {
            Some(p) => p.trim_start(),
            None => continue, // ignore `event:`, `id:`, comments, etc.
        };

        match parse_event(payload) {
            Ok(ParseOutcome::Skip) => continue,
            Ok(ParseOutcome::Done) => return Ok(true),
            Ok(ParseOutcome::Data(chunk, usage)) => {
                trace!(?chunk, "sse chunk");
                assembler.ingest(&chunk)?;
                assembler.set_usage(usage);
                callback(chunk);
            }
            Err(e) => {
                warn!(error = %e, payload = %payload, "failed to parse SSE chunk");
                return Err(LlmError::Malformed(format!("sse chunk parse: {e}")));
            }
        }
    }
    Ok(false)
}

fn map_send_err(e: reqwest::Error, request_timeout: Duration) -> LlmError {
    if e.is_timeout() {
        debug!("reqwest timeout");
        LlmError::Timeout {
            timeout_ms: request_timeout.as_millis() as u64,
        }
    } else {
        LlmError::Connection(e.to_string())
    }
}

async fn error_from_response(response: reqwest::Response) -> LlmError {
    let status = response.status().as_u16();
    let retry_after_secs = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let body = response.text().await.unwrap_or_default();
    LlmError::from_status(status, body, retry_after_secs)
}

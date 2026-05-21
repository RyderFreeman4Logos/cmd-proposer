//! Error types for [`crate::LlmClient`].

/// Failure modes for chat completion calls.
///
/// `Timeout`, `RateLimit`, and `Server` are transient (the caller may retry
/// with backoff). `BadRequest`, `Malformed`, and `Unauthorized` are NOT
/// — they indicate a bug in the request or stale credentials.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("request timed out after {timeout_ms} ms")]
    Timeout { timeout_ms: u64 },

    #[error("connection failure: {0}")]
    Connection(String),

    #[error("authentication failed (401)")]
    Unauthorized,

    #[error("rate limited (429){retry_after}",
        retry_after = .retry_after_secs.map(|s| format!(" — retry after {s}s")).unwrap_or_default())]
    RateLimit { retry_after_secs: Option<u64> },

    #[error("server error ({status}): {body}")]
    Server { status: u16, body: String },

    #[error("bad request ({status}): {body}")]
    BadRequest { status: u16, body: String },

    #[error("malformed response: {0}")]
    Malformed(String),

    #[error("invalid client config: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, LlmError>;

impl LlmError {
    /// True if a retry with backoff stands a reasonable chance of succeeding.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            LlmError::Timeout { .. }
                | LlmError::Connection(_)
                | LlmError::RateLimit { .. }
                | LlmError::Server { .. }
        )
    }

    pub(crate) fn from_status(status: u16, body: String, retry_after_secs: Option<u64>) -> Self {
        match status {
            401 | 403 => LlmError::Unauthorized,
            429 => LlmError::RateLimit { retry_after_secs },
            500..=599 => LlmError::Server { status, body },
            _ => LlmError::BadRequest { status, body },
        }
    }
}

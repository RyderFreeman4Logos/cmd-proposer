//! HuggingFace `tokenizer.json` integration for cmd-proposer token counting.
//!
//! Provides a [`Tokenizer`] trait used by the budget engine and evidence
//! compressor to count tokens deterministically. The primary implementation
//! [`HuggingFaceTokenizer`] loads a `tokenizer.json` file produced by the
//! HuggingFace `tokenizers` library; when the file is unavailable, the
//! factory [`create_tokenizer`] degrades to [`FallbackTokenizer`] (a 4-chars
//! per-token heuristic) instead of failing — `cps` must still boot when the
//! user has not yet downloaded the model assets.

use std::path::{Path, PathBuf};

use cps_config::TokenizerConfig;
use tokenizers::Tokenizer as HfTokenizer;

/// Counts tokens for a given UTF-8 string.
///
/// Implementations MUST be cheap to clone-by-reference (`&self`) and safe to
/// share across threads — the agent uses one shared tokenizer across the main
/// loop and every subagent worker.
pub trait Tokenizer: Send + Sync {
    /// Returns the number of tokens the model would see for `text`.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Errors that may occur when constructing a [`HuggingFaceTokenizer`].
///
/// These are surfaced only from [`HuggingFaceTokenizer::from_path`]; the
/// public [`create_tokenizer`] factory swallows them and falls back.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer file not found: {0}")]
    NotFound(PathBuf),

    #[error("failed to load tokenizer.json at {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
}

/// HuggingFace `tokenizer.json`-backed implementation.
#[derive(Debug)]
pub struct HuggingFaceTokenizer {
    inner: HfTokenizer,
}

impl HuggingFaceTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file on disk.
    ///
    /// Accepts a leading `~/` and expands it to the user's home directory.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let resolved = expand_tilde(path.as_ref());
        if !resolved.exists() {
            return Err(TokenizerError::NotFound(resolved));
        }
        let inner = HfTokenizer::from_file(&resolved).map_err(|e| TokenizerError::Load {
            path: resolved.clone(),
            // tokenizers returns a Box<dyn Error + Send + Sync>; wrap into anyhow
            // so callers get a stable error type without leaking the dep.
            source: anyhow::anyhow!(e.to_string()),
        })?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // `add_special_tokens=false`: we count raw content, not chat-template
        // wrappers. Budget math elsewhere already accounts for prompt overhead.
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().len(),
            Err(e) => {
                // Encoding shouldn't fail for valid UTF-8 input, but if it
                // does, fall back to char heuristic for this single call —
                // crashing the agent over a token count is unacceptable.
                tracing::warn!(
                    error = %e,
                    "huggingface tokenizer encode failed; using char heuristic for this input"
                );
                fallback_estimate(text)
            }
        }
    }
}

/// Char-count / 4 heuristic for environments without a real tokenizer file.
///
/// The factor 4 matches OpenAI's published "≈4 chars per token for English"
/// guideline. It is intentionally coarse: any caller that needs accuracy
/// must provide a real `tokenizer.json`.
pub struct FallbackTokenizer;

impl FallbackTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FallbackTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for FallbackTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        fallback_estimate(text)
    }
}

fn fallback_estimate(text: &str) -> usize {
    // Use byte length: matches the "≈4 chars per token" rule for ASCII and
    // gives a more conservative (higher) count for multi-byte UTF-8, which
    // is the safer direction for budget enforcement.
    text.len() / 4
}

/// Build the tokenizer requested by `cfg`, degrading to [`FallbackTokenizer`]
/// on any load failure.
///
/// The current return type is `anyhow::Result` for forward-compatibility with
/// configs that opt out of the fallback in future versions; today the
/// function never returns `Err`.
pub fn create_tokenizer(cfg: &TokenizerConfig) -> anyhow::Result<Box<dyn Tokenizer>> {
    match cfg.tokenizer_type.as_str() {
        "huggingface_tokenizer_json" => match HuggingFaceTokenizer::from_path(&cfg.path) {
            Ok(t) => Ok(Box::new(t)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %cfg.path,
                    "failed to load tokenizer.json; using char-heuristic fallback (token counts will be approximate)"
                );
                Ok(Box::new(FallbackTokenizer::new()))
            }
        },
        other => {
            tracing::warn!(
                tokenizer_type = %other,
                "unknown tokenizer.type; using char-heuristic fallback"
            );
            Ok(Box::new(FallbackTokenizer::new()))
        }
    }
}

fn expand_tilde(path: &Path) -> PathBuf {
    let s = match path.to_str() {
        Some(s) => s,
        None => return path.to_path_buf(),
    };
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(path: &str) -> TokenizerConfig {
        TokenizerConfig {
            path: path.to_string(),
            tokenizer_type: "huggingface_tokenizer_json".into(),
        }
    }

    #[test]
    fn fallback_empty_string_is_zero() {
        let t = FallbackTokenizer::new();
        assert_eq!(t.count_tokens(""), 0);
    }

    #[test]
    fn fallback_estimates_quarter_of_byte_length() {
        let t = FallbackTokenizer::new();
        assert_eq!(t.count_tokens("abcd"), 1);
        assert_eq!(t.count_tokens("abcdefgh"), 2);
        // 40 ASCII chars -> 10 tokens
        assert_eq!(t.count_tokens(&"a".repeat(40)), 10);
    }

    #[test]
    fn fallback_under_four_chars_rounds_to_zero() {
        let t = FallbackTokenizer::new();
        assert_eq!(t.count_tokens("a"), 0);
        assert_eq!(t.count_tokens("ab"), 0);
        assert_eq!(t.count_tokens("abc"), 0);
    }

    #[test]
    fn create_tokenizer_with_nonexistent_path_returns_fallback() {
        let tok = create_tokenizer(&cfg("/definitely/not/a/real/path/tokenizer.json"))
            .expect("factory must not error on missing file");
        // Fallback estimate should match the heuristic; a real HF tokenizer
        // would produce a very different (usually smaller) count for this
        // sample, so equality here proves we got the fallback.
        let sample = "the quick brown fox jumps over the lazy dog";
        assert_eq!(tok.count_tokens(sample), sample.len() / 4);
        assert_eq!(tok.count_tokens(""), 0);
    }

    #[test]
    fn create_tokenizer_unknown_type_returns_fallback() {
        let mut c = cfg("/whatever");
        c.tokenizer_type = "tiktoken_bpe".into();
        let tok = create_tokenizer(&c).expect("factory must not error");
        let sample = "hello world";
        assert_eq!(tok.count_tokens(sample), sample.len() / 4);
    }

    #[test]
    fn huggingface_from_path_reports_not_found() {
        let err = HuggingFaceTokenizer::from_path("/no/such/tokenizer.json")
            .expect_err("missing file must error");
        assert!(matches!(err, TokenizerError::NotFound(_)));
    }

    #[test]
    fn huggingface_from_path_rejects_invalid_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bogus = dir.path().join("tokenizer.json");
        std::fs::write(&bogus, b"this is not a tokenizer").expect("write");
        let err = HuggingFaceTokenizer::from_path(&bogus).expect_err("invalid file must error");
        assert!(matches!(err, TokenizerError::Load { .. }));
    }

    /// If a real `tokenizer.json` happens to be installed at the default
    /// location, exercise it. Otherwise skipped — CI without model assets
    /// must still pass.
    #[test]
    fn huggingface_real_tokenizer_if_available() {
        let Some(home) = dirs::home_dir() else { return };
        let candidates = [
            home.join(".local/share/models/qwen/tokenizer.json"),
            home.join(".cache/huggingface/tokenizer.json"),
        ];
        let Some(path) = candidates.iter().find(|p| p.exists()) else {
            return;
        };
        let Ok(tok) = HuggingFaceTokenizer::from_path(path) else {
            return;
        };
        let sample = "the quick brown fox jumps over the lazy dog";
        let n = tok.count_tokens(sample);
        assert!(n > 0, "non-empty input must yield non-zero tokens");
        assert!(
            n < sample.len(),
            "tokenizer must compress below byte length; got {n} for {} bytes",
            sample.len()
        );
        assert_eq!(tok.count_tokens(""), 0);
    }
}

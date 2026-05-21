//! `${VAR}` env var interpolation for config text.
//!
//! Substitution is performed on the raw YAML source string before parsing so
//! that values inside the YAML — including the keys of `${...}` references —
//! are resolved consistently regardless of where they appear (`base_url`,
//! `api_key`, list elements, etc.).
//!
//! Syntax:
//! - `${NAME}` is replaced with the value of environment variable `NAME`.
//! - `${NAME:-default}` is replaced with `NAME`, or `default` if `NAME` is
//!   unset or empty.
//! - A literal `$` can be written as `$$`.
//! - Unset variables without a default produce [`InterpolateError::Unset`].

use regex::{Captures, Regex};
use std::sync::OnceLock;

#[derive(Debug, thiserror::Error)]
pub enum InterpolateError {
    #[error("environment variable `{0}` is not set and has no default")]
    Unset(String),

    #[error("malformed reference `{0}` — expected `${{NAME}}` or `${{NAME:-default}}`")]
    Malformed(String),
}

fn pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Two alternates: literal `$$` (escape) OR a `${...}` reference.
    RE.get_or_init(|| Regex::new(r"\$\$|\$\{([^}]*)\}").unwrap())
}

/// Replace `${VAR}` and `${VAR:-default}` in `text` using `lookup`.
///
/// `lookup` is injected for testability; production code passes a closure
/// around `std::env::var`.
pub fn interpolate_with<F>(text: &str, lookup: F) -> Result<String, InterpolateError>
where
    F: Fn(&str) -> Option<String>,
{
    let mut error: Option<InterpolateError> = None;
    let out = pattern().replace_all(text, |caps: &Captures<'_>| {
        // `$$` → literal `$`
        let whole = &caps[0];
        if whole == "$$" {
            return "$".to_string();
        }
        let inner = match caps.get(1) {
            Some(m) => m.as_str(),
            None => {
                error.get_or_insert_with(|| InterpolateError::Malformed(whole.to_string()));
                return String::new();
            }
        };
        let (name, default) = match inner.split_once(":-") {
            Some((n, d)) => (n.trim(), Some(d)),
            None => (inner.trim(), None),
        };
        if name.is_empty() || name.contains(char::is_whitespace) {
            error.get_or_insert_with(|| InterpolateError::Malformed(whole.to_string()));
            return String::new();
        }
        match lookup(name) {
            Some(v) if !v.is_empty() || default.is_none() => v,
            _ => match default {
                Some(d) => d.to_string(),
                None => {
                    error.get_or_insert_with(|| InterpolateError::Unset(name.to_string()));
                    String::new()
                }
            },
        }
    });
    if let Some(e) = error {
        return Err(e);
    }
    let result = out.into_owned();
    if let Some(pos) = result.find("${") {
        let end = result[pos..]
            .find('}')
            .map(|i| pos + i + 1)
            .unwrap_or(result.len().min(pos + 30));
        return Err(InterpolateError::Malformed(result[pos..end].to_string()));
    }
    Ok(result)
}

/// Replace `${VAR}` and `${VAR:-default}` in `text` using `std::env::var`.
pub fn interpolate_env(text: &str) -> Result<String, InterpolateError> {
    interpolate_with(text, |k| std::env::var(k).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup<'a>(map: &'a HashMap<&'a str, &'a str>) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| map.get(k).map(|s| (*s).to_string())
    }

    #[test]
    fn substitutes_single_var() {
        let map = HashMap::from([("LOCAL_ROUTER_BASEURL", "http://localhost:8317/v1")]);
        let out = interpolate_with("url: ${LOCAL_ROUTER_BASEURL}", lookup(&map)).unwrap();
        assert_eq!(out, "url: http://localhost:8317/v1");
    }

    #[test]
    fn substitutes_multiple_vars_in_yaml_body() {
        let map = HashMap::from([
            ("LOCAL_ROUTER_BASEURL", "http://x/v1"),
            ("LOCAL_ROUTER_API_KEY", "sk-abc"),
            ("LLM_MODEL", "qwen"),
        ]);
        let text = "model:\n  base_url: ${LOCAL_ROUTER_BASEURL}\n  api_key: ${LOCAL_ROUTER_API_KEY}\n  model_name: ${LLM_MODEL}\n";
        let out = interpolate_with(text, lookup(&map)).unwrap();
        assert!(out.contains("base_url: http://x/v1"));
        assert!(out.contains("api_key: sk-abc"));
        assert!(out.contains("model_name: qwen"));
    }

    #[test]
    fn unset_without_default_errors() {
        let map: HashMap<&str, &str> = HashMap::new();
        let err = interpolate_with("${MISSING}", lookup(&map)).unwrap_err();
        match err {
            InterpolateError::Unset(k) => assert_eq!(k, "MISSING"),
            other => panic!("expected Unset, got {other:?}"),
        }
    }

    #[test]
    fn unset_with_default_uses_default() {
        let map: HashMap<&str, &str> = HashMap::new();
        let out = interpolate_with("${MISSING:-fallback}", lookup(&map)).unwrap();
        assert_eq!(out, "fallback");
    }

    #[test]
    fn empty_value_uses_default() {
        let map = HashMap::from([("X", "")]);
        let out = interpolate_with("${X:-fallback}", lookup(&map)).unwrap();
        assert_eq!(out, "fallback");
    }

    #[test]
    fn dollar_escape_yields_literal_dollar() {
        let map: HashMap<&str, &str> = HashMap::new();
        let out = interpolate_with("price: $$5", lookup(&map)).unwrap();
        assert_eq!(out, "price: $5");
    }

    #[test]
    fn malformed_reference_errors() {
        let map: HashMap<&str, &str> = HashMap::new();
        let err = interpolate_with("${ }", lookup(&map)).unwrap_err();
        assert!(matches!(err, InterpolateError::Malformed(_)));
    }

    #[test]
    fn unrelated_dollar_in_text_passes_through() {
        let map: HashMap<&str, &str> = HashMap::new();
        // bare `$VAR` (no braces) is NOT a reference; passes through verbatim.
        let out = interpolate_with("see $HOME for details", lookup(&map)).unwrap();
        assert_eq!(out, "see $HOME for details");
    }
}

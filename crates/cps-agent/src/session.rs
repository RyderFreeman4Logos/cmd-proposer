//! Session initialization: the immutable prefix sent to the LLM once per session.
//!
//! Per SPEC §3.6, the system prompt and tool definitions form the KV-cache
//! prefix. They are computed at session start from `Config` and never touched
//! again. The returned `SessionInit` is the canonical input to the LLM client.

use cps_config::Config;
use serde_json::Value;

use crate::system_prompt;
use crate::tools::{self, ToolFeatureFlags};

/// Frozen LLM-facing session prefix.
///
/// Construct once at session start; pass `system_prompt` as the first message
/// and `tool_defs` as the request's `tools` field. Both are stable for the
/// lifetime of the session.
#[derive(Debug, Clone)]
pub struct SessionInit {
    pub system_prompt: String,
    pub tool_defs: Vec<Value>,
}

impl SessionInit {
    /// Build the session prefix from a validated `Config`.
    ///
    /// `web_search` is included iff `config.search.default_enabled` is true.
    /// `spawn` is included iff `config.subagents.enabled` is true. The
    /// `output_language` for the system prompt comes from
    /// `config.proposal.output_language`.
    pub fn from_config(config: &Config) -> Self {
        let flags = ToolFeatureFlags {
            search_enabled: config.search.default_enabled,
            subagents_enabled: config.subagents.enabled,
        };
        Self {
            system_prompt: system_prompt::build(&config.proposal.output_language),
            tool_defs: tools::build(flags),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> Config {
        let yaml = r#"
model:
  base_url: "http://localhost:8317/v1"
  api_key: "sk-test"
  model_name: "qwen3.6-27b"
  tokenizer:
    path: "/tmp/tokenizer.json"

doc_runner:
  allow_programs: [kubectl, helm]
"#;
        serde_yaml::from_str(yaml).expect("parse minimal config")
    }

    #[test]
    fn defaults_include_web_search_and_spawn() {
        let cfg = minimal_config();
        let init = SessionInit::from_config(&cfg);
        let names: Vec<&str> = init.tool_defs.iter().map(tools::tool_name).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"spawn"));
        assert!(init.system_prompt.contains("zh-CN"));
    }

    #[test]
    fn disabling_search_drops_web_search_only() {
        let mut cfg = minimal_config();
        cfg.search.default_enabled = false;
        let init = SessionInit::from_config(&cfg);
        let names: Vec<&str> = init.tool_defs.iter().map(tools::tool_name).collect();
        assert!(!names.contains(&"web_search"));
        assert!(names.contains(&"spawn"));
    }

    #[test]
    fn disabling_subagents_drops_spawn_only() {
        let mut cfg = minimal_config();
        cfg.subagents.enabled = false;
        let init = SessionInit::from_config(&cfg);
        let names: Vec<&str> = init.tool_defs.iter().map(tools::tool_name).collect();
        assert!(!names.contains(&"spawn"));
        assert!(names.contains(&"web_search"));
    }

    #[test]
    fn output_language_threads_through() {
        let mut cfg = minimal_config();
        cfg.proposal.output_language = "en-US".into();
        let init = SessionInit::from_config(&cfg);
        assert!(init.system_prompt.contains("en-US"));
        assert!(!init.system_prompt.contains("zh-CN"));
    }

    #[test]
    fn tool_defs_remain_sorted() {
        let init = SessionInit::from_config(&minimal_config());
        let names: Vec<&str> = init.tool_defs.iter().map(tools::tool_name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
}

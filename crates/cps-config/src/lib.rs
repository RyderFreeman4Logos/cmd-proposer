//! Config parsing, defaults, validation, and env var interpolation for cmd-proposer.
//!
//! See `drafts/SPEC.md` §12 for the schema.
//!
//! Design principle: only the model endpoint, tokenizer path, and the
//! `doc_runner.allow_programs` whitelist are required. Everything else has
//! sensible defaults supplied by `#[serde(default)]` so users write the
//! shortest possible YAML.

mod init;
mod interpolate;
mod load;

pub use init::{full_template, minimal_template};
pub use interpolate::{interpolate_env, InterpolateError};
pub use load::{load, load_from_path, ConfigSource, LoadError};

use serde::{Deserialize, Serialize};

/// Top-level config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Config {
    #[serde(default = "default_version")]
    pub version: u32,

    pub model: ModelConfig,

    #[serde(default)]
    pub thinking: ThinkingConfig,

    #[serde(default)]
    pub runtime: RuntimeConfig,

    #[serde(default)]
    pub search: SearchConfig,

    pub doc_runner: DocRunnerConfig,

    #[serde(default)]
    pub doc_store: DocStoreConfig,

    #[serde(default)]
    pub subagents: SubagentsConfig,

    #[serde(default)]
    pub proposal: ProposalConfig,

    #[serde(default)]
    pub approval: ApprovalConfig,

    #[serde(default)]
    pub execution_runner: ExecutionRunnerConfig,

    #[serde(default)]
    pub risk: RiskConfig,
}

fn default_version() -> u32 {
    1
}

// ---------- model ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub base_url: String,
    pub api_key: String,
    pub model_name: String,

    #[serde(default = "default_provider")]
    pub provider: String,

    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,

    pub tokenizer: TokenizerConfig,
}

fn default_provider() -> String {
    "local_openai_compatible".into()
}
fn default_max_context_tokens() -> u32 {
    200_000
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub path: String,

    #[serde(default = "default_tokenizer_type", rename = "type")]
    pub tokenizer_type: String,
}

fn default_tokenizer_type() -> String {
    "huggingface_tokenizer_json".into()
}

// ---------- thinking ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    #[serde(default = "default_main_agent_thinking")]
    pub main_agent: u32,

    #[serde(default = "default_subagent_thinking")]
    pub subagent_default: u32,
}

fn default_main_agent_thinking() -> u32 {
    32_768
}
fn default_subagent_thinking() -> u32 {
    4_096
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            main_agent: default_main_agent_thinking(),
            subagent_default: default_subagent_thinking(),
        }
    }
}

// ---------- runtime ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    #[serde(default = "default_global_timeout_ms")]
    pub global_timeout_ms: u64,

    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,

    #[serde(default = "default_ui_progress_interval_ms")]
    pub ui_progress_interval_ms: u64,
}

fn default_global_timeout_ms() -> u64 {
    1_200_000
}
fn default_request_timeout_ms() -> u64 {
    1_200_000
}
fn default_ui_progress_interval_ms() -> u64 {
    5_000
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            global_timeout_ms: default_global_timeout_ms(),
            request_timeout_ms: default_request_timeout_ms(),
            ui_progress_interval_ms: default_ui_progress_interval_ms(),
        }
    }
}

// ---------- search ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchConfig {
    #[serde(default = "default_search_enabled")]
    pub default_enabled: bool,

    #[serde(default = "default_search_providers")]
    pub providers: Vec<SearchProvider>,

    #[serde(default = "default_redact_queries")]
    pub redact_queries: bool,

    #[serde(default = "default_redaction_rules")]
    pub redaction_rules: Vec<String>,
}

fn default_search_enabled() -> bool {
    true
}
fn default_redact_queries() -> bool {
    true
}

fn default_search_providers() -> Vec<SearchProvider> {
    vec![SearchProvider {
        name: "ddg_mcp".into(),
        provider_type: "mcp".into(),
        command: Some("duckduckgo-mcp".into()),
        timeout_ms: 60_000,
        max_results: 8,
    }]
}

fn default_redaction_rules() -> Vec<String> {
    vec![
        "hostname".into(),
        "ipv4".into(),
        "ipv6".into(),
        "email".into(),
        "internal_domain".into(),
    ]
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_enabled: default_search_enabled(),
            providers: default_search_providers(),
            redact_queries: default_redact_queries(),
            redaction_rules: default_redaction_rules(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchProvider {
    pub name: String,

    #[serde(rename = "type")]
    pub provider_type: String,

    #[serde(default)]
    pub command: Option<String>,

    #[serde(default = "default_provider_timeout_ms")]
    pub timeout_ms: u64,

    #[serde(default = "default_provider_max_results")]
    pub max_results: u32,
}

fn default_provider_timeout_ms() -> u64 {
    60_000
}
fn default_provider_max_results() -> u32 {
    8
}

// ---------- doc_runner ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocRunnerConfig {
    pub allow_programs: Vec<String>,

    #[serde(default = "default_sandbox")]
    pub sandbox: String,

    #[serde(default = "default_doc_runner_timeout_ms")]
    pub timeout_ms: u64,

    #[serde(default = "default_doc_runner_max_output_bytes")]
    pub max_output_bytes: u64,

    #[serde(default = "default_allow_doc_actions")]
    pub allow_doc_actions: Vec<String>,

    #[serde(default)]
    pub extra_bind_ro: Vec<String>,
}

fn default_sandbox() -> String {
    "bwrap".into()
}
fn default_doc_runner_timeout_ms() -> u64 {
    60_000
}
fn default_doc_runner_max_output_bytes() -> u64 {
    10_485_760
}
fn default_allow_doc_actions() -> Vec<String> {
    vec!["help".into(), "man".into(), "info".into()]
}

// ---------- doc_store ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocStoreConfig {
    #[serde(default = "default_true")]
    pub keep_raw_in_memory: bool,

    #[serde(default = "default_true")]
    pub persist_cache: bool,

    #[serde(default = "default_doc_cache_dir")]
    pub cache_dir: String,
}

fn default_true() -> bool {
    true
}
fn default_doc_cache_dir() -> String {
    "~/.cache/cmd-proposer/docs".into()
}

impl Default for DocStoreConfig {
    fn default() -> Self {
        Self {
            keep_raw_in_memory: true,
            persist_cache: true,
            cache_dir: default_doc_cache_dir(),
        }
    }
}

// ---------- subagents ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_subagents_max_parallel")]
    pub max_parallel: u32,

    #[serde(default = "default_subagents_timeout_ms")]
    pub timeout_ms: u64,

    #[serde(default = "default_true")]
    pub kill_on_main_cancel: bool,
}

fn default_subagents_max_parallel() -> u32 {
    12
}
fn default_subagents_timeout_ms() -> u64 {
    1_200_000
}

impl Default for SubagentsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parallel: default_subagents_max_parallel(),
            timeout_ms: default_subagents_timeout_ms(),
            kill_on_main_cancel: true,
        }
    }
}

// ---------- proposal ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProposalConfig {
    #[serde(default = "default_output_language")]
    pub output_language: String,

    #[serde(default = "default_true")]
    pub require_preflight_for_medium_plus: bool,
}

fn default_output_language() -> String {
    "zh-CN".into()
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            output_language: default_output_language(),
            require_preflight_for_medium_plus: true,
        }
    }
}

// ---------- approval ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalConfig {
    #[serde(default)]
    pub execute_enabled: bool,

    #[serde(default = "default_true")]
    pub high_risk_requires_typed_confirmation: bool,
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            execute_enabled: false,
            high_risk_requires_typed_confirmation: true,
        }
    }
}

// ---------- execution_runner ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionRunnerConfig {
    #[serde(default = "default_audit_log")]
    pub audit_log: String,
}

fn default_audit_log() -> String {
    "~/.local/state/cmd-proposer/audit.log".into()
}

impl Default for ExecutionRunnerConfig {
    fn default() -> Self {
        Self {
            audit_log: default_audit_log(),
        }
    }
}

// ---------- risk ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RiskConfig {
    #[serde(default = "default_high_risk_tokens")]
    pub high_risk_tokens: Vec<String>,

    #[serde(default = "default_always_require_review")]
    pub always_require_review: Vec<String>,
}

fn default_high_risk_tokens() -> Vec<String> {
    [
        "delete", "destroy", "apply", "replace", "patch", "restart", "scale", "drain", "uncordon",
        "stop", "disable", "rm", "prune", "force",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

fn default_always_require_review() -> Vec<String> {
    ["sudo", "su", "ssh", "bash", "sh", "zsh"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            high_risk_tokens: default_high_risk_tokens(),
            always_require_review: default_always_require_review(),
        }
    }
}

// ---------- validation ----------

/// Errors returned by [`Config::validate`].
#[derive(Debug, thiserror::Error)]
pub enum ValidateError {
    #[error("`{0}` is required and must not be empty")]
    EmptyRequired(&'static str),

    #[error("`model.max_context_tokens` ({tokens}) must be greater than `thinking.main_agent` ({thinking})")]
    ThinkingExceedsContext { tokens: u32, thinking: u32 },
}

impl Config {
    /// Validate cross-field constraints that `serde` cannot express.
    pub fn validate(&self) -> Result<(), ValidateError> {
        if self.model.base_url.is_empty() {
            return Err(ValidateError::EmptyRequired("model.base_url"));
        }
        if self.model.model_name.is_empty() {
            return Err(ValidateError::EmptyRequired("model.model_name"));
        }
        if self.model.tokenizer.path.is_empty() {
            return Err(ValidateError::EmptyRequired("model.tokenizer.path"));
        }
        if self.doc_runner.allow_programs.is_empty() {
            return Err(ValidateError::EmptyRequired("doc_runner.allow_programs"));
        }
        // api_key MAY be empty (no-auth local endpoint), per SPEC §12.2.
        // Safety invariant: typed confirmation is non-configurable when execution is enabled.
        if self.approval.execute_enabled && !self.approval.high_risk_requires_typed_confirmation {
            return Err(ValidateError::EmptyRequired(
                "approval.high_risk_requires_typed_confirmation must be true when execute_enabled",
            ));
        }
        if self.thinking.main_agent >= self.model.max_context_tokens {
            return Err(ValidateError::ThinkingExceedsContext {
                tokens: self.model.max_context_tokens,
                thinking: self.thinking.main_agent,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_yaml() -> &'static str {
        r#"
model:
  base_url: "http://localhost:8317/v1"
  api_key: "sk-test"
  model_name: "qwen3.6-27b"
  tokenizer:
    path: "/tmp/tokenizer.json"

doc_runner:
  allow_programs: [kubectl, helm]
"#
    }

    #[test]
    fn minimal_config_parses_with_defaults() {
        let cfg: Config = serde_yaml::from_str(minimal_yaml()).expect("parse");
        cfg.validate().expect("validate");

        assert_eq!(cfg.version, 1);
        assert_eq!(cfg.model.provider, "local_openai_compatible");
        assert_eq!(cfg.model.max_context_tokens, 200_000);
        assert_eq!(
            cfg.model.tokenizer.tokenizer_type,
            "huggingface_tokenizer_json"
        );
        assert_eq!(cfg.thinking.main_agent, 32_768);
        assert_eq!(cfg.thinking.subagent_default, 4_096);
        assert_eq!(cfg.runtime.global_timeout_ms, 1_200_000);
        assert_eq!(cfg.runtime.request_timeout_ms, 1_200_000);
        assert_eq!(cfg.runtime.ui_progress_interval_ms, 5_000);
        assert!(cfg.search.default_enabled);
        assert!(cfg.search.redact_queries);
        assert_eq!(cfg.search.providers.len(), 1);
        assert_eq!(cfg.search.providers[0].name, "ddg_mcp");
        assert_eq!(cfg.doc_runner.sandbox, "bwrap");
        assert_eq!(cfg.doc_runner.timeout_ms, 60_000);
        assert_eq!(cfg.doc_runner.max_output_bytes, 10_485_760);
        assert_eq!(
            cfg.doc_runner.allow_doc_actions,
            vec!["help".to_string(), "man".into(), "info".into()]
        );
        assert!(cfg.doc_runner.extra_bind_ro.is_empty());
        assert!(cfg.doc_store.keep_raw_in_memory);
        assert!(cfg.subagents.enabled);
        assert_eq!(cfg.subagents.max_parallel, 12);
        assert_eq!(cfg.proposal.output_language, "zh-CN");
        assert!(!cfg.approval.execute_enabled);
        assert!(cfg.approval.high_risk_requires_typed_confirmation);
        assert!(cfg.risk.high_risk_tokens.contains(&"delete".to_string()));
        assert!(cfg.risk.always_require_review.contains(&"sudo".to_string()));
    }

    #[test]
    fn missing_model_returns_error() {
        let yaml = r#"
doc_runner:
  allow_programs: [kubectl]
"#;
        let err = serde_yaml::from_str::<Config>(yaml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("model"),
            "error should mention `model`, got: {msg}"
        );
    }

    #[test]
    fn missing_doc_runner_returns_error() {
        let yaml = r#"
model:
  base_url: x
  api_key: y
  model_name: z
  tokenizer:
    path: /tmp/t
"#;
        let err = serde_yaml::from_str::<Config>(yaml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("doc_runner"),
            "error should mention `doc_runner`, got: {msg}"
        );
    }

    #[test]
    fn empty_allow_programs_fails_validate() {
        let yaml = r#"
model:
  base_url: x
  api_key: y
  model_name: z
  tokenizer:
    path: /tmp/t

doc_runner:
  allow_programs: []
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            ValidateError::EmptyRequired("doc_runner.allow_programs")
        ));
    }

    #[test]
    fn empty_api_key_validates_ok() {
        let yaml = r#"
model:
  base_url: http://localhost:8000/v1
  api_key: ""
  model_name: m
  tokenizer:
    path: /tmp/t

doc_runner:
  allow_programs: [kubectl]
"#;
        let cfg: Config = serde_yaml::from_str(yaml).unwrap();
        cfg.validate().expect("empty api_key allowed");
    }

    #[test]
    fn thinking_exceeding_context_fails_validate() {
        let mut cfg: Config = serde_yaml::from_str(minimal_yaml()).unwrap();
        cfg.thinking.main_agent = 1_000_000;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ValidateError::ThinkingExceedsContext { .. }));
    }
}

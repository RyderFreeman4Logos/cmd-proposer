//! `cps init` templates.
//!
//! Two flavours, kept verbatim alongside SPEC.md §12.1 / §12.2:
//! - [`minimal_template`] — only required sections, ready for `${VAR}` env vars.
//! - [`full_template`] — every setting shown with its default, commented out.
//!
//! The minimal template MUST round-trip through `serde_yaml::from_str::<Config>`
//! once the referenced env vars are set; this is enforced by tests in this
//! module.

/// Minimal config: just the required sections. See SPEC.md §12.1.
pub fn minimal_template() -> &'static str {
    MINIMAL
}

/// Full reference config with every default shown commented out. See SPEC.md §12.2.
pub fn full_template() -> &'static str {
    FULL
}

const MINIMAL: &str = r#"# cmd-proposer minimal config. Run `cps init --full` for the full reference.
model:
  base_url: ${LOCAL_ROUTER_BASEURL}
  api_key: ${LOCAL_ROUTER_API_KEY}
  model_name: ${LLM_MODEL}
  tokenizer:
    path: "~/.local/share/models/qwen/tokenizer.json"

doc_runner:
  allow_programs: [kubectl, helm, terraform, systemctl, journalctl]
"#;

const FULL: &str = r#"# cmd-proposer full config reference. Every default is shown commented out
# so you can copy-paste-uncomment only the ones you want to override.
version: 1

model:
  base_url: ${LOCAL_ROUTER_BASEURL}           # REQUIRED
  api_key: ${LOCAL_ROUTER_API_KEY}            # REQUIRED (or empty for no-auth local)
  model_name: ${LLM_MODEL}                    # REQUIRED
  # provider: local_openai_compatible         # default: local_openai_compatible
  # max_context_tokens: 200000                # default: 200000
  tokenizer:
    path: "~/.local/share/models/qwen/tokenizer.json"  # REQUIRED for accurate counting
    # type: huggingface_tokenizer_json        # default: huggingface_tokenizer_json

# thinking:
#   main_agent: 32768                         # default: 32768
#   subagent_default: 4096                    # default: 4096; main agent may override per-spawn

# runtime:
#   global_timeout_ms: 1200000                # default: 1200000 (20 min)
#   request_timeout_ms: 1200000               # default: 1200000
#   ui_progress_interval_ms: 5000             # default: 5000

# search:
#   default_enabled: true                     # default: true
#   providers:                                # default: [ddg_mcp]
#     - name: ddg_mcp
#       type: mcp
#       command: "duckduckgo-mcp"
#       timeout_ms: 60000
#       max_results: 8
#   redact_queries: true                      # default: true
#   redaction_rules:                          # default: all of the below
#     - hostname
#     - ipv4
#     - ipv6
#     - email
#     - internal_domain

doc_runner:
  allow_programs: [kubectl, helm, terraform]  # REQUIRED: whitelist of explorable programs
  # sandbox: bwrap                            # default: bwrap
  # timeout_ms: 60000                         # default: 60000
  # max_output_bytes: 10485760                # default: 10MB
  # allow_doc_actions: [help, man, info]      # default: [help, man, info]

# doc_store:
#   keep_raw_in_memory: true                  # default: true
#   persist_cache: true                       # default: true
#   cache_dir: "~/.cache/cmd-proposer/docs"   # default

# subagents:
#   enabled: true                             # default: true
#   max_parallel: 12                          # default: 12
#   timeout_ms: 1200000                       # default: 1200000
#   kill_on_main_cancel: true                 # default: true

# proposal:
#   output_language: zh-CN                    # default: zh-CN
#   require_preflight_for_medium_plus: true   # default: true
#   # require_argv, forbid_shell_string, require_evidence are always true (not configurable)

# approval:
#   execute_enabled: false                    # default: false
#   high_risk_requires_typed_confirmation: true  # default: true (not configurable when execute_enabled)

# execution_runner:                           # only relevant when approval.execute_enabled: true
#   audit_log: "~/.local/state/cmd-proposer/audit.log"  # default

# risk:                                       # override only if you need to add/remove tokens
#   high_risk_tokens: [delete, destroy, apply, replace, patch, restart,
#                      scale, drain, uncordon, stop, disable, rm, prune, force]
#   always_require_review: [sudo, su, ssh, bash, sh, zsh]
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interpolate::interpolate_with, Config};
    use std::collections::HashMap;

    fn fake_env() -> HashMap<&'static str, &'static str> {
        HashMap::from([
            ("LOCAL_ROUTER_BASEURL", "http://localhost:8317/v1"),
            ("LOCAL_ROUTER_API_KEY", "sk-test"),
            ("LLM_MODEL", "qwen-test"),
        ])
    }

    #[test]
    fn minimal_template_is_valid_yaml() {
        let env = fake_env();
        let resolved = interpolate_with(minimal_template(), |k| {
            env.get(k).map(|s| (*s).to_string())
        })
        .expect("interpolate");
        let cfg: Config = serde_yaml::from_str(&resolved).expect("parse minimal");
        cfg.validate().expect("validate minimal");
        assert_eq!(cfg.model.base_url, "http://localhost:8317/v1");
        assert!(cfg.doc_runner.allow_programs.contains(&"kubectl".to_string()));
    }

    #[test]
    fn full_template_is_valid_yaml() {
        let env = fake_env();
        let resolved = interpolate_with(full_template(), |k| {
            env.get(k).map(|s| (*s).to_string())
        })
        .expect("interpolate");
        let cfg: Config = serde_yaml::from_str(&resolved).expect("parse full");
        cfg.validate().expect("validate full");
        // Defaults survived the commented-out sections.
        assert_eq!(cfg.model.provider, "local_openai_compatible");
        assert_eq!(cfg.model.max_context_tokens, 200_000);
        assert_eq!(cfg.thinking.main_agent, 32_768);
        assert_eq!(cfg.search.providers[0].name, "ddg_mcp");
    }

    #[test]
    fn templates_differ() {
        assert_ne!(minimal_template(), full_template());
        assert!(full_template().len() > minimal_template().len());
    }
}

//! Tool-call validation, proposal checking, and risk classification (the Policy Gate).

use std::collections::{HashMap, HashSet};

use cps_proposal::{CommandProposal, Risk};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const HIGH_RISK_TOKENS: &[&str] = &[
    "delete",
    "destroy",
    "--force",
    "--all",
    "rm",
    "--grace-period=0",
    "drop",
    "truncate",
];

const SHELL_WRAPPERS: &[&str] = &["sudo", "su", "bash", "sh", "zsh"];
const READ_ONLY_VERBS: &[&str] = &[
    "get",
    "describe",
    "list",
    "status",
    "logs",
    "log",
    "explain",
    "version",
    "view",
    "current-context",
    "api-resources",
    "api-versions",
];
const HIGH_RISK_VERBS: &[&str] = &[
    "delete", "scale", "drain", "replace", "apply", "stop", "destroy", "drop", "truncate", "rm",
];
const MUTATING_VERBS: &[&str] = &[
    "rollout", "restart", "create", "set", "edit", "annotate", "label", "cordon", "uncordon",
    "patch",
];

/// Policy options loaded from the runtime configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub allow_programs: Vec<String>,
    pub allowed_roles: Vec<String>,
    pub role_tool_permissions: HashMap<String, Vec<String>>,
    pub search_enabled: bool,
    pub execute_enabled: bool,
    pub max_subagent_context: usize,
}

/// Runtime policy gate.
#[derive(Debug, Clone)]
pub struct PolicyGate {
    config: PolicyConfig,
}

impl PolicyGate {
    #[must_use]
    pub fn new(config: PolicyConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Validate a tool call before the runtime executes it.
    ///
    /// The doc store is represented by [`DocLookup`] so this crate does not
    /// depend on `cps-doc-store`.
    pub fn validate_tool_call<D: DocLookup + ?Sized>(
        &self,
        call: &ToolCall,
        doc_store: &D,
    ) -> Result<(), PolicyError> {
        match call.name.as_str() {
            "read_help" => self.validate_read_help(call),
            "doc_token_count" | "doc_preview" | "doc_grep" | "doc_section" | "doc_lines"
            | "doc_expand_around" => validate_doc_call(call, doc_store),
            "web_search" => self.validate_web_search(call),
            "spawn" => self.validate_spawn(call),
            "execute" => self.validate_execute(call),
            other => Err(PolicyError::UnknownTool(other.to_owned())),
        }
    }

    /// Run proposal-level checks before showing a command to the operator.
    #[must_use]
    pub fn check_proposal(&self, proposal: &CommandProposal) -> Vec<PolicyFinding> {
        let mut findings = Vec::new();

        if proposal.argv.is_empty() {
            findings.push(PolicyFinding::reject(
                PolicyFindingCode::EmptyArgv,
                "proposal argv is empty",
            ));
            return findings;
        }

        if looks_like_shell_string(&proposal.argv) {
            findings.push(PolicyFinding::reject(
                PolicyFindingCode::ShellOperator,
                "proposal argv looks like a shell command string; commands must be argv arrays",
            ));
        }

        for arg in &proposal.argv {
            if has_shell_metacharacter(arg) {
                findings.push(PolicyFinding::reject(
                    PolicyFindingCode::ShellOperator,
                    format!("argument contains shell metacharacter: {arg}"),
                ));
            }
        }

        let program = &proposal.argv[0];
        if is_shell_wrapper(program) {
            findings.push(PolicyFinding::reject(
                PolicyFindingCode::WrapperProgram,
                format!("command is wrapped by forbidden program: {program}"),
            ));
        }

        if !self.program_allowed(program) {
            findings.push(PolicyFinding::reject(
                PolicyFindingCode::ProgramNotAllowed,
                format!("program is not allowed: {program}"),
            ));
        }

        let classified = self.classify_risk(&proposal.argv);
        if classified > proposal.risk {
            findings.push(PolicyFinding::escalation(
                format!(
                    "proposal risk {:?} is below policy classification {:?}",
                    proposal.risk, classified
                ),
                classified,
            ));
        }

        if should_warn_missing_kubectl_namespace(&proposal.argv) {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::MissingScope,
                "kubectl command is missing an explicit namespace",
            ));
        }
        if should_warn_missing_kubectl_context(&proposal.argv) {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::MissingScope,
                "kubectl command is missing an explicit context",
            ));
        }
        if should_warn_missing_aws_region(&proposal.argv) {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::MissingScope,
                "aws command is missing an explicit region",
            ));
        }
        if should_warn_missing_aws_profile(&proposal.argv) {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::MissingScope,
                "aws command is missing an explicit profile",
            ));
        }

        self.check_evidence(proposal, &mut findings);

        if classified >= Risk::Medium && proposal.preflight.is_empty() {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::PreflightRequired,
                "medium-or-higher risk commands require at least one preflight command",
            ));
        }

        if needs_safer_alternative(&proposal.argv) && !has_safer_alternative(proposal) {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::SaferAlternative,
                "destructive command should include a safer alternative such as --dry-run, diff, or plan",
            ));
        }

        findings
    }

    /// Classify command risk from argv tokens.
    #[must_use]
    pub fn classify_risk(&self, argv: &[String]) -> Risk {
        classify_risk(argv)
    }

    fn validate_read_help(&self, call: &ToolCall) -> Result<(), PolicyError> {
        let program = string_arg(&call.arguments, "program")
            .ok_or_else(|| PolicyError::ProgramNotAllowed("<missing>".to_owned()))?;
        if !self.program_allowed(program) {
            return Err(PolicyError::ProgramNotAllowed(program.to_owned()));
        }

        for subcommand in string_array_arg(&call.arguments, "subcommands") {
            if !is_safe_subcommand(&subcommand) {
                return Err(PolicyError::UnsafeSubcommand(subcommand));
            }
        }

        Ok(())
    }

    fn validate_web_search(&self, call: &ToolCall) -> Result<(), PolicyError> {
        if !self.config.search_enabled {
            return Err(PolicyError::SearchDisabled);
        }

        if let Some(query) = string_arg(&call.arguments, "query") {
            tracing::trace!(
                query_len = query.len(),
                "web search query accepted for redaction"
            );
        }
        Ok(())
    }

    fn validate_spawn(&self, call: &ToolCall) -> Result<(), PolicyError> {
        let role = string_arg(&call.arguments, "role")
            .ok_or_else(|| PolicyError::RoleNotAllowed("<missing>".to_owned()))?;
        if !self
            .config
            .allowed_roles
            .iter()
            .any(|allowed| allowed == role)
        {
            return Err(PolicyError::RoleNotAllowed(role.to_owned()));
        }

        let requested = usize_arg(&call.arguments, "context_tokens").unwrap_or(0);
        if requested > self.config.max_subagent_context {
            return Err(PolicyError::ContextBudgetExceeded {
                requested,
                limit: self.config.max_subagent_context,
            });
        }

        let allowed_tools = self
            .config
            .role_tool_permissions
            .get(role)
            .cloned()
            .unwrap_or_default();
        let allowed_tools: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
        for tool in string_array_arg(&call.arguments, "allowed_tools") {
            if !allowed_tools.contains(tool.as_str()) {
                return Err(PolicyError::ToolNotAllowedForRole {
                    tool,
                    role: role.to_owned(),
                });
            }
        }

        Ok(())
    }

    fn validate_execute(&self, call: &ToolCall) -> Result<(), PolicyError> {
        if !self.config.execute_enabled {
            return Err(PolicyError::ExecutionDisabled);
        }
        if !bool_arg(&call.arguments, "user_approved").unwrap_or(false) {
            return Err(PolicyError::ExecutionDisabled);
        }

        let argv = string_array_arg(&call.arguments, "argv");
        let program = argv
            .first()
            .ok_or_else(|| PolicyError::ProgramNotAllowed("<missing>".to_owned()))?;
        if !self.program_allowed(program) {
            return Err(PolicyError::ProgramNotAllowed(program.to_owned()));
        }

        for arg in argv {
            if has_shell_metacharacter(&arg) {
                return Err(PolicyError::ShellMetacharacter(arg));
            }
        }

        Ok(())
    }

    fn program_allowed(&self, program: &str) -> bool {
        self.config
            .allow_programs
            .iter()
            .any(|allowed| allowed == program)
    }

    fn check_evidence(&self, proposal: &CommandProposal, findings: &mut Vec<PolicyFinding>) {
        if proposal.evidence.is_empty() {
            findings.push(PolicyFinding::warn(
                PolicyFindingCode::MissingEvidence,
                "proposal has no evidence",
            ));
            return;
        }

        for token in evidence_required_tokens(&proposal.argv) {
            if !evidence_mentions(proposal, &token) {
                findings.push(PolicyFinding::warn(
                    PolicyFindingCode::MissingEvidence,
                    format!("proposal evidence does not mention key token: {token}"),
                ));
            }
        }
    }
}

/// Minimal tool-call representation consumed by the policy gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

impl ToolCall {
    #[must_use]
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// Build a policy tool call from the raw JSON argument string used by
    /// OpenAI-compatible function calls.
    pub fn from_json_arguments(name: impl Into<String>, arguments: &str) -> anyhow::Result<Self> {
        let arguments = serde_json::from_str(arguments)?;
        Ok(Self::new(name, arguments))
    }
}

/// Trait boundary for checking whether a doc_id exists.
pub trait DocLookup {
    fn contains_doc(&self, doc_id: &str) -> bool;
}

impl DocLookup for HashSet<String> {
    fn contains_doc(&self, doc_id: &str) -> bool {
        self.contains(doc_id)
    }
}

impl DocLookup for [&str] {
    fn contains_doc(&self, doc_id: &str) -> bool {
        self.contains(&doc_id)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PolicyError {
    #[error("program not allowed: {0}")]
    ProgramNotAllowed(String),
    #[error("unsafe subcommand: {0}")]
    UnsafeSubcommand(String),
    #[error("doc not found: {0}")]
    DocNotFound(String),
    #[error("web search is disabled")]
    SearchDisabled,
    #[error("role not allowed: {0}")]
    RoleNotAllowed(String),
    #[error("tool {tool} is not allowed for role {role}")]
    ToolNotAllowedForRole { tool: String, role: String },
    #[error("context budget exceeded: requested {requested}, limit {limit}")]
    ContextBudgetExceeded { requested: usize, limit: usize },
    #[error("execution is disabled or missing user approval")]
    ExecutionDisabled,
    #[error("shell metacharacter found in argument: {0}")]
    ShellMetacharacter(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyFindingSeverity {
    Reject,
    Warn,
    Escalate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyFindingCode {
    EmptyArgv,
    ShellOperator,
    ProgramNotAllowed,
    WrapperProgram,
    RiskEscalation,
    MissingScope,
    MissingEvidence,
    PreflightRequired,
    SaferAlternative,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyFinding {
    pub severity: PolicyFindingSeverity,
    pub code: PolicyFindingCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk: Option<Risk>,
}

impl PolicyFinding {
    #[must_use]
    pub fn reject(code: PolicyFindingCode, message: impl Into<String>) -> Self {
        Self {
            severity: PolicyFindingSeverity::Reject,
            code,
            message: message.into(),
            risk: None,
        }
    }

    #[must_use]
    pub fn warn(code: PolicyFindingCode, message: impl Into<String>) -> Self {
        Self {
            severity: PolicyFindingSeverity::Warn,
            code,
            message: message.into(),
            risk: None,
        }
    }

    #[must_use]
    pub fn escalation(message: impl Into<String>, risk: Risk) -> Self {
        Self {
            severity: PolicyFindingSeverity::Escalate,
            code: PolicyFindingCode::RiskEscalation,
            message: message.into(),
            risk: Some(risk),
        }
    }
}

fn validate_doc_call<D: DocLookup + ?Sized>(
    call: &ToolCall,
    doc_store: &D,
) -> Result<(), PolicyError> {
    let doc_id = string_arg(&call.arguments, "doc")
        .or_else(|| string_arg(&call.arguments, "doc_id"))
        .ok_or_else(|| PolicyError::DocNotFound("<missing>".to_owned()))?;

    if !doc_store.contains_doc(doc_id) {
        return Err(PolicyError::DocNotFound(doc_id.to_owned()));
    }

    Ok(())
}

fn classify_risk(argv: &[String]) -> Risk {
    let tokens = normalized_tokens(argv);

    if tokens.is_empty() {
        return Risk::Low;
    }

    if has_critical_pattern(&tokens) {
        return Risk::Critical;
    }

    if tokens
        .iter()
        .any(|token| token == "--all-namespaces" || token == "-a")
        || tokens
            .iter()
            .any(|token| HIGH_RISK_TOKENS.contains(&token.as_str()))
        || tokens
            .iter()
            .any(|token| HIGH_RISK_VERBS.contains(&token.as_str()))
    {
        return Risk::High;
    }

    if tokens
        .iter()
        .any(|token| MUTATING_VERBS.contains(&token.as_str()))
    {
        return Risk::Medium;
    }

    if tokens
        .iter()
        .any(|token| READ_ONLY_VERBS.contains(&token.as_str()))
    {
        return Risk::Low;
    }

    Risk::Medium
}

fn has_critical_pattern(tokens: &[String]) -> bool {
    if tokens
        .iter()
        .any(|token| contains_word(token, &["destroy", "drop", "truncate"]))
    {
        return true;
    }

    if is_rm_recursive_force(tokens) {
        return true;
    }

    let has_all = tokens.iter().any(|token| token == "--all");
    let has_force = tokens.iter().any(|token| token == "--force");
    let destructive = tokens.iter().any(|token| {
        contains_word(
            token,
            &[
                "delete", "destroy", "rm", "drop", "truncate", "replace", "apply", "stop",
            ],
        )
    });

    (has_all || has_force) && destructive
}

fn is_rm_recursive_force(tokens: &[String]) -> bool {
    if !tokens.iter().any(|token| token == "rm") {
        return false;
    }

    tokens.iter().any(|token| {
        if !token.starts_with('-') || token.starts_with("--") {
            return false;
        }
        token.contains('r') && token.contains('f')
    }) || (tokens.iter().any(|token| token == "-r" || token == "-R")
        && tokens.iter().any(|token| token == "-f"))
}

fn normalized_tokens(argv: &[String]) -> Vec<String> {
    argv.iter()
        .map(|arg| arg.trim().to_ascii_lowercase())
        .filter(|arg| !arg.is_empty())
        .collect()
}

fn contains_word(token: &str, words: &[&str]) -> bool {
    token
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
        .any(|part| words.contains(&part))
}

fn is_safe_subcommand(subcommand: &str) -> bool {
    Regex::new(r"^[A-Za-z0-9-]+$")
        .map(|regex| regex.is_match(subcommand))
        .unwrap_or(false)
}

fn has_shell_metacharacter(arg: &str) -> bool {
    Regex::new(r#"[;&|<>`$(){}\[\]*?!'"\n\r]"#)
        .map(|regex| regex.is_match(arg))
        .unwrap_or(true)
}

fn looks_like_shell_string(argv: &[String]) -> bool {
    argv.len() == 1 && argv[0].split_whitespace().nth(1).is_some()
}

fn string_arg<'a>(arguments: &'a Value, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(Value::as_str)
}

fn string_array_arg(arguments: &Value, key: &str) -> Vec<String> {
    arguments
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn usize_arg(arguments: &Value, key: &str) -> Option<usize> {
    arguments
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
}

fn bool_arg(arguments: &Value, key: &str) -> Option<bool> {
    arguments.get(key).and_then(Value::as_bool)
}

fn is_shell_wrapper(program: &str) -> bool {
    let name = program.rsplit('/').next().unwrap_or(program);
    SHELL_WRAPPERS.contains(&name)
}

fn should_warn_missing_kubectl_namespace(argv: &[String]) -> bool {
    argv.first().is_some_and(|program| program == "kubectl")
        && is_kubectl_namespaced(argv)
        && !has_flag(argv, &["-n", "--namespace"])
}

fn should_warn_missing_kubectl_context(argv: &[String]) -> bool {
    argv.first().is_some_and(|program| program == "kubectl") && !has_flag(argv, &["--context"])
}

fn is_kubectl_namespaced(argv: &[String]) -> bool {
    argv.iter().skip(1).any(|arg| {
        matches!(
            arg.as_str(),
            "get" | "describe" | "logs" | "delete" | "apply" | "rollout" | "scale" | "patch"
        )
    })
}

fn should_warn_missing_aws_region(argv: &[String]) -> bool {
    argv.first().is_some_and(|program| program == "aws") && !has_flag(argv, &["--region"])
}

fn should_warn_missing_aws_profile(argv: &[String]) -> bool {
    argv.first().is_some_and(|program| program == "aws") && !has_flag(argv, &["--profile"])
}

fn has_flag(argv: &[String], names: &[&str]) -> bool {
    argv.iter().enumerate().any(|(index, arg)| {
        names.iter().any(|name| {
            arg == name
                || arg
                    .strip_prefix(name)
                    .is_some_and(|suffix| suffix.starts_with('='))
                || (index > 0 && argv[index - 1] == *name)
        })
    })
}

fn evidence_required_tokens(argv: &[String]) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut skip_next = false;

    for token in argv.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if token.starts_with('-') {
            tokens.push(token.clone());
            if !token.contains('=') {
                skip_next = true;
            }
            continue;
        }
        if token.contains('/') {
            continue;
        }
        if READ_ONLY_VERBS.contains(&token.as_str())
            || MUTATING_VERBS.contains(&token.as_str())
            || HIGH_RISK_VERBS.contains(&token.as_str())
        {
            tokens.push(token.clone());
        }
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn evidence_mentions(proposal: &CommandProposal, token: &str) -> bool {
    let needle = token.to_ascii_lowercase();
    proposal.evidence.iter().any(|evidence| {
        evidence.claim.to_ascii_lowercase().contains(&needle)
            || evidence.source.doc.to_ascii_lowercase().contains(&needle)
    })
}

fn needs_safer_alternative(argv: &[String]) -> bool {
    classify_risk(argv) >= Risk::High
        && normalized_tokens(argv)
            .iter()
            .any(|token| HIGH_RISK_VERBS.contains(&token.as_str()))
}

fn has_safer_alternative(proposal: &CommandProposal) -> bool {
    let argv_has_alternative = proposal.argv.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--dry-run" | "--dry-run=client" | "--dry-run=server" | "diff" | "plan"
        )
    });
    let preflight_has_alternative = proposal.preflight.iter().any(|cmd| {
        cmd.argv.iter().any(|arg| {
            matches!(
                arg.as_str(),
                "--dry-run" | "--dry-run=client" | "--dry-run=server" | "diff" | "plan"
            )
        })
    });

    argv_has_alternative || preflight_has_alternative
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use cps_proposal::{
        Confidence, Evidence, EvidenceKind, EvidenceSource, PreflightCmd, RollbackInfo,
        RollbackStatus,
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn validate_read_help_with_allowed_program_ok() {
        let gate = gate();
        let call = ToolCall::new(
            "read_help",
            json!({
                "program": "kubectl",
                "subcommands": ["rollout", "restart"]
            }),
        );

        assert_eq!(gate.validate_tool_call(&call, &[][..]), Ok(()));
    }

    #[test]
    fn validate_read_help_with_disallowed_program_rejects() {
        let gate = gate();
        let call = ToolCall::new("read_help", json!({ "program": "terraform" }));

        assert_eq!(
            gate.validate_tool_call(&call, &[][..]),
            Err(PolicyError::ProgramNotAllowed("terraform".to_owned()))
        );
    }

    #[test]
    fn validate_read_help_with_shell_metacharacter_subcommand_rejects() {
        let gate = gate();
        let call = ToolCall::new(
            "read_help",
            json!({
                "program": "kubectl",
                "subcommands": ["get;rm"]
            }),
        );

        assert_eq!(
            gate.validate_tool_call(&call, &[][..]),
            Err(PolicyError::UnsafeSubcommand("get;rm".to_owned()))
        );
    }

    #[test]
    fn validate_doc_call_requires_existing_doc() {
        let gate = gate();
        let call = ToolCall::new(
            "doc_grep",
            json!({ "doc": "kubectl-help", "pattern": "delete" }),
        );
        let docs = HashSet::from(["kubectl-help".to_owned()]);

        assert_eq!(gate.validate_tool_call(&call, &docs), Ok(()));
    }

    #[test]
    fn validate_doc_call_missing_doc_rejects() {
        let gate = gate();
        let call = ToolCall::new(
            "doc_lines",
            json!({ "doc": "missing", "start": 1, "end": 2 }),
        );
        let docs = HashSet::from(["other".to_owned()]);

        assert_eq!(
            gate.validate_tool_call(&call, &docs),
            Err(PolicyError::DocNotFound("missing".to_owned()))
        );
    }

    #[test]
    fn validate_web_search_requires_enabled_config() {
        let mut config = config();
        config.search_enabled = false;
        let gate = PolicyGate::new(config);
        let call = ToolCall::new(
            "web_search",
            json!({ "query": "kubectl delete", "max_results": 5 }),
        );

        assert_eq!(
            gate.validate_tool_call(&call, &[][..]),
            Err(PolicyError::SearchDisabled)
        );
    }

    #[test]
    fn validate_spawn_enforces_role_tool_and_context_policy() {
        let gate = gate();
        let call = ToolCall::new(
            "spawn",
            json!({
                "role": "risk_reviewer",
                "allowed_tools": ["doc_grep", "web_search"],
                "context_tokens": 1024
            }),
        );

        assert_eq!(
            gate.validate_tool_call(&call, &[][..]),
            Err(PolicyError::ToolNotAllowedForRole {
                tool: "web_search".to_owned(),
                role: "risk_reviewer".to_owned(),
            })
        );

        let oversized = ToolCall::new(
            "spawn",
            json!({
                "role": "risk_reviewer",
                "allowed_tools": ["doc_grep"],
                "context_tokens": 9000
            }),
        );
        assert_eq!(
            gate.validate_tool_call(&oversized, &[][..]),
            Err(PolicyError::ContextBudgetExceeded {
                requested: 9000,
                limit: 4096,
            })
        );
    }

    #[test]
    fn validate_execute_requires_enabled_approval_allowlist_and_safe_args() {
        let gate = gate();
        let call = ToolCall::new(
            "execute",
            json!({
                "user_approved": true,
                "argv": ["kubectl", "get", "pods"]
            }),
        );
        assert_eq!(gate.validate_tool_call(&call, &[][..]), Ok(()));

        let unsafe_call = ToolCall::new(
            "execute",
            json!({
                "user_approved": true,
                "argv": ["kubectl", "get;rm", "pods"]
            }),
        );
        assert_eq!(
            gate.validate_tool_call(&unsafe_call, &[][..]),
            Err(PolicyError::ShellMetacharacter("get;rm".to_owned()))
        );
    }

    #[test]
    fn classify_risk_for_read_only_commands_is_low() {
        let gate = gate();

        assert_eq!(
            gate.classify_risk(&argv(["kubectl", "get", "pods"])),
            Risk::Low
        );
        assert_eq!(
            gate.classify_risk(&argv(["kubectl", "describe", "deployment/api"])),
            Risk::Low
        );
        assert_eq!(
            gate.classify_risk(&argv(["systemctl", "status", "nginx"])),
            Risk::Low
        );
    }

    #[test]
    fn classify_risk_for_destructive_commands_is_high_or_critical() {
        let gate = gate();

        assert_eq!(
            gate.classify_risk(&argv(["kubectl", "delete", "pod", "api"])),
            Risk::High
        );
        assert_eq!(
            gate.classify_risk(&argv(["kubectl", "delete", "pod", "--all"])),
            Risk::Critical
        );
        assert_eq!(
            gate.classify_risk(&argv(["rm", "-rf", "/tmp/x"])),
            Risk::Critical
        );
        assert_eq!(
            gate.classify_risk(&argv(["psql", "-c", "truncate table users"])),
            Risk::Critical
        );
    }

    #[test]
    fn classify_risk_for_scoped_rollout_restart_is_medium() {
        let gate = gate();

        assert_eq!(
            gate.classify_risk(&argv([
                "kubectl",
                "-n",
                "payments",
                "rollout",
                "restart",
                "deployment/api"
            ])),
            Risk::Medium
        );
    }

    #[test]
    fn check_proposal_detects_sudo_wrapping() {
        let gate = gate();
        let proposal = CommandProposal {
            argv: argv(["sudo", "kubectl", "get", "pods"]),
            ..proposal(argv(["kubectl", "get", "pods"]), Risk::Low)
        };

        let findings = gate.check_proposal(&proposal);

        assert!(has_code(&findings, PolicyFindingCode::WrapperProgram));
    }

    #[test]
    fn check_proposal_escalates_risk_for_high_risk_tokens() {
        let gate = gate();
        let proposal = proposal(argv(["kubectl", "delete", "pod", "api"]), Risk::Low);

        let findings = gate.check_proposal(&proposal);

        assert!(findings.iter().any(|finding| {
            finding.code == PolicyFindingCode::RiskEscalation && finding.risk == Some(Risk::High)
        }));
    }

    #[test]
    fn check_proposal_warns_on_missing_namespace() {
        let gate = gate();
        let proposal = proposal(argv(["kubectl", "get", "pods"]), Risk::Low);

        let findings = gate.check_proposal(&proposal);

        assert!(findings.iter().any(|finding| {
            finding.code == PolicyFindingCode::MissingScope && finding.message.contains("namespace")
        }));
    }

    #[test]
    fn check_proposal_requires_preflight_for_medium_plus() {
        let gate = gate();
        let proposal = proposal(
            argv([
                "kubectl",
                "-n",
                "payments",
                "rollout",
                "restart",
                "deployment/api",
            ]),
            Risk::Medium,
        );

        let findings = gate.check_proposal(&proposal);

        assert!(has_code(&findings, PolicyFindingCode::PreflightRequired));
    }

    fn gate() -> PolicyGate {
        PolicyGate::new(config())
    }

    fn config() -> PolicyConfig {
        let mut permissions = HashMap::new();
        permissions.insert(
            "doc_explorer".to_owned(),
            vec![
                "read_help".to_owned(),
                "doc_grep".to_owned(),
                "doc_lines".to_owned(),
            ],
        );
        permissions.insert(
            "risk_reviewer".to_owned(),
            vec!["doc_grep".to_owned(), "doc_section".to_owned()],
        );

        PolicyConfig {
            allow_programs: vec![
                "kubectl".to_owned(),
                "systemctl".to_owned(),
                "rm".to_owned(),
                "psql".to_owned(),
            ],
            allowed_roles: vec!["doc_explorer".to_owned(), "risk_reviewer".to_owned()],
            role_tool_permissions: permissions,
            search_enabled: true,
            execute_enabled: true,
            max_subagent_context: 4096,
        }
    }

    fn proposal(argv: Vec<String>, risk: Risk) -> CommandProposal {
        CommandProposal {
            summary: "test proposal".to_owned(),
            display: argv.join(" "),
            argv,
            risk,
            risk_reasons: Vec::new(),
            assumptions: Vec::new(),
            preflight: Vec::new(),
            rollback: Some(RollbackInfo {
                available: RollbackStatus::Partial,
                notes: "rollback depends on the target command".to_owned(),
            }),
            evidence: vec![Evidence {
                claim: "kubectl get/delete/rollout/restart flags are documented".to_owned(),
                source: EvidenceSource {
                    kind: EvidenceKind::LocalDoc,
                    doc: "kubectl-help".to_owned(),
                    lines: Some((1, 20)),
                },
                confidence: Confidence::High,
            }],
            missing_confirmations: Vec::new(),
        }
    }

    fn argv<const N: usize>(items: [&str; N]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn has_code(findings: &[PolicyFinding], code: PolicyFindingCode) -> bool {
        findings.iter().any(|finding| finding.code == code)
    }

    #[test]
    fn proposal_fixture_can_include_preflight() {
        let mut proposal = proposal(argv(["kubectl", "delete", "pod", "api"]), Risk::High);
        proposal.preflight.push(PreflightCmd {
            argv: argv(["kubectl", "get", "pod", "api"]),
            reason: "confirm target exists".to_owned(),
        });

        assert!(!proposal.preflight.is_empty());
    }
}

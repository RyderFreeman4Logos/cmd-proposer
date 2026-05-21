//! Web search providers (MCP/DuckDuckGo) and query redaction.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use thiserror::Error;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryRedactor {
    rules: Vec<RedactionRule>,
    enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedactionRule {
    Hostname,
    IPv4,
    IPv6,
    Email,
    InternalDomain(Vec<String>),
    K8sNamespace(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionResult {
    pub redacted_query: String,
    pub redactions_applied: Vec<RedactionApplied>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionApplied {
    pub rule: String,
    pub original: String,
    pub position: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RedactionError {
    #[error("unknown redaction rule `{0}`")]
    UnknownRule(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RedactionCandidate {
    rule: &'static str,
    original: String,
    start: usize,
    end: usize,
    placeholder: &'static str,
    priority: u8,
}

impl Default for QueryRedactor {
    fn default() -> Self {
        Self::new(true)
    }
}

impl QueryRedactor {
    pub fn new(enabled: bool) -> Self {
        Self {
            rules: default_rules(Vec::new(), Vec::new()),
            enabled,
        }
    }

    pub fn with_rules(rules: Vec<RedactionRule>, enabled: bool) -> Self {
        Self { rules, enabled }
    }

    pub fn from_config(
        redact_queries: bool,
        redaction_rules: &[String],
        internal_domains: Vec<String>,
        sensitive_namespaces: Vec<String>,
    ) -> Result<Self, RedactionError> {
        let mut rules = Vec::with_capacity(redaction_rules.len());

        for rule in redaction_rules {
            match rule.trim().to_ascii_lowercase().as_str() {
                "hostname" => rules.push(RedactionRule::Hostname),
                "ipv4" => rules.push(RedactionRule::IPv4),
                "ipv6" => rules.push(RedactionRule::IPv6),
                "email" => rules.push(RedactionRule::Email),
                "internal_domain" | "internal_domains" => {
                    rules.push(RedactionRule::InternalDomain(internal_domains.clone()));
                }
                "k8s_namespace"
                | "k8s_namespaces"
                | "sensitive_namespace"
                | "sensitive_namespaces" => {
                    rules.push(RedactionRule::K8sNamespace(sensitive_namespaces.clone()));
                }
                other => return Err(RedactionError::UnknownRule(other.to_owned())),
            }
        }

        Ok(Self::with_rules(rules, redact_queries))
    }

    pub fn rules(&self) -> &[RedactionRule] {
        &self.rules
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn redact(&self, query: &str) -> RedactionResult {
        if !self.enabled || query.is_empty() {
            if !self.enabled {
                debug!(
                    redaction_enabled = false,
                    query_len = query.len(),
                    "search query redaction disabled"
                );
            }

            return RedactionResult {
                redacted_query: query.to_owned(),
                redactions_applied: Vec::new(),
            };
        }

        let selected = select_non_overlapping_candidates(self.collect_candidates(query));
        let mut redacted_query = String::with_capacity(query.len());
        let mut last_end = 0;
        let mut redactions_applied = Vec::with_capacity(selected.len());

        for candidate in selected {
            redacted_query.push_str(&query[last_end..candidate.start]);
            redacted_query.push_str(candidate.placeholder);
            last_end = candidate.end;
            redactions_applied.push(RedactionApplied {
                rule: candidate.rule.to_owned(),
                original: candidate.original,
                position: candidate.start,
            });
        }

        redacted_query.push_str(&query[last_end..]);
        debug!(
            redacted_query = %redacted_query,
            redactions_applied = redactions_applied.len(),
            "redacted search query"
        );

        RedactionResult {
            redacted_query,
            redactions_applied,
        }
    }

    fn collect_candidates(&self, query: &str) -> Vec<RedactionCandidate> {
        let mut candidates = Vec::new();

        for rule in &self.rules {
            match rule {
                RedactionRule::Hostname => collect_hostname_candidates(query, &mut candidates),
                RedactionRule::IPv4 => collect_static_regex_candidates(
                    query,
                    ipv4_regex(),
                    "ipv4",
                    "<IP>",
                    is_ip_boundary,
                    &mut candidates,
                ),
                RedactionRule::IPv6 => collect_static_regex_candidates(
                    query,
                    ipv6_regex(),
                    "ipv6",
                    "<IP>",
                    is_ipv6_boundary,
                    &mut candidates,
                ),
                RedactionRule::Email => collect_static_regex_candidates(
                    query,
                    email_regex(),
                    "email",
                    "<EMAIL>",
                    is_email_boundary,
                    &mut candidates,
                ),
                RedactionRule::InternalDomain(patterns) => collect_pattern_candidates(
                    query,
                    patterns,
                    PatternKind::Domain,
                    "internal_domain",
                    "<DOMAIN>",
                    is_domain_boundary,
                    &mut candidates,
                ),
                RedactionRule::K8sNamespace(patterns) => collect_pattern_candidates(
                    query,
                    patterns,
                    PatternKind::Namespace,
                    "k8s_namespace",
                    "<NAMESPACE>",
                    is_namespace_boundary,
                    &mut candidates,
                ),
            }
        }

        candidates
    }
}

pub fn default_rules(
    internal_domains: Vec<String>,
    sensitive_namespaces: Vec<String>,
) -> Vec<RedactionRule> {
    vec![
        RedactionRule::Hostname,
        RedactionRule::IPv4,
        RedactionRule::IPv6,
        RedactionRule::Email,
        RedactionRule::InternalDomain(internal_domains),
        RedactionRule::K8sNamespace(sensitive_namespaces),
    ]
}

fn collect_hostname_candidates(query: &str, candidates: &mut Vec<RedactionCandidate>) {
    for found in hostname_regex().find_iter(query) {
        let original = found.as_str();
        if !original.bytes().any(|byte| byte.is_ascii_alphabetic()) {
            continue;
        }
        if !is_domain_boundary(query, found.start(), found.end()) {
            continue;
        }

        candidates.push(candidate(
            "hostname",
            original,
            found.start(),
            found.end(),
            "<HOST>",
        ));
    }
}

fn collect_static_regex_candidates(
    query: &str,
    regex: &Regex,
    rule: &'static str,
    placeholder: &'static str,
    boundary: fn(&str, usize, usize) -> bool,
    candidates: &mut Vec<RedactionCandidate>,
) {
    for found in regex.find_iter(query) {
        if boundary(query, found.start(), found.end()) {
            candidates.push(candidate(
                rule,
                found.as_str(),
                found.start(),
                found.end(),
                placeholder,
            ));
        }
    }
}

fn collect_pattern_candidates(
    query: &str,
    patterns: &[String],
    kind: PatternKind,
    rule: &'static str,
    placeholder: &'static str,
    boundary: fn(&str, usize, usize) -> bool,
    candidates: &mut Vec<RedactionCandidate>,
) {
    for pattern in patterns {
        let pattern = pattern.trim();
        if pattern.is_empty() {
            continue;
        }

        let regex = wildcard_pattern_regex(pattern, kind);
        for found in regex.find_iter(query) {
            if boundary(query, found.start(), found.end()) {
                candidates.push(candidate(
                    rule,
                    found.as_str(),
                    found.start(),
                    found.end(),
                    placeholder,
                ));
            }
        }
    }
}

fn select_non_overlapping_candidates(
    mut candidates: Vec<RedactionCandidate>,
) -> Vec<RedactionCandidate> {
    candidates.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then(left.priority.cmp(&right.priority))
            .then((right.end - right.start).cmp(&(left.end - left.start)))
    });

    let mut selected: Vec<RedactionCandidate> = Vec::new();
    for candidate in candidates {
        if selected
            .last()
            .is_some_and(|previous| candidate.start < previous.end)
        {
            continue;
        }
        selected.push(candidate);
    }

    selected
}

fn candidate(
    rule: &'static str,
    original: &str,
    start: usize,
    end: usize,
    placeholder: &'static str,
) -> RedactionCandidate {
    RedactionCandidate {
        rule,
        original: original.to_owned(),
        start,
        end,
        placeholder,
        priority: rule_priority(rule),
    }
}

fn rule_priority(rule: &str) -> u8 {
    match rule {
        "email" => 0,
        "ipv6" => 1,
        "ipv4" => 2,
        "internal_domain" => 3,
        "hostname" => 4,
        "k8s_namespace" => 5,
        _ => u8::MAX,
    }
}

#[derive(Debug, Clone, Copy)]
enum PatternKind {
    Domain,
    Namespace,
}

fn wildcard_pattern_regex(pattern: &str, kind: PatternKind) -> Regex {
    let mut regex = String::new();
    match kind {
        PatternKind::Domain => regex.push_str("(?i)"),
        PatternKind::Namespace => {}
    }

    for ch in pattern.chars() {
        if ch == '*' {
            match kind {
                PatternKind::Domain => regex.push_str("[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*"),
                PatternKind::Namespace => regex.push_str("[A-Za-z0-9-]*"),
            }
        } else {
            regex.push_str(&regex::escape(&ch.to_string()));
        }
    }

    Regex::new(&regex).expect("wildcard conversion must produce valid regex")
}

fn is_domain_boundary(query: &str, start: usize, end: usize) -> bool {
    previous_char(query, start).is_none_or(|ch| !is_domain_char(ch) && ch != '@')
        && next_char(query, end).is_none_or(|ch| !is_domain_char(ch))
}

fn is_namespace_boundary(query: &str, start: usize, end: usize) -> bool {
    previous_char(query, start).is_none_or(|ch| !is_namespace_char(ch))
        && next_char(query, end).is_none_or(|ch| !is_namespace_char(ch))
}

fn is_ip_boundary(query: &str, start: usize, end: usize) -> bool {
    previous_char(query, start).is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '.')
        && next_char(query, end).is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '.')
}

fn is_ipv6_boundary(query: &str, start: usize, end: usize) -> bool {
    previous_char(query, start).is_none_or(|ch| !ch.is_ascii_hexdigit() && ch != ':')
        && next_char(query, end).is_none_or(|ch| !ch.is_ascii_hexdigit() && ch != ':')
}

fn is_email_boundary(query: &str, start: usize, end: usize) -> bool {
    previous_char(query, start).is_none_or(|ch| !is_email_char(ch))
        && next_char(query, end).is_none_or(|ch| !is_email_char(ch))
}

fn is_domain_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '.')
}

fn is_namespace_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')
}

fn is_email_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || matches!(
            ch,
            '.' | '!'
                | '#'
                | '$'
                | '%'
                | '&'
                | '\''
                | '*'
                | '+'
                | '/'
                | '='
                | '?'
                | '^'
                | '_'
                | '`'
                | '{'
                | '|'
                | '}'
                | '~'
                | '-'
                | '@'
        )
}

fn previous_char(input: &str, byte_index: usize) -> Option<char> {
    input[..byte_index].chars().next_back()
}

fn next_char(input: &str, byte_index: usize) -> Option<char> {
    input[byte_index..].chars().next()
}

fn hostname_regex() -> &'static Regex {
    static HOSTNAME: OnceLock<Regex> = OnceLock::new();
    HOSTNAME.get_or_init(|| {
        Regex::new(r"(?i)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}")
            .expect("hostname regex must compile")
    })
}

fn ipv4_regex() -> &'static Regex {
    static IPV4: OnceLock<Regex> = OnceLock::new();
    IPV4.get_or_init(|| {
        Regex::new(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
        )
        .expect("IPv4 regex must compile")
    })
}

fn ipv6_regex() -> &'static Regex {
    static IPV6: OnceLock<Regex> = OnceLock::new();
    IPV6.get_or_init(|| {
        Regex::new(
            r"(?xi)
            (?:
                (?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}
                | (?:[0-9a-f]{1,4}:){1,7}:
                | (?:[0-9a-f]{1,4}:){1,6}:[0-9a-f]{1,4}
                | (?:[0-9a-f]{1,4}:){1,5}(?::[0-9a-f]{1,4}){1,2}
                | (?:[0-9a-f]{1,4}:){1,4}(?::[0-9a-f]{1,4}){1,3}
                | (?:[0-9a-f]{1,4}:){1,3}(?::[0-9a-f]{1,4}){1,4}
                | (?:[0-9a-f]{1,4}:){1,2}(?::[0-9a-f]{1,4}){1,5}
                | [0-9a-f]{1,4}:(?:(?::[0-9a-f]{1,4}){1,6})
                | :(?:(?::[0-9a-f]{1,4}){1,7}|:)
            )",
        )
        .expect("IPv6 regex must compile")
    })
}

fn email_regex() -> &'static Regex {
    static EMAIL: OnceLock<Regex> = OnceLock::new();
    EMAIL.get_or_init(|| {
        Regex::new(
            r"(?i)[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+",
        )
        .expect("email regex must compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redact_with(rule: RedactionRule, query: &str) -> RedactionResult {
        QueryRedactor::with_rules(vec![rule], true).redact(query)
    }

    #[test]
    fn redacts_ipv4_address() {
        let result = redact_with(RedactionRule::IPv4, "restart nginx on 10.0.1.42");

        assert_eq!(result.redacted_query, "restart nginx on <IP>");
        assert_eq!(result.redactions_applied[0].rule, "ipv4");
        assert_eq!(result.redactions_applied[0].original, "10.0.1.42");
        assert_eq!(result.redactions_applied[0].position, 17);
    }

    #[test]
    fn redacts_full_ipv6_address() {
        let result = redact_with(
            RedactionRule::IPv6,
            "node 2001:0db8:85a3:0000:0000:8a2e:0370:7334 is slow",
        );

        assert_eq!(result.redacted_query, "node <IP> is slow");
        assert_eq!(
            result.redactions_applied[0].original,
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        );
    }

    #[test]
    fn redacts_abbreviated_ipv6_address() {
        let result = redact_with(RedactionRule::IPv6, "loopback ::1 failed");

        assert_eq!(result.redacted_query, "loopback <IP> failed");
        assert_eq!(result.redactions_applied[0].original, "::1");
    }

    #[test]
    fn redacts_hostname() {
        let result = redact_with(
            RedactionRule::Hostname,
            "check prod-api-03.internal.corp health",
        );

        assert_eq!(result.redacted_query, "check <HOST> health");
        assert_eq!(
            result.redactions_applied[0].original,
            "prod-api-03.internal.corp"
        );
    }

    #[test]
    fn redacts_email() {
        let result = redact_with(RedactionRule::Email, "user@internal.corp permissions");

        assert_eq!(result.redacted_query, "<EMAIL> permissions");
        assert_eq!(result.redactions_applied[0].rule, "email");
        assert_eq!(result.redactions_applied[0].original, "user@internal.corp");
    }

    #[test]
    fn redacts_internal_domain_patterns() {
        let result = redact_with(
            RedactionRule::InternalDomain(vec![
                "*.internal.corp".to_owned(),
                "*.prod.*".to_owned(),
            ]),
            "compare api.internal.corp and edge.prod.eu",
        );

        assert_eq!(result.redacted_query, "compare <DOMAIN> and <DOMAIN>");
        assert_eq!(result.redactions_applied.len(), 2);
    }

    #[test]
    fn redacts_k8s_namespace_patterns() {
        let result = redact_with(
            RedactionRule::K8sNamespace(vec!["prod-*".to_owned(), "staging-*".to_owned()]),
            "restart statefulset in prod-us-west-2 and staging-payments",
        );

        assert_eq!(
            result.redacted_query,
            "restart statefulset in <NAMESPACE> and <NAMESPACE>"
        );
        assert_eq!(result.redactions_applied[0].rule, "k8s_namespace");
        assert_eq!(result.redactions_applied[0].original, "prod-us-west-2");
    }

    #[test]
    fn redacts_multiple_values_in_single_query() {
        let redactor = QueryRedactor::with_rules(
            vec![
                RedactionRule::Email,
                RedactionRule::IPv4,
                RedactionRule::Hostname,
            ],
            true,
        );

        let result = redactor.redact("ask user@internal.corp about 10.0.1.42 on api.prod.local");

        assert_eq!(result.redacted_query, "ask <EMAIL> about <IP> on <HOST>");
        assert_eq!(result.redactions_applied.len(), 3);
    }

    #[test]
    fn disabled_redactor_returns_original_query() {
        let result = QueryRedactor::new(false).redact("10.0.1.42 user@internal.corp");

        assert_eq!(result.redacted_query, "10.0.1.42 user@internal.corp");
        assert!(result.redactions_applied.is_empty());
    }

    #[test]
    fn empty_query_has_no_redactions() {
        let result = QueryRedactor::default().redact("");

        assert_eq!(result.redacted_query, "");
        assert!(result.redactions_applied.is_empty());
    }

    #[test]
    fn redacts_chinese_text_with_embedded_ips_and_hostnames() {
        let result =
            QueryRedactor::with_rules(vec![RedactionRule::IPv4, RedactionRule::Hostname], true)
                .redact("10.0.1.42 上的 prod-api-03.internal.corp 怎么重启");

        assert_eq!(result.redacted_query, "<IP> 上的 <HOST> 怎么重启");
        assert_eq!(result.redactions_applied.len(), 2);
    }

    #[test]
    fn audit_trail_is_populated_for_each_redaction() {
        let result =
            QueryRedactor::with_rules(vec![RedactionRule::Email, RedactionRule::IPv4], true)
                .redact("mail user@internal.corp from 10.0.1.42");

        assert_eq!(
            result.redactions_applied,
            vec![
                RedactionApplied {
                    rule: "email".to_owned(),
                    original: "user@internal.corp".to_owned(),
                    position: 5,
                },
                RedactionApplied {
                    rule: "ipv4".to_owned(),
                    original: "10.0.1.42".to_owned(),
                    position: 29,
                },
            ]
        );

        let audit_json = serde_json::to_string(&result).expect("audit result serializes");
        assert!(audit_json.contains("user@internal.corp"));
        assert!(audit_json.contains("10.0.1.42"));
    }

    #[test]
    fn email_takes_priority_over_domain_and_hostname_matches() {
        let redactor = QueryRedactor::with_rules(
            vec![
                RedactionRule::Hostname,
                RedactionRule::InternalDomain(vec!["*.corp".to_owned()]),
                RedactionRule::Email,
            ],
            true,
        );

        let result = redactor.redact("owner user@internal.corp");

        assert_eq!(result.redacted_query, "owner <EMAIL>");
        assert_eq!(result.redactions_applied.len(), 1);
        assert_eq!(result.redactions_applied[0].rule, "email");
    }

    #[test]
    fn from_config_rejects_unknown_rules() {
        let error = QueryRedactor::from_config(true, &[String::from("token")], vec![], vec![])
            .expect_err("unknown rule should fail");

        assert_eq!(error, RedactionError::UnknownRule("token".to_owned()));
    }

    #[test]
    fn from_config_applies_custom_patterns() {
        let redactor = QueryRedactor::from_config(
            true,
            &[
                "internal_domain".to_string(),
                "k8s_namespace".to_string(),
                "ipv4".to_string(),
            ],
            vec!["*.internal.corp".to_string()],
            vec!["prod-*".to_string()],
        )
        .expect("config should build redactor");

        let result = redactor.redact("10.0.1.42 api.internal.corp prod-us-west-2");

        assert_eq!(result.redacted_query, "<IP> <DOMAIN> <NAMESPACE>");
        assert_eq!(result.redactions_applied.len(), 3);
    }
}

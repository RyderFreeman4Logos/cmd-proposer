//! `CommandProposal` schema and display formatting for human review.

use serde::{Deserialize, Serialize};
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Risk {
    Low,
    Medium,
    High,
    Critical,
}

impl Risk {
    #[must_use]
    pub const fn color(self) -> &'static str {
        match self {
            Self::Low => "\x1b[32m",
            Self::Medium => "\x1b[33m",
            Self::High => "\x1b[31m",
            Self::Critical => "\x1b[91m",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Low => "● LOW",
            Self::Medium => "⚠ MEDIUM",
            Self::High => "◆ HIGH",
            Self::Critical => "✖ CRITICAL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    OtherWeb,
    OfficialWebDoc,
    LocalDoc,
    LocalSchema,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvidenceSource {
    pub kind: EvidenceKind,
    pub doc: String,
    pub lines: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Evidence {
    pub claim: String,
    pub source: EvidenceSource,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PreflightCmd {
    pub argv: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollbackStatus {
    Available,
    Partial,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RollbackInfo {
    pub available: RollbackStatus,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CommandProposal {
    pub summary: String,
    pub argv: Vec<String>,
    pub display: String,
    pub risk: Risk,
    pub risk_reasons: Vec<String>,
    pub assumptions: Vec<String>,
    pub preflight: Vec<PreflightCmd>,
    pub rollback: Option<RollbackInfo>,
    pub evidence: Vec<Evidence>,
    pub missing_confirmations: Vec<String>,
}

impl CommandProposal {
    #[must_use]
    pub fn display_command(&self) -> String {
        display_argv(&self.argv)
    }

    #[must_use]
    pub fn format_summary(&self) -> String {
        let mut output = String::new();
        push_line(&mut output, "Summary", &self.summary);
        push_line(&mut output, "Risk", self.risk.label());

        let command = self.display_command();
        push_line(
            &mut output,
            "Command",
            if command.is_empty() {
                "<empty>"
            } else {
                command.as_str()
            },
        );

        push_list(&mut output, "Risk reasons", &self.risk_reasons);
        push_list(&mut output, "Assumptions", &self.assumptions);
        push_preflight(&mut output, &self.preflight);
        push_rollback(&mut output, self.rollback.as_ref());
        push_evidence(&mut output, &self.evidence);
        push_list(
            &mut output,
            "Missing confirmations",
            &self.missing_confirmations,
        );

        output.trim_end().to_owned()
    }
}

#[must_use]
fn display_argv(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_escape(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

#[must_use]
fn shell_escape(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_owned();
    }

    if arg.bytes().all(is_shell_safe) {
        return arg.to_owned();
    }

    let mut quoted = String::with_capacity(arg.len() + 2);
    quoted.push('\'');
    for ch in arg.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

const fn is_shell_safe(byte: u8) -> bool {
    matches!(
        byte,
        b'a'..=b'z'
            | b'A'..=b'Z'
            | b'0'..=b'9'
            | b'_'
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b','
            | b'+'
            | b'='
            | b'@'
            | b'%'
    )
}

fn push_line(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(output, "{label}: {value}");
}

fn push_list(output: &mut String, label: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }

    let _ = writeln!(output, "{label}:");
    for item in items {
        let _ = writeln!(output, "- {item}");
    }
}

fn push_preflight(output: &mut String, preflight: &[PreflightCmd]) {
    if preflight.is_empty() {
        return;
    }

    let _ = writeln!(output, "Preflight:");
    for cmd in preflight {
        let command = display_argv(&cmd.argv);
        let _ = writeln!(output, "- {command} ({})", cmd.reason);
    }
}

fn push_rollback(output: &mut String, rollback: Option<&RollbackInfo>) {
    let Some(rollback) = rollback else {
        return;
    };

    let _ = writeln!(
        output,
        "Rollback: {:?} - {}",
        rollback.available, rollback.notes
    );
}

fn push_evidence(output: &mut String, evidence: &[Evidence]) {
    if evidence.is_empty() {
        return;
    }

    let _ = writeln!(output, "Evidence:");
    for item in evidence {
        let line_suffix = item
            .source
            .lines
            .map_or_else(String::new, |(start, end)| format!(":{start}-{end}"));
        let _ = writeln!(
            output,
            "- {} ({:?}, {:?}, {}{})",
            item.claim, item.confidence, item.source.kind, item.source.doc, line_suffix
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn risk_ordering_matches_approval_flow() {
        assert!(Risk::Low < Risk::Medium);
        assert!(Risk::Medium < Risk::High);
        assert!(Risk::High < Risk::Critical);
    }

    #[test]
    fn command_proposal_roundtrips_through_json() {
        let proposal = sample_proposal();

        let value = serde_json::to_value(&proposal).expect("serialize proposal");
        let decoded: CommandProposal =
            serde_json::from_value(value.clone()).expect("deserialize proposal");

        assert_eq!(decoded, proposal);
        assert_eq!(value["risk"], json!("high"));
        assert_eq!(
            value["evidence"][0]["source"]["kind"],
            json!("local_schema")
        );
        assert_eq!(value["evidence"][1]["source"]["kind"], json!("other_web"));
        assert_eq!(value["evidence"][0]["confidence"], json!("high"));
        assert_eq!(value["rollback"]["available"], json!("partial"));
        assert_eq!(value["evidence"][0]["source"]["lines"], json!([10, 20]));
    }

    #[test]
    fn display_command_escapes_shell_special_arguments() {
        let proposal = CommandProposal {
            argv: vec![
                "kubectl".to_owned(),
                "-n".to_owned(),
                "payments prod".to_owned(),
                "rollout".to_owned(),
                "restart".to_owned(),
                "deployment/api".to_owned(),
                "--field-selector=metadata.name=api pod".to_owned(),
                "contains'quote".to_owned(),
                "$HOME".to_owned(),
                "semi;colon".to_owned(),
                String::new(),
            ],
            ..sample_proposal()
        };

        assert_eq!(
            proposal.display_command(),
            "kubectl -n 'payments prod' rollout restart deployment/api '--field-selector=metadata.name=api pod' 'contains'\\''quote' '$HOME' 'semi;colon' ''"
        );
    }

    #[test]
    fn evidence_kind_ordering_matches_source_priority() {
        assert!(EvidenceKind::LocalSchema > EvidenceKind::OtherWeb);
        assert!(EvidenceKind::LocalDoc > EvidenceKind::OfficialWebDoc);
        assert!(EvidenceKind::OfficialWebDoc > EvidenceKind::OtherWeb);
    }

    #[test]
    fn empty_argv_displays_empty_command() {
        let proposal = CommandProposal {
            argv: Vec::new(),
            display: String::new(),
            ..sample_proposal()
        };

        assert_eq!(proposal.display_command(), "");
        assert!(proposal.format_summary().contains("Command: <empty>"));
    }

    #[test]
    fn risk_label_and_color_match_spec_display() {
        assert_eq!(Risk::Low.color(), "\x1b[32m");
        assert_eq!(Risk::Medium.color(), "\x1b[33m");
        assert_eq!(Risk::High.color(), "\x1b[31m");
        assert_eq!(Risk::Critical.color(), "\x1b[91m");
        assert_eq!(Risk::Low.label(), "● LOW");
        assert_eq!(Risk::Medium.label(), "⚠ MEDIUM");
        assert_eq!(Risk::High.label(), "◆ HIGH");
        assert_eq!(Risk::Critical.label(), "✖ CRITICAL");
    }

    fn sample_proposal() -> CommandProposal {
        CommandProposal {
            summary: "重启 payments 命名空间中的 api deployment".to_owned(),
            argv: vec![
                "kubectl".to_owned(),
                "-n".to_owned(),
                "payments".to_owned(),
                "rollout".to_owned(),
                "restart".to_owned(),
                "deployment/api".to_owned(),
            ],
            display: "kubectl -n payments rollout restart deployment/api".to_owned(),
            risk: Risk::High,
            risk_reasons: vec!["mutates a production workload".to_owned()],
            assumptions: vec!["api is a deployment".to_owned()],
            preflight: vec![PreflightCmd {
                argv: vec![
                    "kubectl".to_owned(),
                    "-n".to_owned(),
                    "payments".to_owned(),
                    "get".to_owned(),
                    "deployment/api".to_owned(),
                ],
                reason: "confirm the workload exists".to_owned(),
            }],
            rollback: Some(RollbackInfo {
                available: RollbackStatus::Partial,
                notes: "rollout undo may depend on deployment history".to_owned(),
            }),
            evidence: vec![
                Evidence {
                    claim: "deployment/api is a scoped workload target".to_owned(),
                    source: EvidenceSource {
                        kind: EvidenceKind::LocalSchema,
                        doc: "kubectl-schema".to_owned(),
                        lines: Some((10, 20)),
                    },
                    confidence: Confidence::High,
                },
                Evidence {
                    claim: "rollout restart restarts a resource".to_owned(),
                    source: EvidenceSource {
                        kind: EvidenceKind::OtherWeb,
                        doc: "web-search-result".to_owned(),
                        lines: None,
                    },
                    confidence: Confidence::Low,
                },
            ],
            missing_confirmations: vec!["confirm production window".to_owned()],
        }
    }
}

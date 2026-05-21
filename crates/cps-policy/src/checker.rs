//! Proposal checker: full risk grading and evidence validation pipeline.
//!
//! Wraps [`PolicyGate`] and adds deeper evidence quality checks, safer
//! alternative suggestions, assumption validation, and risk adjustment.

use cps_proposal::{CommandProposal, Confidence, EvidenceKind, Risk};

use crate::{DocLookup, PolicyFinding, PolicyFindingCode, PolicyFindingSeverity, PolicyGate};

/// Known dry-run / preflight flag equivalents.
const DRY_RUN_FLAGS: &[&str] = &[
    "--dry-run",
    "--dry-run=client",
    "--dry-run=server",
    "diff",
    "plan",
    "--diff",
    "--check",
    "--what-if",
    "--noop",
    "--no-op",
];

/// Programs that typically accept a dry-run style flag.
const DRY_RUN_PROGRAMS: &[&str] = &[
    "kubectl",
    "terraform",
    "ansible-playbook",
    "helm",
    "rsync",
    "make",
];

/// K8s-style programs that require explicit namespace/context.
const K8S_PROGRAMS: &[&str] = &["kubectl", "helm", "oc"];

/// Cloud CLIs that require explicit region.
const CLOUD_PROGRAMS: &[&str] = &["aws", "gcloud", "az"];

/// Proposal checker with full risk grading and evidence validation.
///
/// Extends [`PolicyGate::check_proposal`] with deeper analysis:
///
/// - Evidence quality: warns on web-only evidence, validates doc_id references.
/// - Risk adjustment: escalates for missing scope or weak evidence, de-escalates
///   for dry-run flags or strong local evidence.
/// - Safer alternative suggestions.
/// - Assumption validation for k8s/cloud commands.
#[derive(Debug, Clone)]
pub struct ProposalChecker {
    policy: PolicyGate,
}

/// Final verdict produced by [`ProposalChecker::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalVerdict {
    /// Final adjusted risk level.
    pub risk: Risk,
    /// All policy findings (reject / warn / escalate).
    pub findings: Vec<PolicyFinding>,
    /// Non-blocking informational warnings.
    pub warnings: Vec<String>,
    /// Actionable suggestions (e.g., "consider adding --dry-run").
    pub suggestions: Vec<String>,
    /// `false` if any finding has `Reject` severity.
    pub approved: bool,
}

impl ProposalChecker {
    /// Create a new checker wrapping an existing [`PolicyGate`].
    #[must_use]
    pub fn new(policy: PolicyGate) -> Self {
        Self { policy }
    }

    /// Access the underlying policy gate.
    #[must_use]
    pub fn policy(&self) -> &PolicyGate {
        &self.policy
    }

    /// Run the full validation pipeline on a proposal.
    ///
    /// 1. Runs all existing [`PolicyGate::check_proposal`] checks.
    /// 2. Validates evidence quality and doc_id references.
    /// 3. Checks for safer alternatives and missing preflight.
    /// 4. Validates assumptions (namespace/context/region).
    /// 5. Computes the final adjusted risk level.
    pub fn check<D: DocLookup + ?Sized>(
        &self,
        proposal: &CommandProposal,
        doc_store: &D,
    ) -> ProposalVerdict {
        // Step 1: baseline findings from PolicyGate
        let mut findings = self.policy.check_proposal(proposal);
        let mut warnings: Vec<String> = Vec::new();
        let mut suggestions: Vec<String> = Vec::new();

        // Step 2: evidence quality checks
        self.check_evidence_quality(proposal, doc_store, &mut findings, &mut warnings);

        // Step 3: safer alternatives
        self.suggest_alternatives(proposal, &mut suggestions);

        // Step 4: assumption validation
        self.validate_assumptions(proposal, &mut warnings);

        // Step 5: risk adjustment
        let baseline = self.policy.classify_risk(&proposal.argv);
        let adjusted = self.adjust_risk(baseline, proposal, &findings, &warnings);
        let final_risk = baseline.max(adjusted);

        let approved = !findings
            .iter()
            .any(|f| f.severity == PolicyFindingSeverity::Reject);

        ProposalVerdict {
            risk: final_risk,
            findings,
            warnings,
            suggestions,
            approved,
        }
    }

    /// Check evidence quality beyond what PolicyGate already does.
    fn check_evidence_quality<D: DocLookup + ?Sized>(
        &self,
        proposal: &CommandProposal,
        doc_store: &D,
        findings: &mut Vec<PolicyFinding>,
        warnings: &mut Vec<String>,
    ) {
        if proposal.evidence.is_empty() {
            // PolicyGate already warns about missing evidence.
            return;
        }

        // (a) Web-only evidence warning
        let has_local = proposal.evidence.iter().any(|e| {
            matches!(
                e.source.kind,
                EvidenceKind::LocalDoc | EvidenceKind::LocalSchema
            )
        });
        if !has_local {
            warnings.push("no local doc evidence".to_owned());
        }

        // (b) Validate doc_id references
        for evidence in &proposal.evidence {
            if matches!(
                evidence.source.kind,
                EvidenceKind::LocalDoc | EvidenceKind::LocalSchema
            ) && !doc_store.contains_doc(&evidence.source.doc)
            {
                findings.push(PolicyFinding::warn(
                    PolicyFindingCode::MissingEvidence,
                    format!(
                        "evidence references unknown doc_id: {}",
                        evidence.source.doc
                    ),
                ));
            }
        }

        // (c) Missing evidence for high-risk commands
        let risk = self.policy.classify_risk(&proposal.argv);
        if risk >= Risk::High {
            let has_high_confidence = proposal
                .evidence
                .iter()
                .any(|e| e.confidence == Confidence::High);
            if !has_high_confidence {
                findings.push(PolicyFinding::escalation(
                    "high-risk command lacks high-confidence evidence",
                    risk,
                ));
            }
        }
    }

    /// Suggest safer alternatives if applicable.
    fn suggest_alternatives(
        &self,
        proposal: &CommandProposal,
        suggestions: &mut Vec<String>,
    ) {
        let risk = self.policy.classify_risk(&proposal.argv);

        // (a) Suggest dry-run if the program supports it and it's not already present
        if risk >= Risk::Medium && has_dry_run_support(proposal) && !has_dry_run_flag(proposal) {
            suggestions.push("consider adding --dry-run".to_owned());
        }

        // (b) Missing preflight for medium+ risk (PolicyGate already warns, but we
        //     add a concrete suggestion)
        if risk >= Risk::Medium && proposal.preflight.is_empty() {
            let program = proposal.argv.first().map(String::as_str).unwrap_or("");
            if is_k8s_program(program) {
                suggestions.push(format!(
                    "consider adding a preflight check, e.g.: {} get <resource>",
                    program
                ));
            } else {
                suggestions.push(
                    "consider adding a preflight command to verify state before mutation"
                        .to_owned(),
                );
            }
        }
    }

    /// Validate implicit assumptions in proposals for k8s/cloud commands.
    fn validate_assumptions(
        &self,
        proposal: &CommandProposal,
        warnings: &mut Vec<String>,
    ) {
        let program = match proposal.argv.first() {
            Some(p) => p.as_str(),
            None => return,
        };

        // (a) Missing namespace/context for k8s commands
        if is_k8s_program(program) {
            let has_namespace = proposal
                .argv
                .iter()
                .any(|a| a == "-n" || a == "--namespace" || a.starts_with("--namespace="));
            let has_context = proposal
                .argv
                .iter()
                .any(|a| a == "--context" || a.starts_with("--context="));

            if !has_namespace && has_namespaced_verb(&proposal.argv) {
                if !proposal.assumptions.iter().any(|a| {
                    let lower = a.to_ascii_lowercase();
                    lower.contains("namespace") || lower.contains("default namespace")
                }) {
                    warnings.push(format!(
                        "{} command uses implicit default namespace",
                        program
                    ));
                }
            }

            if !has_context
                && !proposal.assumptions.iter().any(|a| {
                    let lower = a.to_ascii_lowercase();
                    lower.contains("context") || lower.contains("cluster")
                })
            {
                warnings.push(format!(
                    "{} command uses implicit current context",
                    program
                ));
            }
        }

        // (b) Missing region for cloud CLIs
        if is_cloud_program(program) {
            let has_region = proposal.argv.iter().any(|a| {
                a == "--region"
                    || a.starts_with("--region=")
                    || a == "--location"
                    || a.starts_with("--location=")
            });
            if !has_region
                && !proposal.assumptions.iter().any(|a| {
                    let lower = a.to_ascii_lowercase();
                    lower.contains("region") || lower.contains("location")
                })
            {
                warnings.push(format!(
                    "{} command uses implicit default region",
                    program
                ));
            }
        }
    }

    /// Compute risk adjustment based on evidence, flags, and scope.
    fn adjust_risk(
        &self,
        baseline: Risk,
        proposal: &CommandProposal,
        findings: &[PolicyFinding],
        warnings: &[String],
    ) -> Risk {
        let mut level = baseline;

        // --- Escalation factors ---

        // High-risk tokens with missing scope
        let has_missing_scope = findings
            .iter()
            .any(|f| f.code == PolicyFindingCode::MissingScope);
        if has_missing_scope && level < Risk::High {
            level = escalate(level);
        }

        // Web-only evidence
        let web_only = warnings.iter().any(|w| w == "no local doc evidence");
        if web_only && level >= Risk::Medium {
            level = escalate(level);
        }

        // Missing evidence entirely
        let no_evidence = findings
            .iter()
            .any(|f| f.code == PolicyFindingCode::MissingEvidence && f.message == "proposal has no evidence");
        if no_evidence && level >= Risk::Medium {
            level = escalate(level);
        }

        // --- De-escalation factors ---

        // Dry-run flag present
        if has_dry_run_flag(proposal) && level > Risk::Low {
            level = de_escalate(level);
        }

        // Preflight provided for medium+ baseline
        if !proposal.preflight.is_empty() && level > Risk::Low {
            // Only de-escalate if there's also evidence
            if !proposal.evidence.is_empty() {
                level = de_escalate(level);
            }
        }

        // Strong local evidence (LocalSchema with High confidence)
        let has_strong_local = proposal.evidence.iter().any(|e| {
            e.source.kind == EvidenceKind::LocalSchema && e.confidence == Confidence::High
        });
        if has_strong_local && level > Risk::Low {
            level = de_escalate(level);
        }

        level
    }
}

/// Escalate risk by one level, capped at Critical.
fn escalate(risk: Risk) -> Risk {
    match risk {
        Risk::Low => Risk::Medium,
        Risk::Medium => Risk::High,
        Risk::High => Risk::Critical,
        Risk::Critical => Risk::Critical,
    }
}

/// De-escalate risk by one level, floored at Low.
fn de_escalate(risk: Risk) -> Risk {
    match risk {
        Risk::Low => Risk::Low,
        Risk::Medium => Risk::Low,
        Risk::High => Risk::Medium,
        Risk::Critical => Risk::High,
    }
}

/// Check if the proposal's program supports dry-run style flags.
fn has_dry_run_support(proposal: &CommandProposal) -> bool {
    proposal
        .argv
        .first()
        .map(|p| DRY_RUN_PROGRAMS.contains(&p.as_str()))
        .unwrap_or(false)
}

/// Check if the proposal already has a dry-run flag in argv or preflight.
fn has_dry_run_flag(proposal: &CommandProposal) -> bool {
    let in_argv = proposal
        .argv
        .iter()
        .any(|a| DRY_RUN_FLAGS.contains(&a.as_str()));

    let in_preflight = proposal.preflight.iter().any(|cmd| {
        cmd.argv
            .iter()
            .any(|a| DRY_RUN_FLAGS.contains(&a.as_str()))
    });

    in_argv || in_preflight
}

fn is_k8s_program(program: &str) -> bool {
    K8S_PROGRAMS.contains(&program)
}

fn is_cloud_program(program: &str) -> bool {
    CLOUD_PROGRAMS.contains(&program)
}

/// Check if the argv contains a verb that operates on namespaced resources.
fn has_namespaced_verb(argv: &[String]) -> bool {
    argv.iter().skip(1).any(|arg| {
        matches!(
            arg.as_str(),
            "get"
                | "describe"
                | "logs"
                | "delete"
                | "apply"
                | "rollout"
                | "scale"
                | "patch"
                | "create"
                | "edit"
                | "label"
                | "annotate"
                | "exec"
                | "port-forward"
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use cps_proposal::{
        Confidence, Evidence, EvidenceKind, EvidenceSource, PreflightCmd, RollbackInfo,
        RollbackStatus,
    };

    use super::*;
    use crate::PolicyConfig;

    // ---- Helpers ----

    fn checker() -> ProposalChecker {
        ProposalChecker::new(PolicyGate::new(config()))
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

        PolicyConfig {
            allow_programs: vec![
                "kubectl".to_owned(),
                "aws".to_owned(),
                "terraform".to_owned(),
                "helm".to_owned(),
                "rm".to_owned(),
            ],
            allowed_roles: vec!["doc_explorer".to_owned()],
            role_tool_permissions: permissions,
            search_enabled: true,
            execute_enabled: true,
            max_subagent_context: 4096,
        }
    }

    fn argv<const N: usize>(items: [&str; N]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn base_proposal(argv: Vec<String>, risk: Risk) -> CommandProposal {
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
                notes: "test".to_owned(),
            }),
            evidence: vec![Evidence {
                claim: "kubectl get/delete/rollout/restart flags documented".to_owned(),
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

    fn web_only_evidence() -> Vec<Evidence> {
        vec![Evidence {
            claim: "found via web search".to_owned(),
            source: EvidenceSource {
                kind: EvidenceKind::OtherWeb,
                doc: "web-result-1".to_owned(),
                lines: None,
            },
            confidence: Confidence::Low,
        }]
    }

    fn strong_local_evidence() -> Vec<Evidence> {
        vec![Evidence {
            claim: "kubectl delete flags documented in schema".to_owned(),
            source: EvidenceSource {
                kind: EvidenceKind::LocalSchema,
                doc: "kubectl-schema".to_owned(),
                lines: Some((1, 50)),
            },
            confidence: Confidence::High,
        }]
    }

    fn doc_store_with(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    // ---- Tests ----

    #[test]
    fn web_only_evidence_emits_warning() {
        let checker = checker();
        let mut proposal = base_proposal(argv(["kubectl", "get", "pods"]), Risk::Low);
        proposal.evidence = web_only_evidence();

        let verdict = checker.check(&proposal, &doc_store_with(&[]));

        assert!(
            verdict.warnings.iter().any(|w| w == "no local doc evidence"),
            "expected 'no local doc evidence' warning, got: {:?}",
            verdict.warnings
        );
    }

    #[test]
    fn missing_namespace_escalates_risk() {
        let checker = checker();
        // kubectl get pods -- no namespace, baseline Low
        let proposal = base_proposal(argv(["kubectl", "get", "pods"]), Risk::Low);
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        // MissingScope finding should be present (from PolicyGate)
        assert!(
            verdict
                .findings
                .iter()
                .any(|f| f.code == PolicyFindingCode::MissingScope),
            "expected MissingScope finding"
        );
        // Risk should be escalated from Low
        assert!(
            verdict.risk >= Risk::Medium,
            "expected risk >= Medium due to missing namespace, got {:?}",
            verdict.risk
        );
    }

    #[test]
    fn dry_run_flag_de_escalates_risk() {
        let checker = checker();
        let mut proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "delete", "pod", "api", "--dry-run=server"]),
            Risk::High,
        );
        proposal.evidence = strong_local_evidence();
        let docs = doc_store_with(&["kubectl-schema"]);

        let verdict = checker.check(&proposal, &docs);

        // Baseline is High (delete verb), but --dry-run should de-escalate
        assert!(
            verdict.risk < Risk::Critical,
            "expected risk below Critical with --dry-run, got {:?}",
            verdict.risk
        );
    }

    #[test]
    fn high_risk_without_preflight_gets_suggestion() {
        let checker = checker();
        let mut proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "delete", "pod", "api"]),
            Risk::High,
        );
        proposal.preflight = Vec::new(); // no preflight
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            verdict
                .suggestions
                .iter()
                .any(|s| s.contains("preflight")),
            "expected preflight suggestion, got: {:?}",
            verdict.suggestions
        );
    }

    #[test]
    fn full_pipeline_produces_correct_verdict() {
        let checker = checker();
        let mut proposal = base_proposal(
            argv(["kubectl", "-n", "payments", "rollout", "restart", "deployment/api"]),
            Risk::Medium,
        );
        proposal.evidence = vec![Evidence {
            claim: "rollout restart is a scoped mutation".to_owned(),
            source: EvidenceSource {
                kind: EvidenceKind::LocalDoc,
                doc: "kubectl-help".to_owned(),
                lines: Some((10, 30)),
            },
            confidence: Confidence::High,
        }];
        proposal.preflight = vec![PreflightCmd {
            argv: argv(["kubectl", "-n", "payments", "get", "deployment/api"]),
            reason: "confirm workload exists".to_owned(),
        }];
        proposal.assumptions = vec!["current context is production cluster".to_owned()];
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        // Should be approved (no Reject findings)
        assert!(verdict.approved, "expected approved verdict");
        // Risk should be reasonable (Medium baseline, with preflight + evidence -> de-escalated)
        assert!(
            verdict.risk <= Risk::Medium,
            "expected risk <= Medium, got {:?}",
            verdict.risk
        );
        // No "no local doc evidence" warning (has local doc)
        assert!(
            !verdict.warnings.iter().any(|w| w == "no local doc evidence"),
            "should not warn about web-only evidence"
        );
    }

    #[test]
    fn invalid_doc_id_in_evidence_produces_finding() {
        let checker = checker();
        let mut proposal = base_proposal(argv(["kubectl", "get", "pods"]), Risk::Low);
        proposal.evidence = vec![Evidence {
            claim: "from local doc".to_owned(),
            source: EvidenceSource {
                kind: EvidenceKind::LocalDoc,
                doc: "nonexistent-doc".to_owned(),
                lines: None,
            },
            confidence: Confidence::Medium,
        }];
        let docs = doc_store_with(&[]); // doc store has nothing

        let verdict = checker.check(&proposal, &docs);

        assert!(
            verdict.findings.iter().any(|f| {
                f.code == PolicyFindingCode::MissingEvidence
                    && f.message.contains("nonexistent-doc")
            }),
            "expected finding about unknown doc_id"
        );
    }

    #[test]
    fn dry_run_suggestion_for_medium_risk_with_dry_run_support() {
        let checker = checker();
        let proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "rollout", "restart", "deployment/api"]),
            Risk::Medium,
        );
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            verdict
                .suggestions
                .iter()
                .any(|s| s.contains("--dry-run")),
            "expected --dry-run suggestion for medium-risk kubectl command, got: {:?}",
            verdict.suggestions
        );
    }

    #[test]
    fn reject_finding_means_not_approved() {
        let checker = checker();
        let proposal = base_proposal(Vec::new(), Risk::Low);
        let docs = doc_store_with(&[]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            !verdict.approved,
            "empty argv should produce Reject finding -> not approved"
        );
    }

    #[test]
    fn cloud_command_missing_region_warns() {
        let checker = checker();
        let proposal = base_proposal(
            argv(["aws", "s3", "ls"]),
            Risk::Low,
        );
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            verdict
                .warnings
                .iter()
                .any(|w| w.contains("implicit default region")),
            "expected implicit region warning for aws, got: {:?}",
            verdict.warnings
        );
    }

    #[test]
    fn strong_local_evidence_de_escalates() {
        let checker = checker();
        let mut proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "--context=staging", "delete", "pod", "api"]),
            Risk::High,
        );
        proposal.evidence = strong_local_evidence();
        proposal.preflight = vec![PreflightCmd {
            argv: argv(["kubectl", "-n", "prod", "get", "pod", "api"]),
            reason: "confirm target exists".to_owned(),
        }];
        let docs = doc_store_with(&["kubectl-schema"]);

        let verdict = checker.check(&proposal, &docs);

        // Strong local evidence + preflight + explicit scope should keep risk moderate
        assert!(
            verdict.risk <= Risk::High,
            "strong evidence + preflight should not escalate beyond High, got {:?}",
            verdict.risk
        );
    }

    #[test]
    fn escalate_and_de_escalate_helpers() {
        assert_eq!(escalate(Risk::Low), Risk::Medium);
        assert_eq!(escalate(Risk::Medium), Risk::High);
        assert_eq!(escalate(Risk::High), Risk::Critical);
        assert_eq!(escalate(Risk::Critical), Risk::Critical);

        assert_eq!(de_escalate(Risk::Low), Risk::Low);
        assert_eq!(de_escalate(Risk::Medium), Risk::Low);
        assert_eq!(de_escalate(Risk::High), Risk::Medium);
        assert_eq!(de_escalate(Risk::Critical), Risk::High);
    }

    #[test]
    fn k8s_implicit_context_warning() {
        let checker = checker();
        let proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "get", "pods"]),
            Risk::Low,
        );
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            verdict
                .warnings
                .iter()
                .any(|w| w.contains("implicit current context")),
            "expected implicit context warning, got: {:?}",
            verdict.warnings
        );
    }

    #[test]
    fn assumption_about_context_suppresses_warning() {
        let checker = checker();
        let mut proposal = base_proposal(
            argv(["kubectl", "-n", "prod", "get", "pods"]),
            Risk::Low,
        );
        proposal.assumptions = vec!["current context is the staging cluster".to_owned()];
        let docs = doc_store_with(&["kubectl-help"]);

        let verdict = checker.check(&proposal, &docs);

        assert!(
            !verdict
                .warnings
                .iter()
                .any(|w| w.contains("implicit current context")),
            "context warning should be suppressed when assumption mentions context"
        );
    }
}

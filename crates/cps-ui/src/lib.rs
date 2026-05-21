//! ratatui three-zone UI: key bindings and progress display.
//!
//! Three zones:
//! - Zone 1 (RISK): Color-coded risk bar at the top.
//! - Zone 2 (COMMAND): Monospace command display with action hints.
//! - Zone 3 (DETAILS): Collapsible sections (risk reasons, evidence, preflight, etc.).

use cps_policy::ProposalVerdict;
use cps_proposal::{CommandProposal, Risk};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

// ---------------------------------------------------------------------------
// Risk color mapping
// ---------------------------------------------------------------------------

/// Map a [`Risk`] level to a ratatui [`Color`].
///
/// - Low -> Green
/// - Medium -> Yellow
/// - High -> Red
/// - Critical -> bold Red (LightRed for terminal compatibility)
#[must_use]
pub fn risk_color(risk: &Risk) -> Color {
    match risk {
        Risk::Low => Color::Green,
        Risk::Medium => Color::Yellow,
        Risk::High => Color::Red,
        Risk::Critical => Color::LightRed,
    }
}

// ---------------------------------------------------------------------------
// Collapsible detail sections
// ---------------------------------------------------------------------------

/// Identifiers for each collapsible section in Zone 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailSection {
    RiskReasons,
    Assumptions,
    Evidence,
    Preflight,
    Rollback,
    MissingConfirmations,
    Warnings,
    Suggestions,
}

/// All sections in display order.
pub const ALL_SECTIONS: [DetailSection; 8] = [
    DetailSection::RiskReasons,
    DetailSection::Assumptions,
    DetailSection::Evidence,
    DetailSection::Preflight,
    DetailSection::Rollback,
    DetailSection::MissingConfirmations,
    DetailSection::Warnings,
    DetailSection::Suggestions,
];

impl DetailSection {
    /// Human-readable label for this section.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::RiskReasons => "Risk Reasons",
            Self::Assumptions => "Assumptions",
            Self::Evidence => "Evidence",
            Self::Preflight => "Preflight",
            Self::Rollback => "Rollback",
            Self::MissingConfirmations => "Missing Confirmations",
            Self::Warnings => "Warnings",
            Self::Suggestions => "Suggestions",
        }
    }
}

// ---------------------------------------------------------------------------
// ProposalView
// ---------------------------------------------------------------------------

/// View state for a single proposal being reviewed by the human operator.
#[derive(Debug, Clone)]
pub struct ProposalView {
    pub proposal: CommandProposal,
    pub verdict: ProposalVerdict,
    /// Per-section expand state (indexed by position in [`ALL_SECTIONS`]).
    pub expanded: Vec<bool>,
    /// Currently selected (highlighted) section index.
    pub selected_section: usize,
}

impl ProposalView {
    /// Create a new view with risk-based default expansion:
    ///
    /// - Low: all collapsed
    /// - Medium: risk reasons section open
    /// - High / Critical: all sections open
    #[must_use]
    pub fn new(proposal: CommandProposal, verdict: ProposalVerdict) -> Self {
        let risk = verdict.risk;
        let expanded = default_expansion(&risk);
        Self {
            proposal,
            verdict,
            expanded,
            selected_section: 0,
        }
    }

    /// Toggle the expand state of the currently selected section.
    pub fn toggle_selected(&mut self) {
        if let Some(state) = self.expanded.get_mut(self.selected_section) {
            *state = !*state;
        }
    }

    /// Move selection up, wrapping at the top.
    pub fn select_prev(&mut self) {
        if self.selected_section == 0 {
            self.selected_section = ALL_SECTIONS.len().saturating_sub(1);
        } else {
            self.selected_section -= 1;
        }
    }

    /// Move selection down, wrapping at the bottom.
    pub fn select_next(&mut self) {
        self.selected_section = (self.selected_section + 1) % ALL_SECTIONS.len();
    }

    /// Whether Ctrl+Enter execution is allowed for this risk level.
    #[must_use]
    pub fn ctrl_enter_allowed(&self) -> bool {
        matches!(self.verdict.risk, Risk::Low | Risk::Medium)
    }
}

/// Compute the default expand state per the risk-based rules.
#[must_use]
fn default_expansion(risk: &Risk) -> Vec<bool> {
    match risk {
        Risk::Low => vec![false; ALL_SECTIONS.len()],
        Risk::Medium => ALL_SECTIONS
            .iter()
            .map(|s| *s == DetailSection::RiskReasons)
            .collect(),
        Risk::High | Risk::Critical => vec![true; ALL_SECTIONS.len()],
    }
}

// ---------------------------------------------------------------------------
// AppState / App
// ---------------------------------------------------------------------------

/// Top-level application state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Idle,
    Exploring {
        elapsed_secs: u64,
        active_agents: Vec<String>,
    },
    Proposing,
    Editing,
    Confirming {
        risk: Risk,
    },
}

/// Root UI state container.
#[derive(Debug, Clone)]
pub struct App {
    pub state: AppState,
    pub proposal_view: Option<ProposalView>,
    pub input: String,
    pub messages: Vec<String>,
}

impl App {
    /// Create a new app in the idle state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: AppState::Idle,
            proposal_view: None,
            input: String::new(),
            messages: Vec::new(),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Action / key handling
// ---------------------------------------------------------------------------

/// Semantic actions produced by key bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Ctrl+Enter — execute proposal (low/medium risk only).
    Execute,
    /// Tab — switch to edit mode.
    Edit,
    /// Esc — cancel current operation.
    Cancel,
    /// Number key 1-9 — run a specific preflight command.
    RunPreflight(u8),
    /// Up arrow — navigate up in the details zone.
    NavigateUp,
    /// Down arrow — navigate down in the details zone.
    NavigateDown,
    /// Enter — toggle expand on the selected section.
    ToggleExpand,
    /// Ctrl+C — abort the application.
    Abort,
    /// Character input.
    Input(char),
    /// Typed confirmation string for high/critical risk.
    Confirm(String),
    /// No action.
    Noop,
}

/// Map a crossterm [`KeyEvent`] to a semantic [`Action`].
#[must_use]
pub fn handle_key(event: KeyEvent) -> Action {
    // Ctrl+C always aborts.
    if event.modifiers.contains(KeyModifiers::CONTROL) && event.code == KeyCode::Char('c') {
        return Action::Abort;
    }

    // Ctrl+Enter -> Execute.
    if event.modifiers.contains(KeyModifiers::CONTROL) && event.code == KeyCode::Enter {
        return Action::Execute;
    }

    match event.code {
        KeyCode::Tab => Action::Edit,
        KeyCode::Esc => Action::Cancel,
        KeyCode::Up => Action::NavigateUp,
        KeyCode::Down => Action::NavigateDown,
        KeyCode::Enter => Action::ToggleExpand,
        KeyCode::Char(ch) if ch.is_ascii_digit() && ch != '0' => {
            Action::RunPreflight(ch as u8 - b'0')
        }
        KeyCode::Char(ch) => Action::Input(ch),
        _ => Action::Noop,
    }
}

// ---------------------------------------------------------------------------
// Rendering: Zone 1 — Risk bar
// ---------------------------------------------------------------------------

/// Render the risk bar as a ratatui [`Paragraph`] widget.
///
/// Shows the risk label in the appropriate color, with bold styling
/// for Critical risk.
#[must_use]
pub fn render_risk_bar(risk: &Risk) -> Paragraph<'static> {
    let color = risk_color(risk);
    let mut style = Style::default().fg(color);
    if matches!(risk, Risk::Critical) {
        style = style.add_modifier(Modifier::BOLD);
    }

    let label = risk.label().to_owned();
    let hint = if matches!(risk, Risk::Low | Risk::Medium) {
        "  [Ctrl+Enter to execute]"
    } else if matches!(risk, Risk::High) {
        "  [type 'yes' to confirm]"
    } else {
        "  [type full command to confirm]"
    };

    let line = Line::from(vec![
        Span::styled(label, style),
        Span::styled(hint.to_owned(), Style::default().fg(Color::DarkGray)),
    ]);

    Paragraph::new(line).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Risk ")
            .border_style(Style::default().fg(color)),
    )
}

// ---------------------------------------------------------------------------
// Rendering: Zone 2 — Command display
// ---------------------------------------------------------------------------

/// Render the command zone as a [`Paragraph`] with monospace styling.
#[must_use]
pub fn render_command_zone(proposal: &CommandProposal) -> Paragraph<'static> {
    let command = proposal.display_command();
    let display = if command.is_empty() {
        "<empty command>".to_owned()
    } else {
        command
    };

    let lines = vec![
        Line::from(Span::styled(
            proposal.summary.clone(),
            Style::default().fg(Color::White),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("$ {display}"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
    ];

    Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Command "),
    )
}

// ---------------------------------------------------------------------------
// Rendering: Zone 3 — Collapsible details
// ---------------------------------------------------------------------------

/// Render the details zone with collapsible sections.
#[must_use]
pub fn render_details_zone(view: &ProposalView) -> Paragraph<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for (idx, section) in ALL_SECTIONS.iter().enumerate() {
        let is_selected = idx == view.selected_section;
        let is_expanded = view.expanded.get(idx).copied().unwrap_or(false);
        let items = section_items(section, &view.proposal, &view.verdict);

        // Section header
        let marker = if is_expanded { "▼" } else { "▶" };
        let count = items.len();
        let header_text = format!("{marker} {} ({count})", section.label());

        let header_style = if is_selected {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::Gray)
        };
        lines.push(Line::from(Span::styled(header_text, header_style)));

        // Section body (only when expanded)
        if is_expanded {
            if items.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  (none)".to_owned(),
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                for item in &items {
                    lines.push(Line::from(Span::styled(
                        format!("  - {item}"),
                        Style::default().fg(Color::White),
                    )));
                }
            }
        }
    }

    Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Details "),
    )
}

/// Extract the display items for a given section from the proposal and verdict.
fn section_items(
    section: &DetailSection,
    proposal: &CommandProposal,
    verdict: &ProposalVerdict,
) -> Vec<String> {
    match section {
        DetailSection::RiskReasons => proposal.risk_reasons.clone(),
        DetailSection::Assumptions => proposal.assumptions.clone(),
        DetailSection::Evidence => proposal
            .evidence
            .iter()
            .map(|e| format!("{} ({:?})", e.claim, e.confidence))
            .collect(),
        DetailSection::Preflight => proposal
            .preflight
            .iter()
            .map(|p| {
                let cmd = p
                    .argv
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{cmd} — {}", p.reason)
            })
            .collect(),
        DetailSection::Rollback => proposal
            .rollback
            .as_ref()
            .map(|r| vec![format!("{:?}: {}", r.available, r.notes)])
            .unwrap_or_default(),
        DetailSection::MissingConfirmations => proposal.missing_confirmations.clone(),
        DetailSection::Warnings => verdict.warnings.clone(),
        DetailSection::Suggestions => verdict.suggestions.clone(),
    }
}

// ---------------------------------------------------------------------------
// Rendering: full app
// ---------------------------------------------------------------------------

/// Render the entire application into the given buffer area.
///
/// Layout: risk bar (3 lines) | command zone (5 lines) | details (remaining).
pub fn render_app(app: &App, area: Rect, buf: &mut Buffer) {
    match &app.proposal_view {
        Some(view) => {
            let chunks = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Min(5),
            ])
            .split(area);

            render_risk_bar(&view.verdict.risk).render(chunks[0], buf);
            render_command_zone(&view.proposal).render(chunks[1], buf);
            render_details_zone(view).render(chunks[2], buf);
        }
        None => {
            let status = match &app.state {
                AppState::Idle => "Ready. Type your intent and press Enter.".to_owned(),
                AppState::Exploring {
                    elapsed_secs,
                    active_agents,
                } => {
                    let agents = active_agents.join(", ");
                    format!("Exploring ({elapsed_secs}s) — agents: {agents}")
                }
                AppState::Proposing => "Generating proposal...".to_owned(),
                AppState::Editing => "Editing command...".to_owned(),
                AppState::Confirming { risk } => {
                    format!("Confirming {:?} risk action...", risk)
                }
            };

            let paragraph = Paragraph::new(status).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" cmd-proposer "),
            );
            paragraph.render(area, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cps_policy::ProposalVerdict;
    use cps_proposal::{
        Confidence, Evidence, EvidenceKind, EvidenceSource, PreflightCmd, RollbackInfo,
        RollbackStatus,
    };

    // ---- Helpers ----

    fn sample_proposal(risk: Risk) -> CommandProposal {
        CommandProposal {
            summary: "restart api deployment".to_owned(),
            argv: vec![
                "kubectl".to_owned(),
                "-n".to_owned(),
                "payments".to_owned(),
                "rollout".to_owned(),
                "restart".to_owned(),
                "deployment/api".to_owned(),
            ],
            display: "kubectl -n payments rollout restart deployment/api".to_owned(),
            risk,
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
            evidence: vec![Evidence {
                claim: "rollout restart is a scoped mutation".to_owned(),
                source: EvidenceSource {
                    kind: EvidenceKind::LocalDoc,
                    doc: "kubectl-help".to_owned(),
                    lines: Some((10, 30)),
                },
                confidence: Confidence::High,
            }],
            missing_confirmations: vec!["confirm production window".to_owned()],
        }
    }

    fn sample_verdict(risk: Risk) -> ProposalVerdict {
        ProposalVerdict {
            risk,
            findings: Vec::new(),
            warnings: vec!["implicit current context".to_owned()],
            suggestions: vec!["consider adding --dry-run".to_owned()],
            approved: true,
        }
    }

    // ---- Risk color mapping ----

    #[test]
    fn risk_color_low_is_green() {
        assert_eq!(risk_color(&Risk::Low), Color::Green);
    }

    #[test]
    fn risk_color_medium_is_yellow() {
        assert_eq!(risk_color(&Risk::Medium), Color::Yellow);
    }

    #[test]
    fn risk_color_high_is_red() {
        assert_eq!(risk_color(&Risk::High), Color::Red);
    }

    #[test]
    fn risk_color_critical_is_light_red() {
        assert_eq!(risk_color(&Risk::Critical), Color::LightRed);
    }

    // ---- Risk-based default expansion ----

    #[test]
    fn low_risk_all_collapsed() {
        let expanded = default_expansion(&Risk::Low);
        assert_eq!(expanded.len(), ALL_SECTIONS.len());
        assert!(expanded.iter().all(|e| !*e));
    }

    #[test]
    fn medium_risk_only_risk_reasons_expanded() {
        let expanded = default_expansion(&Risk::Medium);
        assert_eq!(expanded.len(), ALL_SECTIONS.len());
        // Only the first section (RiskReasons) should be expanded.
        assert!(expanded[0], "risk reasons should be expanded");
        for (idx, &val) in expanded.iter().enumerate().skip(1) {
            assert!(!val, "section at index {idx} should be collapsed");
        }
    }

    #[test]
    fn high_risk_all_expanded() {
        let expanded = default_expansion(&Risk::High);
        assert!(expanded.iter().all(|e| *e));
    }

    #[test]
    fn critical_risk_all_expanded() {
        let expanded = default_expansion(&Risk::Critical);
        assert!(expanded.iter().all(|e| *e));
    }

    // ---- Section expand/collapse ----

    #[test]
    fn toggle_expand_flips_selected_section() {
        let mut view = ProposalView::new(sample_proposal(Risk::Low), sample_verdict(Risk::Low));
        // All collapsed for low risk.
        assert!(!view.expanded[0]);
        view.toggle_selected();
        assert!(view.expanded[0]);
        view.toggle_selected();
        assert!(!view.expanded[0]);
    }

    #[test]
    fn navigate_up_wraps_around() {
        let mut view = ProposalView::new(sample_proposal(Risk::Low), sample_verdict(Risk::Low));
        assert_eq!(view.selected_section, 0);
        view.select_prev();
        assert_eq!(view.selected_section, ALL_SECTIONS.len() - 1);
    }

    #[test]
    fn navigate_down_wraps_around() {
        let mut view = ProposalView::new(sample_proposal(Risk::Low), sample_verdict(Risk::Low));
        view.selected_section = ALL_SECTIONS.len() - 1;
        view.select_next();
        assert_eq!(view.selected_section, 0);
    }

    // ---- Ctrl+Enter guard ----

    #[test]
    fn ctrl_enter_allowed_for_low_risk() {
        let view = ProposalView::new(sample_proposal(Risk::Low), sample_verdict(Risk::Low));
        assert!(view.ctrl_enter_allowed());
    }

    #[test]
    fn ctrl_enter_allowed_for_medium_risk() {
        let view =
            ProposalView::new(sample_proposal(Risk::Medium), sample_verdict(Risk::Medium));
        assert!(view.ctrl_enter_allowed());
    }

    #[test]
    fn ctrl_enter_disabled_for_high_risk() {
        let view = ProposalView::new(sample_proposal(Risk::High), sample_verdict(Risk::High));
        assert!(!view.ctrl_enter_allowed());
    }

    #[test]
    fn ctrl_enter_disabled_for_critical_risk() {
        let view = ProposalView::new(
            sample_proposal(Risk::Critical),
            sample_verdict(Risk::Critical),
        );
        assert!(!view.ctrl_enter_allowed());
    }

    // ---- Key binding dispatch ----

    #[test]
    fn ctrl_c_maps_to_abort() {
        let event = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(handle_key(event), Action::Abort);
    }

    #[test]
    fn ctrl_enter_maps_to_execute() {
        let event = KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL);
        assert_eq!(handle_key(event), Action::Execute);
    }

    #[test]
    fn tab_maps_to_edit() {
        let event = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::Edit);
    }

    #[test]
    fn esc_maps_to_cancel() {
        let event = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::Cancel);
    }

    #[test]
    fn up_maps_to_navigate_up() {
        let event = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::NavigateUp);
    }

    #[test]
    fn down_maps_to_navigate_down() {
        let event = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::NavigateDown);
    }

    #[test]
    fn enter_maps_to_toggle_expand() {
        let event = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::ToggleExpand);
    }

    #[test]
    fn digit_keys_map_to_run_preflight() {
        for digit in 1u8..=9 {
            let ch = (b'0' + digit) as char;
            let event = KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE);
            assert_eq!(handle_key(event), Action::RunPreflight(digit));
        }
    }

    #[test]
    fn zero_is_regular_input_not_preflight() {
        let event = KeyEvent::new(KeyCode::Char('0'), KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::Input('0'));
    }

    #[test]
    fn letter_maps_to_input() {
        let event = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::Input('a'));
    }

    #[test]
    fn unknown_key_maps_to_noop() {
        let event = KeyEvent::new(KeyCode::F(5), KeyModifiers::NONE);
        assert_eq!(handle_key(event), Action::Noop);
    }

    // ---- ProposalView creation ----

    #[test]
    fn proposal_view_new_uses_verdict_risk_for_expansion() {
        // Verdict says High even though proposal says Medium.
        let view = ProposalView::new(sample_proposal(Risk::Medium), sample_verdict(Risk::High));
        // High risk -> all expanded.
        assert!(view.expanded.iter().all(|e| *e));
        assert!(!view.ctrl_enter_allowed());
    }

    #[test]
    fn proposal_view_has_correct_section_count() {
        let view = ProposalView::new(sample_proposal(Risk::Low), sample_verdict(Risk::Low));
        assert_eq!(view.expanded.len(), ALL_SECTIONS.len());
    }

    // ---- App state ----

    #[test]
    fn app_default_is_idle() {
        let app = App::new();
        assert_eq!(app.state, AppState::Idle);
        assert!(app.proposal_view.is_none());
        assert!(app.input.is_empty());
        assert!(app.messages.is_empty());
    }

    // ---- Rendering smoke tests ----

    #[test]
    fn render_risk_bar_does_not_panic() {
        for risk in [Risk::Low, Risk::Medium, Risk::High, Risk::Critical] {
            let _widget = render_risk_bar(&risk);
        }
    }

    #[test]
    fn render_command_zone_does_not_panic() {
        let proposal = sample_proposal(Risk::Medium);
        let _widget = render_command_zone(&proposal);
    }

    #[test]
    fn render_details_zone_does_not_panic() {
        let view = ProposalView::new(sample_proposal(Risk::High), sample_verdict(Risk::High));
        let _widget = render_details_zone(&view);
    }

    #[test]
    fn render_app_without_proposal_does_not_panic() {
        let app = App::new();
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        render_app(&app, area, &mut buf);
    }

    #[test]
    fn render_app_with_proposal_does_not_panic() {
        let mut app = App::new();
        app.proposal_view = Some(ProposalView::new(
            sample_proposal(Risk::Medium),
            sample_verdict(Risk::Medium),
        ));
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        render_app(&app, area, &mut buf);
    }

    // ---- Section items ----

    #[test]
    fn section_items_returns_correct_data() {
        let proposal = sample_proposal(Risk::Medium);
        let verdict = sample_verdict(Risk::Medium);

        let risk_items = section_items(&DetailSection::RiskReasons, &proposal, &verdict);
        assert_eq!(risk_items, vec!["mutates a production workload"]);

        let assumption_items = section_items(&DetailSection::Assumptions, &proposal, &verdict);
        assert_eq!(assumption_items, vec!["api is a deployment"]);

        let evidence_items = section_items(&DetailSection::Evidence, &proposal, &verdict);
        assert_eq!(evidence_items.len(), 1);
        assert!(evidence_items[0].contains("rollout restart"));

        let preflight_items = section_items(&DetailSection::Preflight, &proposal, &verdict);
        assert_eq!(preflight_items.len(), 1);
        assert!(preflight_items[0].contains("confirm the workload exists"));

        let rollback_items = section_items(&DetailSection::Rollback, &proposal, &verdict);
        assert_eq!(rollback_items.len(), 1);
        assert!(rollback_items[0].contains("Partial"));

        let missing_items =
            section_items(&DetailSection::MissingConfirmations, &proposal, &verdict);
        assert_eq!(missing_items, vec!["confirm production window"]);

        let warning_items = section_items(&DetailSection::Warnings, &proposal, &verdict);
        assert_eq!(warning_items, vec!["implicit current context"]);

        let suggestion_items = section_items(&DetailSection::Suggestions, &proposal, &verdict);
        assert_eq!(suggestion_items, vec!["consider adding --dry-run"]);
    }
}

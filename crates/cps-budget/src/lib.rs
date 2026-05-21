//! Token budget computation as percentages of `effective_context` for cmd-proposer.
//!
//! See `drafts/SPEC.md` §3.
//!
//! The LLM context window includes thinking tokens, so all percentage budgets
//! are taken from `effective_context = max_context_tokens - thinking_budget`,
//! not from `max_context_tokens` directly.
//!
//! Percentage constants are spec invariants (Layer 0/1/2/3 + Reserve sum to
//! 100%). Only token counts are forbidden as hardcoded values — fractions are
//! definitional.

use cps_config::{Config, ModelConfig, ThinkingConfig};

// ---------- layer percentages (SPEC §3.4) ----------

const LAYER_SYSTEM_FRAC: f64 = 0.05;
const LAYER_CONVERSATION_FRAC: f64 = 0.15;
const LAYER_EVIDENCE_FRAC: f64 = 0.25;
const LAYER_TEMP_OUTPUT_FRAC: f64 = 0.30;
const LAYER_RESERVE_FRAC: f64 = 0.25;

// ---------- per-component fractions (SPEC §3.5) ----------

const TOOL_PREVIEW_FRAC: f64 = 0.05;
const SMALL_DOC_FULL_READ_FRAC: f64 = 0.10;
const SECTION_READ_FRAC: f64 = 0.15;
const SUBAGENT_CONTEXT_FRAC: f64 = 0.40;
const SUBAGENT_MAX_OUTPUT_FRAC: f64 = 0.08;
const FINAL_PROPOSAL_FRAC: f64 = 0.06;

/// Logical layer of the effective context, per SPEC §3.4.
///
/// The five layers partition `effective_context` (5 + 15 + 25 + 30 + 25 = 100).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    /// Layer 0: system prompt + safety rules (5%).
    Layer0System,
    /// Layer 1: user intent + conversation history (15%).
    Layer1Conversation,
    /// Layer 2: structured evidence retained across turns (25%).
    Layer2Evidence,
    /// Layer 3: temp tool output, discarded after retrieval (30%).
    Layer3TempOutput,
    /// Output generation reserve (25%).
    Reserve,
}

impl Layer {
    /// Fraction of `effective_context` allocated to this layer.
    pub const fn fraction(self) -> f64 {
        match self {
            Layer::Layer0System => LAYER_SYSTEM_FRAC,
            Layer::Layer1Conversation => LAYER_CONVERSATION_FRAC,
            Layer::Layer2Evidence => LAYER_EVIDENCE_FRAC,
            Layer::Layer3TempOutput => LAYER_TEMP_OUTPUT_FRAC,
            Layer::Reserve => LAYER_RESERVE_FRAC,
        }
    }
}

/// Computes all token budgets as percentages of `effective_context`.
///
/// All values are derived from `max_context_tokens` and the agent's
/// per-request `thinking_budget`. No token counts are hardcoded.
///
/// Construction never fails: if `thinking_budget >= max_context_tokens`
/// (a configuration error caught by [`cps_config::Config::validate`]),
/// `effective_context()` saturates to 0 and all derived limits are 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetEngine {
    max_context_tokens: u32,
    thinking_budget: u32,
}

impl BudgetEngine {
    /// Primary constructor. `thinking_budget` is the main agent's per-request
    /// reasoning budget (subagents construct their own engine with a smaller
    /// budget via [`BudgetEngine::for_subagent`]).
    pub fn new(max_context_tokens: u32, thinking_budget: u32) -> Self {
        Self {
            max_context_tokens,
            thinking_budget,
        }
    }

    /// Build the main-agent engine from a parsed config.
    pub fn from_config(cfg: &Config) -> Self {
        Self::from_model_and_thinking(&cfg.model, &cfg.thinking)
    }

    /// Build the main-agent engine from the two relevant config sections.
    pub fn from_model_and_thinking(model: &ModelConfig, thinking: &ThinkingConfig) -> Self {
        Self::new(model.max_context_tokens, thinking.main_agent)
    }

    /// Build a subagent engine, sharing `max_context_tokens` but with its own
    /// thinking budget (typically smaller). The main agent MAY pass an
    /// override per spawn; pass `None` to use `config.thinking.subagent_default`.
    pub fn for_subagent(&self, thinking: &ThinkingConfig, override_thinking: Option<u32>) -> Self {
        let t = override_thinking.unwrap_or(thinking.subagent_default);
        Self::new(self.max_context_tokens, t)
    }

    /// Configured max context window (includes thinking tokens).
    pub fn max_context_tokens(&self) -> u32 {
        self.max_context_tokens
    }

    /// This engine's per-request thinking budget.
    pub fn thinking_budget(&self) -> u32 {
        self.thinking_budget
    }

    /// `effective_context = max_context_tokens - thinking_budget`.
    ///
    /// Saturates to 0 if the configured thinking budget exceeds the window.
    pub fn effective_context(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.thinking_budget) as usize
    }

    /// Effective context for an agent (e.g. a subagent) sharing this engine's
    /// `max_context_tokens` but using a different `thinking_budget`.
    ///
    /// Saturates to 0 if `thinking_budget >= max_context_tokens`.
    pub fn effective_context_for(&self, thinking_budget: u32) -> usize {
        self.max_context_tokens.saturating_sub(thinking_budget) as usize
    }

    /// Token budget for a logical layer (SPEC §3.4).
    pub fn layer(&self, layer: Layer) -> usize {
        self.scaled(layer.fraction())
    }

    // ---------- per-component limits (SPEC §3.5) ----------

    /// Maximum tokens a tool-output preview may inject into the LLM context.
    pub fn tool_preview(&self) -> usize {
        self.scaled(TOOL_PREVIEW_FRAC)
    }

    /// Below this size, a document is read in full instead of being indexed.
    pub fn small_doc_full_read(&self) -> usize {
        self.scaled(SMALL_DOC_FULL_READ_FRAC)
    }

    /// Maximum tokens returned by a single `read_section` retrieval.
    pub fn section_read(&self) -> usize {
        self.scaled(SECTION_READ_FRAC)
    }

    /// Tokens of context the main agent may pass to one subagent at spawn.
    pub fn subagent_context(&self) -> usize {
        self.scaled(SUBAGENT_CONTEXT_FRAC)
    }

    /// Maximum tokens a subagent may emit in its response.
    pub fn subagent_max_output(&self) -> usize {
        self.scaled(SUBAGENT_MAX_OUTPUT_FRAC)
    }

    /// Evidence budget retained across main-agent turns (= Layer 2).
    pub fn main_evidence_budget(&self) -> usize {
        self.layer(Layer::Layer2Evidence)
    }

    /// Maximum tokens for the final structured proposal payload.
    pub fn final_proposal(&self) -> usize {
        self.scaled(FINAL_PROPOSAL_FRAC)
    }

    fn scaled(&self, frac: f64) -> usize {
        ((self.effective_context() as f64) * frac).floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default config values from cps-config: 200K context, 32_768 main thinking,
    // 4_096 subagent default thinking. effective = 167_232.
    const MAX_CTX: u32 = 200_000;
    const MAIN_THINKING: u32 = 32_768;
    const SUBAGENT_THINKING: u32 = 4_096;
    const EFFECTIVE: usize = (MAX_CTX - MAIN_THINKING) as usize; // 167_232

    fn floor_pct(eff: usize, frac: f64) -> usize {
        ((eff as f64) * frac).floor() as usize
    }

    #[test]
    fn effective_context_with_default_config() {
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        assert_eq!(b.effective_context(), EFFECTIVE);
        assert_eq!(b.effective_context(), 167_232);
    }

    #[test]
    fn layer_budgets_match_spec_percentages() {
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        assert_eq!(b.layer(Layer::Layer0System), floor_pct(EFFECTIVE, 0.05));
        assert_eq!(
            b.layer(Layer::Layer1Conversation),
            floor_pct(EFFECTIVE, 0.15)
        );
        assert_eq!(b.layer(Layer::Layer2Evidence), floor_pct(EFFECTIVE, 0.25));
        assert_eq!(b.layer(Layer::Layer3TempOutput), floor_pct(EFFECTIVE, 0.30));
        assert_eq!(b.layer(Layer::Reserve), floor_pct(EFFECTIVE, 0.25));
    }

    #[test]
    fn layer_fractions_sum_to_one() {
        let total = Layer::Layer0System.fraction()
            + Layer::Layer1Conversation.fraction()
            + Layer::Layer2Evidence.fraction()
            + Layer::Layer3TempOutput.fraction()
            + Layer::Reserve.fraction();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "layer fractions sum to {total}, expected 1.0"
        );
    }

    #[test]
    fn layer_budgets_floor_sum_within_rounding() {
        // Floor of fractional shares can lose a few tokens vs effective; ensure
        // the loss is bounded by the layer count (5 layers => ≤ 5 tokens lost).
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        let sum = b.layer(Layer::Layer0System)
            + b.layer(Layer::Layer1Conversation)
            + b.layer(Layer::Layer2Evidence)
            + b.layer(Layer::Layer3TempOutput)
            + b.layer(Layer::Reserve);
        let diff = EFFECTIVE - sum;
        assert!(
            diff <= 5,
            "layer sum {sum} differs from effective {EFFECTIVE} by {diff}"
        );
    }

    #[test]
    fn component_limits_match_spec_table() {
        // Values from SPEC.md §3.5 example column (eff ≈ 167K, 32K thinking).
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        assert_eq!(b.tool_preview(), 8_361);
        assert_eq!(b.small_doc_full_read(), 16_723);
        assert_eq!(b.section_read(), 25_084);
        assert_eq!(b.subagent_context(), 66_892);
        assert_eq!(b.subagent_max_output(), 13_378);
        assert_eq!(b.main_evidence_budget(), 41_808);
        assert_eq!(b.final_proposal(), 10_033);
    }

    #[test]
    fn main_evidence_budget_equals_layer_2() {
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        assert_eq!(b.main_evidence_budget(), b.layer(Layer::Layer2Evidence));
    }

    #[test]
    fn effective_context_for_subagent() {
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        assert_eq!(
            b.effective_context_for(SUBAGENT_THINKING),
            (MAX_CTX - SUBAGENT_THINKING) as usize,
        );
        // Custom override (e.g. risk_reviewer may want 8K).
        assert_eq!(b.effective_context_for(8_192), (MAX_CTX - 8_192) as usize);
    }

    #[test]
    fn edge_thinking_zero_uses_full_window() {
        let b = BudgetEngine::new(MAX_CTX, 0);
        assert_eq!(b.effective_context(), MAX_CTX as usize);
        assert_eq!(b.tool_preview(), floor_pct(MAX_CTX as usize, 0.05));
    }

    #[test]
    fn edge_thinking_max_minus_one_leaves_one_token() {
        let b = BudgetEngine::new(MAX_CTX, MAX_CTX - 1);
        assert_eq!(b.effective_context(), 1);
        // Every fractional component floors to 0 at effective=1.
        assert_eq!(b.tool_preview(), 0);
        assert_eq!(b.subagent_context(), 0);
        assert_eq!(b.layer(Layer::Layer2Evidence), 0);
    }

    #[test]
    fn edge_thinking_equals_max_saturates_to_zero() {
        let b = BudgetEngine::new(MAX_CTX, MAX_CTX);
        assert_eq!(b.effective_context(), 0);
        assert_eq!(b.layer(Layer::Layer0System), 0);
        assert_eq!(b.tool_preview(), 0);
    }

    #[test]
    fn edge_thinking_exceeds_max_saturates_to_zero() {
        // Config::validate rejects this; engine handles it gracefully anyway.
        let b = BudgetEngine::new(MAX_CTX, MAX_CTX + 10_000);
        assert_eq!(b.effective_context(), 0);
        assert_eq!(b.effective_context_for(MAX_CTX + 1), 0);
    }

    #[test]
    fn from_config_uses_main_agent_thinking() {
        let yaml = r#"
model:
  base_url: "http://localhost:8317/v1"
  api_key: "k"
  model_name: "m"
  tokenizer:
    path: "/tmp/t"

doc_runner:
  allow_programs: [kubectl]
"#;
        let cfg: Config = serde_yaml::from_str(yaml).expect("parse");
        cfg.validate().expect("validate");
        let b = BudgetEngine::from_config(&cfg);
        assert_eq!(b.max_context_tokens(), 200_000);
        assert_eq!(b.thinking_budget(), 32_768);
        assert_eq!(b.effective_context(), 167_232);
    }

    #[test]
    fn for_subagent_default_and_override() {
        let b = BudgetEngine::new(MAX_CTX, MAIN_THINKING);
        let thinking = ThinkingConfig::default();
        let sub_default = b.for_subagent(&thinking, None);
        assert_eq!(sub_default.thinking_budget(), SUBAGENT_THINKING);
        assert_eq!(
            sub_default.effective_context(),
            (MAX_CTX - SUBAGENT_THINKING) as usize
        );

        let sub_heavy = b.for_subagent(&thinking, Some(16_384));
        assert_eq!(sub_heavy.thinking_budget(), 16_384);
        assert_eq!(sub_heavy.effective_context(), (MAX_CTX - 16_384) as usize);
    }

    #[test]
    fn small_context_window_still_consistent() {
        // Smaller window; every limit should still be a strict fraction.
        let b = BudgetEngine::new(32_000, 4_000);
        let eff = 28_000usize;
        assert_eq!(b.effective_context(), eff);
        assert_eq!(b.tool_preview(), floor_pct(eff, 0.05));
        assert_eq!(b.subagent_context(), floor_pct(eff, 0.40));
        assert_eq!(b.layer(Layer::Layer3TempOutput), floor_pct(eff, 0.30));
    }
}

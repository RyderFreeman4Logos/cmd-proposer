//! Runtime token-budget tracking for the main conversation loop.

use std::collections::HashMap;
use std::sync::Arc;

use cps_budget::{BudgetEngine, Layer};
use cps_llm::{Message, Role};
use cps_tokenizer::Tokenizer;
use serde_json::Value;

const APPROACHING_LIMIT_FRAC: f64 = 0.80;
const EVIDENCE_SUMMARY_PREFIX: &str = "Evidence summary (Layer 2):";

/// Current budget status for a candidate chat request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetStatus {
    pub within_budget: bool,
    pub layer_usage: HashMap<Layer, usize>,
    pub suggestions: Vec<String>,
}

/// Counts conversation tokens by SPEC context layer.
pub struct BudgetTracker {
    pub budget: BudgetEngine,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub layer_usage: HashMap<Layer, usize>,
    immutable_usage: HashMap<Layer, usize>,
}

impl BudgetTracker {
    /// Create a tracker with immutable Layer 0 and Layer 1 prefix costs.
    pub fn new(
        budget: BudgetEngine,
        tokenizer: Arc<dyn Tokenizer>,
        system_prompt: &str,
        tool_definitions: &[Value],
    ) -> Self {
        let mut immutable_usage = empty_layer_usage();
        add_layer_tokens(
            &mut immutable_usage,
            Layer::Layer0System,
            tokenizer.count_tokens(system_prompt),
        );
        add_layer_tokens(
            &mut immutable_usage,
            Layer::Layer1Conversation,
            count_tool_definition_tokens(tokenizer.as_ref(), tool_definitions),
        );

        Self {
            budget,
            tokenizer,
            layer_usage: immutable_usage.clone(),
            immutable_usage,
        }
    }

    /// Recompute token usage for the current mutable message history.
    pub fn check_budget(&mut self, messages: &[Message]) -> BudgetStatus {
        let mut usage = self.immutable_usage.clone();
        for message in messages {
            let layer = classify_message_layer(message);
            add_layer_tokens(&mut usage, layer, self.count_message_tokens(message));
        }

        let suggestions = build_suggestions(self.budget, &usage);
        let within_budget = suggestions
            .iter()
            .all(|suggestion| !suggestion.starts_with("over budget"));

        self.layer_usage = usage.clone();
        BudgetStatus {
            within_budget,
            layer_usage: usage,
            suggestions,
        }
    }

    pub fn count_message_tokens(&self, message: &Message) -> usize {
        count_message_tokens(self.tokenizer.as_ref(), message)
    }
}

pub fn classify_message_layer(message: &Message) -> Layer {
    match message.role {
        Role::System => Layer::Layer0System,
        Role::Tool => Layer::Layer3TempOutput,
        Role::Assistant if is_evidence_message(message) => Layer::Layer2Evidence,
        Role::Assistant | Role::User => Layer::Layer1Conversation,
    }
}

pub fn is_evidence_message(message: &Message) -> bool {
    message.content.starts_with(EVIDENCE_SUMMARY_PREFIX)
}

pub fn count_message_tokens(tokenizer: &dyn Tokenizer, message: &Message) -> usize {
    match serde_json::to_string(message) {
        Ok(serialized) => tokenizer.count_tokens(&serialized),
        Err(_) => tokenizer.count_tokens(&message.content),
    }
}

pub fn count_tool_definition_tokens(
    tokenizer: &dyn Tokenizer,
    tool_definitions: &[Value],
) -> usize {
    tool_definitions
        .iter()
        .map(|tool| {
            serde_json::to_string(tool)
                .map(|serialized| tokenizer.count_tokens(&serialized))
                .unwrap_or_default()
        })
        .sum()
}

fn build_suggestions(budget: BudgetEngine, usage: &HashMap<Layer, usize>) -> Vec<String> {
    let mut suggestions = Vec::new();
    for layer in [
        Layer::Layer0System,
        Layer::Layer1Conversation,
        Layer::Layer2Evidence,
        Layer::Layer3TempOutput,
    ] {
        let used = layer_tokens(usage, layer);
        let limit = budget.layer(layer);
        if used > limit {
            suggestions.push(format!(
                "over budget: {layer:?} uses {used} tokens; limit is {limit}"
            ));
        } else if limit > 0 && (used as f64) >= (limit as f64 * APPROACHING_LIMIT_FRAC) {
            suggestions.push(format!(
                "approaching budget: {layer:?} uses {used} tokens; limit is {limit}"
            ));
        }
    }

    let reserve = budget.layer(Layer::Reserve);
    let used_without_reserve = [
        Layer::Layer0System,
        Layer::Layer1Conversation,
        Layer::Layer2Evidence,
        Layer::Layer3TempOutput,
    ]
    .into_iter()
    .map(|layer| layer_tokens(usage, layer))
    .sum::<usize>();
    if used_without_reserve.saturating_add(reserve) > budget.effective_context() {
        suggestions.push(format!(
            "over budget: request uses {used_without_reserve} tokens, leaving less than reserve {reserve}"
        ));
    }

    suggestions
}

pub fn layer_tokens(usage: &HashMap<Layer, usize>, layer: Layer) -> usize {
    usage.get(&layer).copied().unwrap_or_default()
}

fn empty_layer_usage() -> HashMap<Layer, usize> {
    [
        Layer::Layer0System,
        Layer::Layer1Conversation,
        Layer::Layer2Evidence,
        Layer::Layer3TempOutput,
        Layer::Reserve,
    ]
    .into_iter()
    .map(|layer| (layer, 0))
    .collect()
}

fn add_layer_tokens(usage: &mut HashMap<Layer, usize>, layer: Layer, tokens: usize) {
    *usage.entry(layer).or_default() += tokens;
}

#[cfg(test)]
mod tests {
    use cps_llm::Message;
    use cps_tokenizer::FallbackTokenizer;
    use serde_json::json;

    use super::*;

    fn tracker(budget: BudgetEngine) -> BudgetTracker {
        BudgetTracker::new(budget, Arc::new(FallbackTokenizer::new()), "", &[])
    }

    #[test]
    fn token_counting_accumulates_across_messages() {
        let mut tracker = tracker(BudgetEngine::new(10_000, 1_000));
        let messages = vec![
            Message::user("a".repeat(400)),
            Message::assistant("b".repeat(800)),
        ];

        let status = tracker.check_budget(&messages);

        assert!(
            layer_tokens(&status.layer_usage, Layer::Layer1Conversation) > 0,
            "conversation messages should be counted"
        );
        assert_eq!(
            layer_tokens(&status.layer_usage, Layer::Layer2Evidence),
            0,
            "no evidence messages were present"
        );
    }

    #[test]
    fn budget_check_detects_over_limit() {
        let mut tracker = tracker(BudgetEngine::new(400, 100));
        let messages = vec![Message::tool_result("call_1", "x".repeat(2_000))];

        let status = tracker.check_budget(&messages);

        assert!(!status.within_budget);
        assert!(status
            .suggestions
            .iter()
            .any(|suggestion| suggestion.contains("Layer3TempOutput")));
    }

    #[test]
    fn layer_classification_matches_message_roles_and_prefixes() {
        let system = Message::system("system");
        let user = Message::user("user");
        let evidence = Message::assistant("Evidence summary (Layer 2):\n- read_help: ok\n");
        let raw = Message::tool_result("call_1", "{}");

        assert_eq!(classify_message_layer(&system), Layer::Layer0System);
        assert_eq!(classify_message_layer(&user), Layer::Layer1Conversation);
        assert_eq!(classify_message_layer(&evidence), Layer::Layer2Evidence);
        assert_eq!(classify_message_layer(&raw), Layer::Layer3TempOutput);
    }

    #[test]
    fn tool_definitions_are_counted_in_layer_one_prefix() {
        let mut tracker = BudgetTracker::new(
            BudgetEngine::new(10_000, 1_000),
            Arc::new(FallbackTokenizer::new()),
            "system prompt",
            &[json!({
                "type": "function",
                "function": {
                    "name": "read_help",
                    "description": "Read command help text"
                }
            })],
        );

        let status = tracker.check_budget(&[]);

        assert!(layer_tokens(&status.layer_usage, Layer::Layer0System) > 0);
        assert!(layer_tokens(&status.layer_usage, Layer::Layer1Conversation) > 0);
    }
}

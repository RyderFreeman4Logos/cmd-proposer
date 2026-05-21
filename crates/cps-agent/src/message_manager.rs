//! KV cache-friendly message management.
//!
//! Enforces the two critical KV cache invariants from SPEC section 3.6:
//!
//! - **Rule 3**: Conversation is append-only. Messages are never rewritten or
//!   deleted. When token budget compression is needed, a new summary message is
//!   appended instead.
//! - **Rule 4**: Subagent prompts share a common prefix. The prefix hash tracks
//!   the (system_prompt, tool_definitions) pair so that any unexpected mutation
//!   is detected and logged as an error.
//!
//! The module also tracks per-message token usage and marks which messages are
//! compressible Layer 3 raw output.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use cps_budget::Layer;
use cps_llm::Message;
use cps_tokenizer::Tokenizer;
use serde_json::Value;

use crate::budget::classify_message_layer;

/// Metadata tracked for each message in the conversation.
#[derive(Debug, Clone)]
struct MessageEntry {
    message: Message,
    token_count: usize,
    layer: Layer,
    compressible: bool,
    /// Index of the summary message that supersedes this entry's content.
    /// `None` means the entry is still "live" (not yet compressed).
    compressed_by: Option<usize>,
}

/// Append-only, KV cache-aware message store.
///
/// All mutations go through `add_message` (which only appends). There are no
/// `remove`, `swap`, or `replace` methods. Compression appends a new summary
/// message and marks the originals as compressed -- it never touches their
/// content.
pub struct MessageManager {
    entries: Vec<MessageEntry>,
    tokenizer: Arc<dyn Tokenizer>,
    prefix_hash: u64,
}

impl MessageManager {
    /// Create a new manager and compute the initial prefix hash.
    ///
    /// `system_prompt` and `tool_definitions` form the immutable prefix that
    /// the LLM provider's KV cache keys on. The hash is checked before every
    /// LLM call via [`Self::verify_prefix`].
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        system_prompt: &str,
        tool_definitions: &[Value],
    ) -> Self {
        let prefix_hash = compute_prefix_hash(system_prompt, tool_definitions);
        tracing::debug!(prefix_hash, "MessageManager initialized with prefix hash");
        Self {
            entries: Vec::new(),
            tokenizer,
            prefix_hash,
        }
    }

    /// Append a message to the conversation.
    ///
    /// This is the only way to add content. The message is classified into a
    /// context layer and its tokens are counted immediately.
    pub fn add_message(&mut self, message: Message) {
        let token_count = self.count_tokens(&message);
        let layer = classify_message_layer(&message);
        let compressible = layer == Layer::Layer3TempOutput;
        self.entries.push(MessageEntry {
            message,
            token_count,
            layer,
            compressible,
            compressed_by: None,
        });
    }

    /// Number of messages in the conversation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the conversation is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only view of all messages, in order.
    #[must_use]
    pub fn messages(&self) -> Vec<Message> {
        self.entries.iter().map(|e| e.message.clone()).collect()
    }

    /// Token count for a specific message by index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[must_use]
    pub fn token_count_at(&self, index: usize) -> Option<usize> {
        self.entries.get(index).map(|e| e.token_count)
    }

    /// Total tokens across all messages.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.token_count).sum()
    }

    /// The prefix hash computed at initialization.
    ///
    /// This is a non-cryptographic hash of (system_prompt + tool_definitions).
    /// It should remain constant for the lifetime of the session.
    #[must_use]
    pub fn conversation_prefix_hash(&self) -> u64 {
        self.prefix_hash
    }

    /// Verify that the prefix has not changed since initialization.
    ///
    /// Returns `true` if the prefix is stable (expected). If it returns `false`,
    /// the caller should treat this as a bug -- it means the system prompt or
    /// tool definitions were mutated mid-session, which invalidates the
    /// provider's KV cache.
    pub fn verify_prefix(&self, system_prompt: &str, tool_definitions: &[Value]) -> bool {
        let current = compute_prefix_hash(system_prompt, tool_definitions);
        if current != self.prefix_hash {
            tracing::error!(
                expected = self.prefix_hash,
                actual = current,
                "KV cache prefix changed unexpectedly -- this is a bug that causes cache misses"
            );
            return false;
        }
        true
    }

    /// Context layer classification for a message at the given index.
    #[must_use]
    pub fn layer_at(&self, index: usize) -> Option<Layer> {
        self.entries.get(index).map(|e| e.layer)
    }

    /// Indices of messages that are compressible (Layer 3 raw output) and have
    /// not already been compressed.
    #[must_use]
    pub fn compressible_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.compressible && e.compressed_by.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Append a compression summary for one or more source messages.
    ///
    /// This enforces the append-only rule: the original messages are never
    /// modified. Instead, a new summary message is appended and the originals
    /// are marked as `compressed_by` pointing to the summary's index.
    ///
    /// Returns the index of the newly appended summary message, or `None` if
    /// `source_indices` was empty or contained only already-compressed entries.
    pub fn append_compression_summary(
        &mut self,
        source_indices: &[usize],
        summary: Message,
    ) -> Option<usize> {
        // Filter to valid, uncompressed indices.
        let valid: Vec<usize> = source_indices
            .iter()
            .copied()
            .filter(|&i| {
                self.entries
                    .get(i)
                    .is_some_and(|e| e.compressible && e.compressed_by.is_none())
            })
            .collect();

        if valid.is_empty() {
            return None;
        }

        let summary_index = self.entries.len();
        let token_count = self.count_tokens(&summary);
        let layer = classify_message_layer(&summary);
        self.entries.push(MessageEntry {
            message: summary,
            token_count,
            layer,
            compressible: false,
            compressed_by: None,
        });

        for &i in &valid {
            self.entries[i].compressed_by = Some(summary_index);
        }

        Some(summary_index)
    }

    /// Whether a message at the given index has been compressed (superseded by
    /// a later summary).
    #[must_use]
    pub fn is_compressed(&self, index: usize) -> bool {
        self.entries
            .get(index)
            .is_some_and(|e| e.compressed_by.is_some())
    }

    fn count_tokens(&self, message: &Message) -> usize {
        crate::budget::count_message_tokens(self.tokenizer.as_ref(), message)
    }
}

/// Compute a non-cryptographic hash of the immutable prefix.
fn compute_prefix_hash(system_prompt: &str, tool_definitions: &[Value]) -> u64 {
    let mut hasher = DefaultHasher::new();
    system_prompt.hash(&mut hasher);
    for tool in tool_definitions {
        // Deterministic serialization: serde_json with sorted keys would be
        // ideal, but the tool list is built from static `json!()` literals
        // whose key order is insertion-order-deterministic. Hashing the
        // `to_string()` output is sufficient for same-process stability.
        if let Ok(s) = serde_json::to_string(tool) {
            s.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cps_llm::{Message, Role};
    use cps_tokenizer::FallbackTokenizer;
    use serde_json::json;

    use super::*;

    fn test_tokenizer() -> Arc<dyn Tokenizer> {
        Arc::new(FallbackTokenizer::new())
    }

    fn sample_tool_defs() -> Vec<Value> {
        vec![
            json!({
                "type": "function",
                "function": {
                    "name": "read_help",
                    "description": "Read help text",
                    "parameters": { "type": "object", "properties": {} }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "done",
                    "description": "End the turn",
                    "parameters": { "type": "object", "properties": {} }
                }
            }),
        ]
    }

    fn manager() -> MessageManager {
        MessageManager::new(test_tokenizer(), "system prompt", &sample_tool_defs())
    }

    // ---------------------------------------------------------------
    // Append-only: messages can only grow, never shrink
    // ---------------------------------------------------------------

    #[test]
    fn messages_can_only_grow() {
        let mut mgr = manager();
        assert_eq!(mgr.len(), 0);

        mgr.add_message(Message::user("hello"));
        assert_eq!(mgr.len(), 1);

        mgr.add_message(Message::assistant("hi"));
        assert_eq!(mgr.len(), 2);

        mgr.add_message(Message::user("what now?"));
        assert_eq!(mgr.len(), 3);

        // There are no remove/swap/replace methods, so length can never
        // decrease. This test codifies that the API surface enforces
        // append-only semantics.
    }

    #[test]
    fn message_ordering_is_preserved() {
        let mut mgr = manager();
        mgr.add_message(Message::user("first"));
        mgr.add_message(Message::assistant("second"));
        mgr.add_message(Message::user("third"));

        let messages = mgr.messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "first");
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].content, "second");
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[2].content, "third");
        assert_eq!(messages[2].role, Role::User);
    }

    #[test]
    fn prior_messages_unchanged_after_append() {
        let mut mgr = manager();
        mgr.add_message(Message::user("first"));
        let snapshot = mgr.messages();

        mgr.add_message(Message::assistant("second"));
        mgr.add_message(Message::tool_result("call_1", "result data"));

        let current = mgr.messages();
        // All prior messages are still exactly where they were.
        assert_eq!(current[0].content, snapshot[0].content);
        assert_eq!(current[0].role, snapshot[0].role);
        assert_eq!(current.len(), 3);
    }

    // ---------------------------------------------------------------
    // Prefix hash is stable across calls
    // ---------------------------------------------------------------

    #[test]
    fn prefix_hash_is_stable_across_calls() {
        let mgr = manager();
        let h1 = mgr.conversation_prefix_hash();
        let h2 = mgr.conversation_prefix_hash();
        let h3 = mgr.conversation_prefix_hash();
        assert_eq!(h1, h2);
        assert_eq!(h2, h3);
    }

    #[test]
    fn prefix_hash_differs_for_different_prompts() {
        let tools = sample_tool_defs();
        let mgr_a = MessageManager::new(test_tokenizer(), "prompt A", &tools);
        let mgr_b = MessageManager::new(test_tokenizer(), "prompt B", &tools);
        assert_ne!(mgr_a.conversation_prefix_hash(), mgr_b.conversation_prefix_hash());
    }

    #[test]
    fn prefix_hash_differs_for_different_tools() {
        let tools_a = vec![json!({
            "type": "function",
            "function": { "name": "alpha", "description": "a" }
        })];
        let tools_b = vec![json!({
            "type": "function",
            "function": { "name": "beta", "description": "b" }
        })];
        let mgr_a = MessageManager::new(test_tokenizer(), "same", &tools_a);
        let mgr_b = MessageManager::new(test_tokenizer(), "same", &tools_b);
        assert_ne!(mgr_a.conversation_prefix_hash(), mgr_b.conversation_prefix_hash());
    }

    #[test]
    fn verify_prefix_succeeds_when_unchanged() {
        let mgr = manager();
        assert!(mgr.verify_prefix("system prompt", &sample_tool_defs()));
    }

    #[test]
    fn verify_prefix_fails_when_prompt_changes() {
        let mgr = manager();
        assert!(!mgr.verify_prefix("CHANGED prompt", &sample_tool_defs()));
    }

    #[test]
    fn verify_prefix_fails_when_tools_change() {
        let mgr = manager();
        let different_tools = vec![json!({
            "type": "function",
            "function": { "name": "new_tool", "description": "x" }
        })];
        assert!(!mgr.verify_prefix("system prompt", &different_tools));
    }

    // ---------------------------------------------------------------
    // Compression appends new message rather than modifying existing
    // ---------------------------------------------------------------

    #[test]
    fn compression_appends_summary_without_modifying_originals() {
        let mut mgr = manager();

        // Add a compressible Layer 3 tool result.
        mgr.add_message(Message::tool_result("call_1", "large output data here"));
        let original_content = mgr.messages()[0].content.clone();
        let original_tokens = mgr.token_count_at(0).unwrap();

        // Compress it.
        let summary = Message::assistant("Layer 3 compressed: summary of call_1");
        let summary_idx = mgr
            .append_compression_summary(&[0], summary.clone())
            .expect("compression should succeed");

        // The original message is untouched.
        assert_eq!(mgr.messages()[0].content, original_content);
        assert_eq!(mgr.token_count_at(0).unwrap(), original_tokens);

        // A new summary message was appended.
        assert_eq!(mgr.len(), 2);
        assert_eq!(summary_idx, 1);
        assert_eq!(mgr.messages()[1].content, summary.content);

        // The original is marked as compressed.
        assert!(mgr.is_compressed(0));
        // The summary itself is not compressed.
        assert!(!mgr.is_compressed(1));
    }

    #[test]
    fn double_compression_is_idempotent() {
        let mut mgr = manager();
        mgr.add_message(Message::tool_result("call_1", "output"));
        mgr.append_compression_summary(&[0], Message::assistant("summary 1"));

        // Second compression attempt on the same index should be a no-op.
        let result = mgr.append_compression_summary(&[0], Message::assistant("summary 2"));
        assert!(result.is_none(), "already-compressed message should be skipped");
        assert_eq!(mgr.len(), 2, "no new message should have been added");
    }

    #[test]
    fn compression_of_empty_indices_is_noop() {
        let mut mgr = manager();
        mgr.add_message(Message::user("hello"));
        let result = mgr.append_compression_summary(&[], Message::assistant("summary"));
        assert!(result.is_none());
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn only_layer3_messages_are_compressible() {
        let mut mgr = manager();
        mgr.add_message(Message::user("user input"));
        mgr.add_message(Message::assistant("assistant reply"));
        mgr.add_message(Message::tool_result("call_1", "tool output"));

        let compressible = mgr.compressible_indices();
        assert_eq!(compressible, vec![2], "only the tool result (Layer 3) should be compressible");
    }

    // ---------------------------------------------------------------
    // Token tracking
    // ---------------------------------------------------------------

    #[test]
    fn token_counts_are_tracked_per_message() {
        let mut mgr = manager();
        mgr.add_message(Message::user("short"));
        mgr.add_message(Message::user("a".repeat(500)));

        let t0 = mgr.token_count_at(0).unwrap();
        let t1 = mgr.token_count_at(1).unwrap();
        assert!(t0 > 0);
        assert!(t1 > t0, "longer message should have more tokens");
        assert_eq!(mgr.total_tokens(), t0 + t1);
    }

    #[test]
    fn out_of_bounds_token_count_returns_none() {
        let mgr = manager();
        assert!(mgr.token_count_at(0).is_none());
        assert!(mgr.token_count_at(999).is_none());
    }

    #[test]
    fn is_empty_reflects_state() {
        let mut mgr = manager();
        assert!(mgr.is_empty());
        mgr.add_message(Message::user("x"));
        assert!(!mgr.is_empty());
    }
}

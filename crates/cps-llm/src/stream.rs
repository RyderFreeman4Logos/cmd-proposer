//! SSE chunk parsing and streaming assembly.
//!
//! OpenAI-compatible servers stream chat completions as Server-Sent Events:
//! each `data: <json>` line carries a partial choice. Content arrives as
//! `delta.content` string deltas; tool calls arrive as `delta.tool_calls`
//! whose `function.arguments` are concatenated across many chunks.
//!
//! The final event is `data: [DONE]`. Some providers also include `usage`
//! on the last chunk; we capture it when present.

use serde::Deserialize;

use crate::error::{LlmError, Result};
use crate::types::{FunctionCall, Message, Role, ToolCall, Usage};

const MAX_PARALLEL_TOOL_CALLS: usize = 128;

/// One streamed event the caller's callback observes.
///
/// `delta_content` is the incremental token chunk for the assistant message.
/// `delta_tool_calls` is the incremental tool-call payload — multiple chunks
/// must be assembled by index before the JSON arguments are parseable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamChunk {
    pub delta_content: Option<String>,
    pub delta_tool_calls: Option<Vec<ToolCallDelta>>,
    pub finish_reason: Option<String>,
}

/// Incremental fragment of a tool call inside a stream chunk.
///
/// `index` is the stable position of this tool call within the assistant's
/// list (deltas for the same call carry the same `index`). `id` and
/// `function.name` arrive on the first fragment; subsequent fragments carry
/// only `function.arguments` string deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function_name: Option<String>,
    pub function_arguments_delta: Option<String>,
}

// ---------- wire types for SSE payload deserialization ----------

#[derive(Debug, Deserialize)]
struct StreamEvent {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RawToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct RawToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<RawFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct RawFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Parse one SSE `data:` payload into a [`StreamChunk`] plus optional usage.
///
/// Returns `Ok(None)` for `[DONE]` sentinels and for events that carry
/// neither a delta nor a finish reason (e.g. role-only header chunks); the
/// caller should skip these without invoking the user callback.
pub(crate) fn parse_event(
    json_str: &str,
) -> std::result::Result<Option<(StreamChunk, Option<Usage>)>, serde_json::Error> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(None);
    }
    let event: StreamEvent = serde_json::from_str(trimmed)?;
    let usage = event.usage;
    let mut chunk = StreamChunk::default();

    if let Some(choice) = event.choices.into_iter().next() {
        chunk.delta_content = choice.delta.content.filter(|s| !s.is_empty());
        chunk.finish_reason = choice.finish_reason;
        chunk.delta_tool_calls = choice.delta.tool_calls.map(|raws| {
            raws.into_iter()
                .enumerate()
                .map(|(fallback_index, raw)| ToolCallDelta {
                    // Some providers omit `index` on the first chunk of the
                    // first call; fall back to positional order.
                    index: raw.index.unwrap_or(fallback_index),
                    id: raw.id,
                    function_name: raw.function.as_ref().and_then(|f| f.name.clone()),
                    function_arguments_delta: raw.function.and_then(|f| f.arguments),
                })
                .collect()
        });
    }

    // A bare `{"choices":[]}` keepalive or a role-only header carries no
    // useful signal; skip those rather than firing the callback.
    if chunk.delta_content.is_none()
        && chunk.delta_tool_calls.is_none()
        && chunk.finish_reason.is_none()
        && usage.is_none()
    {
        return Ok(None);
    }

    Ok(Some((chunk, usage)))
}

/// Accumulates streamed deltas into a fully-formed assistant [`Message`].
///
/// Tool calls are reassembled by their `index`, since servers may interleave
/// fragments for multiple parallel calls. `id` and `function.name` are
/// retained from whichever fragment first carried them; `arguments` are
/// concatenated in arrival order.
#[derive(Debug, Default)]
pub(crate) struct StreamAssembler {
    content: String,
    /// Sparse map: index → (id, name, arguments_buffer).
    tool_calls: Vec<Option<PartialToolCall>>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: Option<Usage>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl StreamAssembler {
    pub(crate) fn ingest(&mut self, chunk: &StreamChunk) -> Result<()> {
        if let Some(c) = &chunk.delta_content {
            self.content.push_str(c);
        }
        if let Some(deltas) = &chunk.delta_tool_calls {
            for d in deltas {
                let required_len = d.index.checked_add(1).ok_or_else(|| {
                    LlmError::Malformed(format!("tool call index {} overflows usize", d.index))
                })?;
                if required_len > MAX_PARALLEL_TOOL_CALLS {
                    return Err(LlmError::Malformed(format!(
                        "tool call index {} exceeds max parallel tool calls {}",
                        d.index, MAX_PARALLEL_TOOL_CALLS
                    )));
                }
                if self.tool_calls.len() <= d.index {
                    self.tool_calls.resize_with(required_len, || None);
                }
                let slot = self.tool_calls[d.index].get_or_insert_with(PartialToolCall::default);
                if let Some(id) = &d.id {
                    slot.id = Some(id.clone());
                }
                if let Some(name) = &d.function_name {
                    slot.name = Some(name.clone());
                }
                if let Some(args) = &d.function_arguments_delta {
                    slot.arguments.push_str(args);
                }
            }
        }
        if let Some(r) = &chunk.finish_reason {
            self.finish_reason = Some(r.clone());
        }
        Ok(())
    }

    pub(crate) fn set_usage(&mut self, usage: Option<Usage>) {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
    }

    /// Convert accumulated state into a final assistant [`Message`].
    ///
    /// Tool calls missing both `id` and `name` are dropped (incomplete
    /// fragments that never coalesced). Sparse gaps from out-of-order
    /// indices are likewise skipped.
    pub(crate) fn finalize(self) -> (Message, Option<String>, Option<Usage>) {
        let tool_calls: Vec<ToolCall> = self
            .tool_calls
            .into_iter()
            .flatten()
            .filter_map(|p| {
                let id = p.id?;
                let name = p.name?;
                Some(ToolCall {
                    id,
                    call_type: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: p.arguments,
                    },
                })
            })
            .collect();

        let message = Message {
            role: Role::Assistant,
            content: self.content,
            tool_call_id: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        };
        (message, self.finish_reason, self.usage)
    }
}

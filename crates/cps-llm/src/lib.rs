//! OpenAI-compatible LLM API client: streaming and tool-call parsing.
//!
//! See `drafts/SPEC.md` §2, §4, §12.
//!
//! The client speaks the OpenAI `/v1/chat/completions` schema (which is the
//! de-facto standard exposed by vLLM, SGLang, llama.cpp's HTTP server, and
//! most commercial endpoints). It supports:
//!
//! - Non-streaming completion via [`LlmClient::complete`].
//! - Streaming completion via [`LlmClient::complete_streaming`], assembled
//!   from incremental SSE chunks while invoking a user callback for
//!   progressive display.
//! - Tool/function calls — round-trip via the [`Message::tool_calls`] /
//!   [`Message::tool_result`] message types.
//! - Per-request thinking budget passed as `max_completion_tokens`.
//!
//! KV cache friendliness (SPEC §3.6) is enforced by the *caller*: this
//! crate sends `tools` exactly as supplied (no re-sorting) and never
//! rewrites prior messages. Stable ordering is the agent's responsibility.

mod client;
mod error;
mod stream;
mod types;

pub use client::{ClientConfig, LlmClient};
pub use error::{LlmError, Result};
pub use stream::{StreamChunk, ToolCallDelta};
pub use types::{ChatRequest, ChatResponse, FunctionCall, Message, Role, ToolCall, Usage};

// `parse_event` and `StreamAssembler` are intentionally crate-private — the
// only supported way to consume streams is via `LlmClient::complete_streaming`.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------- serialization shape ----------

    #[test]
    fn message_serialization_matches_openai_format() {
        // System
        let m = Message::system("you are helpful");
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(j, json!({"role": "system", "content": "you are helpful"}));

        // User
        let m = Message::user("hi");
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(j, json!({"role": "user", "content": "hi"}));

        // Assistant with tool calls
        let m = Message {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_abc".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_help".into(),
                    arguments: r#"{"program":"kubectl"}"#.into(),
                },
            }]),
        };
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(
            j,
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "read_help",
                        "arguments": r#"{"program":"kubectl"}"#
                    }
                }]
            })
        );

        // Tool result
        let m = Message::tool_result("call_abc", "kubectl --help output...");
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(
            j,
            json!({
                "role": "tool",
                "content": "kubectl --help output...",
                "tool_call_id": "call_abc"
            })
        );
    }

    #[test]
    fn chat_request_serialization_includes_only_set_fields() {
        let req = ChatRequest {
            model: "qwen3-27b".into(),
            messages: vec![Message::user("hello")],
            tools: Some(vec![json!({
                "type": "function",
                "function": { "name": "doc_grep", "description": "grep a doc" }
            })]),
            max_completion_tokens: Some(2048),
            stream: true,
            temperature: None,
        };

        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "qwen3-27b");
        assert_eq!(v["stream"], true);
        assert_eq!(v["max_completion_tokens"], 2048);
        assert!(v["tools"].is_array());
        // `temperature` was None → must not serialize.
        assert!(v.get("temperature").is_none());
    }

    #[test]
    fn chat_request_minimal_omits_optional_fields() {
        let req = ChatRequest::new("m", vec![Message::user("hi")]);
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("tools").is_none());
        assert!(v.get("max_completion_tokens").is_none());
        assert!(v.get("temperature").is_none());
        assert_eq!(v["stream"], false);
    }

    #[test]
    fn tools_preserve_caller_supplied_order() {
        // SPEC §3.6: tool definitions must be in a stable, caller-defined
        // order for KV cache reuse. We MUST NOT re-sort.
        let tools = vec![
            json!({"function": {"name": "zzz"}}),
            json!({"function": {"name": "aaa"}}),
            json!({"function": {"name": "mmm"}}),
        ];
        let req = ChatRequest {
            tools: Some(tools.clone()),
            ..ChatRequest::new("m", vec![])
        };
        let v = serde_json::to_value(&req).unwrap();
        let serialized = v["tools"].as_array().unwrap();
        assert_eq!(serialized[0]["function"]["name"], "zzz");
        assert_eq!(serialized[1]["function"]["name"], "aaa");
        assert_eq!(serialized[2]["function"]["name"], "mmm");
    }

    // ---------- response parsing ----------

    #[test]
    fn non_streaming_response_with_tool_call_parses() {
        let raw = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "read_help",
                            "arguments": "{\"program\":\"kubectl\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
        });
        let parsed: super::types::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        let choice = &parsed.choices[0];
        assert_eq!(choice.message.role, Role::Assistant);
        let calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_xyz");
        assert_eq!(calls[0].function.name, "read_help");
        assert_eq!(calls[0].function.arguments, "{\"program\":\"kubectl\"}");
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(parsed.usage.unwrap().total_tokens, 46);
    }

    #[test]
    fn non_streaming_tool_call_response_accepts_null_content() {
        let raw = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_null",
                        "type": "function",
                        "function": {
                            "name": "read_help",
                            "arguments": "{\"program\":\"kubectl\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let parsed: super::types::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
        let message = &parsed.choices[0].message;
        assert_eq!(message.content, "");
        assert_eq!(message.tool_calls.as_ref().unwrap()[0].id, "call_null");
    }

    #[test]
    fn non_streaming_plain_assistant_response_parses() {
        let raw = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop"
            }]
        });
        let parsed: super::types::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
        let m = &parsed.choices[0].message;
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.content, "hi there");
        assert!(m.tool_calls.is_none());
    }

    // ---------- streaming parsing ----------

    #[test]
    fn parse_event_handles_done_sentinel() {
        let r = super::stream::parse_event("[DONE]").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn parse_event_handles_content_delta() {
        let payload = r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#;
        let (chunk, usage) = super::stream::parse_event(payload).unwrap().unwrap();
        assert_eq!(chunk.delta_content.as_deref(), Some("Hel"));
        assert!(chunk.delta_tool_calls.is_none());
        assert!(usage.is_none());
    }

    #[test]
    fn parse_event_handles_tool_call_delta() {
        let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_help","arguments":"{\"p"}}]}}]}"#;
        let (chunk, _) = super::stream::parse_event(payload).unwrap().unwrap();
        let deltas = chunk.delta_tool_calls.unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(deltas[0].function_name.as_deref(), Some("read_help"));
        assert_eq!(deltas[0].function_arguments_delta.as_deref(), Some("{\"p"));
    }

    #[test]
    fn parse_event_handles_finish_reason_chunk() {
        let payload = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let (chunk, _) = super::stream::parse_event(payload).unwrap().unwrap();
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn parse_event_skips_empty_keepalive() {
        let payload = r#"{"choices":[]}"#;
        let r = super::stream::parse_event(payload).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn stream_assembler_concatenates_tool_call_arguments() {
        let mut a = super::stream::StreamAssembler::default();
        // Fragment 1: header
        let (c1, _) = super::stream::parse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"read_help","arguments":""}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        a.ingest(&c1).unwrap();
        // Fragment 2: partial args
        let (c2, _) = super::stream::parse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"program\":\"kub"}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        a.ingest(&c2).unwrap();
        // Fragment 3: rest of args + finish
        let (c3, _) = super::stream::parse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ectl\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap()
        .unwrap();
        a.ingest(&c3).unwrap();

        let (msg, fr, _usage) = a.finalize();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(fr.as_deref(), Some("tool_calls"));
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_x");
        assert_eq!(calls[0].function.name, "read_help");
        assert_eq!(calls[0].function.arguments, "{\"program\":\"kubectl\"}");
        // Argument JSON parses cleanly now that it's reassembled.
        let v: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(v["program"], "kubectl");
    }

    #[test]
    fn stream_assembler_drops_incomplete_tool_call() {
        // Saw a fragment with arguments but never the header (no id/name).
        let mut a = super::stream::StreamAssembler::default();
        let (c1, _) = super::stream::parse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        a.ingest(&c1).unwrap();
        let (msg, _, _) = a.finalize();
        assert!(
            msg.tool_calls.is_none(),
            "tool call without id/name must be dropped"
        );
    }

    #[test]
    fn stream_assembler_concatenates_content_deltas() {
        let mut a = super::stream::StreamAssembler::default();
        for word in ["Hel", "lo, ", "world", "!"] {
            let payload = format!(r#"{{"choices":[{{"delta":{{"content":"{word}"}}}}]}}"#);
            let (chunk, _) = super::stream::parse_event(&payload).unwrap().unwrap();
            a.ingest(&chunk).unwrap();
        }
        let (msg, _, _) = a.finalize();
        assert_eq!(msg.content, "Hello, world!");
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn stream_assembler_rejects_oversized_tool_call_index() {
        let mut a = super::stream::StreamAssembler::default();
        let chunk = StreamChunk {
            delta_tool_calls: Some(vec![ToolCallDelta {
                index: 1_000_000_000,
                id: Some("call_large".into()),
                function_name: Some("read_help".into()),
                function_arguments_delta: Some("{}".into()),
            }]),
            ..StreamChunk::default()
        };

        let err = a.ingest(&chunk).unwrap_err();
        assert!(matches!(err, LlmError::Malformed(message) if message.contains("tool call index")));
    }

    #[tokio::test]
    async fn complete_streaming_handles_utf8_split_across_http_chunks() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        async fn write_http_chunk(socket: &mut tokio::net::TcpStream, bytes: &[u8]) {
            socket
                .write_all(format!("{:x}\r\n", bytes.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(bytes).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
        }

        let event = b"data: {\"choices\":[{\"delta\":{\"content\":\"\xe4\xbd\xa0\"}}]}\n\n";
        let glyph_start = event.windows(3).position(|w| w == b"\xe4\xbd\xa0").unwrap();
        let split_at = glyph_start + 2;
        let first_chunk = event[..split_at].to_vec();
        let second_chunk = event[split_at..].to_vec();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let _bytes_read = socket.read(&mut request).await.unwrap();

            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
                )
                .await
                .unwrap();
            write_http_chunk(&mut socket, &first_chunk).await;
            write_http_chunk(&mut socket, &second_chunk).await;
            write_http_chunk(&mut socket, b"data: [DONE]\n\n").await;
            socket.write_all(b"0\r\n\r\n").await.unwrap();
            socket.shutdown().await.unwrap();
        });

        let client = LlmClient::new(&ClientConfig {
            base_url: format!("http://{addr}/v1"),
            api_key: String::new(),
            request_timeout_ms: 1000,
        })
        .unwrap();
        let request = ChatRequest::new("test-model", vec![Message::user("hello")]);
        let mut streamed = String::new();

        let response = client
            .complete_streaming(&request, |chunk| {
                if let Some(content) = chunk.delta_content {
                    streamed.push_str(&content);
                }
            })
            .await
            .unwrap();

        server.await.unwrap();
        assert_eq!(streamed, "\u{4f60}");
        assert_eq!(response.message.content, "\u{4f60}");
    }

    // ---------- error classification ----------

    #[test]
    fn from_status_classifies_correctly() {
        assert!(matches!(
            LlmError::from_status(408, String::new(), None),
            LlmError::Timeout { timeout_ms: 0 }
        ));
        assert!(matches!(
            LlmError::from_status(401, String::new(), None),
            LlmError::Unauthorized
        ));
        assert!(matches!(
            LlmError::from_status(403, String::new(), None),
            LlmError::Unauthorized
        ));
        assert!(matches!(
            LlmError::from_status(429, String::new(), Some(30)),
            LlmError::RateLimit {
                retry_after_secs: Some(30)
            }
        ));
        assert!(matches!(
            LlmError::from_status(425, "too early".into(), None),
            LlmError::Server { status: 425, .. }
        ));
        assert!(matches!(
            LlmError::from_status(500, "boom".into(), None),
            LlmError::Server { status: 500, .. }
        ));
        assert!(matches!(
            LlmError::from_status(503, "down".into(), None),
            LlmError::Server { status: 503, .. }
        ));
        assert!(matches!(
            LlmError::from_status(400, "bad json".into(), None),
            LlmError::BadRequest { status: 400, .. }
        ));
    }

    #[test]
    fn is_transient_matches_retryable_classes() {
        assert!(LlmError::Timeout { timeout_ms: 1000 }.is_transient());
        assert!(LlmError::Connection("refused".into()).is_transient());
        assert!(LlmError::RateLimit {
            retry_after_secs: None
        }
        .is_transient());
        assert!(LlmError::from_status(408, String::new(), None).is_transient());
        assert!(LlmError::from_status(425, String::new(), None).is_transient());
        assert!(LlmError::Server {
            status: 500,
            body: String::new()
        }
        .is_transient());

        assert!(!LlmError::Unauthorized.is_transient());
        assert!(!LlmError::BadRequest {
            status: 400,
            body: String::new()
        }
        .is_transient());
        assert!(!LlmError::Malformed("x".into()).is_transient());
        assert!(!LlmError::InvalidConfig("x".into()).is_transient());
    }

    // ---------- construction ----------

    #[test]
    fn new_rejects_empty_base_url() {
        let cfg = ClientConfig {
            base_url: String::new(),
            api_key: "sk".into(),
            request_timeout_ms: 1000,
        };
        let err = LlmClient::new(&cfg).unwrap_err();
        assert!(matches!(err, LlmError::InvalidConfig(_)));
    }

    #[test]
    fn new_accepts_empty_api_key_for_local_endpoint() {
        let cfg = ClientConfig {
            base_url: "http://localhost:8000/v1".into(),
            api_key: String::new(),
            request_timeout_ms: 1000,
        };
        LlmClient::new(&cfg).expect("empty api_key allowed for no-auth local endpoint");
    }

    #[test]
    fn new_rejects_api_key_with_invalid_header_bytes() {
        let cfg = ClientConfig {
            base_url: "http://localhost:8000/v1".into(),
            // HTTP header values reject NUL and CR/LF.
            api_key: "bad\nkey".into(),
            request_timeout_ms: 1000,
        };
        let err = LlmClient::new(&cfg).unwrap_err();
        assert!(matches!(err, LlmError::InvalidConfig(_)));
    }
}

//! Main agent loop, evidence compression, and parallel subagent pool.
//!
//! Today this crate exposes only the session-prefix builder
//! ([`SessionInit`]): the immutable system prompt + tool definitions sent to
//! the LLM once per session. The agent loop itself is added by a later issue.

pub mod session;
pub mod system_prompt;
pub mod tools;

pub use session::SessionInit;
pub use tools::ToolFeatureFlags;

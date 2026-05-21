//! Main agent loop, evidence compression, and parallel subagent pool.
//!
//! The crate owns both the immutable session prefix ([`SessionInit`]) and the
//! main conversation runtime ([`AgentLoop`]).

pub mod agent_loop;
pub mod session;
pub mod system_prompt;
pub mod tools;

pub use agent_loop::{
    AgentError, AgentLoop, AgentLoopParts, AgentState, CompletionClient, LayerTokenUsage, Result,
    TurnResult,
};
pub use session::SessionInit;
pub use tools::ToolFeatureFlags;

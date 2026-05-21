//! Main agent loop, evidence compression, and parallel subagent pool.
//!
//! The crate owns both the immutable session prefix ([`SessionInit`]) and the
//! main conversation runtime ([`AgentLoop`]).

pub mod agent_loop;
pub mod budget;
pub mod message_manager;
pub mod session;
pub mod subagent;
pub mod system_prompt;
pub mod tools;

pub use agent_loop::{
    AgentError, AgentLoop, AgentLoopParts, AgentState, CompletionClient, LayerTokenUsage, Result,
    TokenCounters, TurnResult,
};
pub use budget::{BudgetStatus, BudgetTracker};
pub use message_manager::MessageManager;
pub use session::SessionInit;
pub use subagent::{
    SpawnRequest, SubagentError, SubagentPool, SubagentResult, SubagentRole, SubagentStatus,
};
pub use tools::ToolFeatureFlags;

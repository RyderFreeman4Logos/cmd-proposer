//! Subagent pool for parallel lightweight LLM workers.
//!
//! The subagent pool manages spawning and executing parallel LLM conversations
//! with bounded context and tool access, designed for lightweight exploration
//! and risk review tasks per SPEC §9.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cps_llm::{Message, ChatRequest, ChatResponse};
use crate::agent_loop::CompletionClient;
use cps_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use thiserror::Error;

/// MVP subagent roles per SPEC §9.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentRole {
    /// Explore help/man, find syntax, return evidence spans
    DocExplorer,
    /// Review candidate argv for risk, suggest preflight/dry-run
    RiskReviewer,
}

impl SubagentRole {
    /// Return the allowed tools for this role per SPEC §9.1
    pub fn allowed_tools(self) -> Vec<&'static str> {
        match self {
            SubagentRole::DocExplorer => vec![
                "read_help",
                "read_man",
                "read_info",
                "doc_token_count",
                "doc_preview",
                "doc_grep",
                "doc_section",
                "doc_lines",
                "doc_expand_around",
                "web_search",
            ],
            SubagentRole::RiskReviewer => vec![
                "doc_token_count",
                "doc_preview",
                "doc_grep",
                "doc_section",
                "doc_lines",
                "doc_expand_around",
            ],
        }
    }

    /// Get system prompt for this role
    pub fn system_prompt(self) -> &'static str {
        match self {
            SubagentRole::DocExplorer => {
                "You are a documentation explorer subagent. Your role is to efficiently explore \
                help pages, man pages, and documentation to find specific syntax, flags, and usage \
                patterns. You MUST return structured findings with evidence spans. Stay focused \
                on your assigned goal and do not explore beyond the scope given to you."
            }
            SubagentRole::RiskReviewer => {
                "You are a risk review subagent. Your role is to review command proposals for \
                potential risks and suggest preflight checks or dry-run alternatives. You MUST \
                analyze the blast radius, identify destructive operations, and provide concrete \
                safety recommendations. Stay focused on risk analysis only."
            }
        }
    }
}

/// Request to spawn a subagent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnRequest {
    pub role: SubagentRole,
    pub goal: String,
    pub input: Value,
    pub allowed_tools: Vec<String>,
    pub context_tokens: usize,
    pub thinking_budget: Option<u32>,
    pub timeout_ms: u64,
    pub output_schema: String,
}

/// Result returned by a subagent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub subagent_id: String,
    pub status: SubagentStatus,
    pub findings: Vec<String>,
    pub candidate_fragments: Vec<String>,
    pub open_questions: Vec<String>,
}

/// Status of a subagent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Running,
    WaitingTool,
    Summarizing,
    Completed,
    Timeout,
    Error,
}

/// Error types for subagent operations
#[derive(Debug, Error)]
pub enum SubagentError {
    #[error("subagent pool at capacity: {active}/{max}")]
    PoolAtCapacity { active: usize, max: usize },

    #[error("invalid role: {role} (allowed: {allowed:?})")]
    InvalidRole { role: String, allowed: Vec<String> },

    #[error("tool not allowed for role {role:?}: {tool} (allowed: {allowed:?})")]
    ToolNotAllowed {
        role: SubagentRole,
        tool: String,
        allowed: Vec<String>,
    },

    #[error("context budget exceeded: {requested} > {max}")]
    ContextBudgetExceeded { requested: usize, max: usize },

    #[error("timeout exceeded: {timeout_ms}ms")]
    TimeoutExceeded { timeout_ms: u64 },

    #[error("subagent execution failed: {source}")]
    ExecutionFailed {
        #[from]
        source: cps_llm::LlmError,
    },

    #[error("JSON parsing failed: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },
}

/// Manages parallel subagent execution
pub struct SubagentPool {
    max_parallel: usize,
    #[allow(dead_code)]
    default_timeout: Duration,
    active: Arc<Mutex<Vec<JoinHandle<Result<SubagentResult, SubagentError>>>>>,
    next_id: Arc<Mutex<u64>>,
}

impl SubagentPool {
    /// Create a new subagent pool with the given configuration
    pub fn new(max_parallel: usize, default_timeout_ms: u64) -> Self {
        Self {
            max_parallel,
            default_timeout: Duration::from_millis(default_timeout_ms),
            active: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Spawn a new subagent with the given request
    pub async fn spawn<L>(
        &self,
        request: SpawnRequest,
        llm: &L,
        model_name: &str,
        max_subagent_context: usize,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<SubagentResult, SubagentError>
    where
        L: CompletionClient + Clone + 'static,
    {
        // Validate the request
        self.validate_request(&request, max_subagent_context)?;

        // Check if we can spawn another subagent
        let mut active_handles = self.active.lock().map_err(|_| SubagentError::ExecutionFailed {
            source: cps_llm::LlmError::Timeout { timeout_ms: 0 }
        })?;

        // Clean up completed handles
        active_handles.retain(|handle| !handle.is_finished());

        if active_handles.len() >= self.max_parallel {
            return Err(SubagentError::PoolAtCapacity {
                active: active_handles.len(),
                max: self.max_parallel,
            });
        }

        // Generate unique ID
        let subagent_id = {
            let mut counter = self.next_id.lock().map_err(|_| SubagentError::ExecutionFailed {
                source: cps_llm::LlmError::Timeout { timeout_ms: 0 },
            })?;
            let id = *counter;
            *counter += 1;
            format!("{:?}-{}", request.role, id)
        };

        // Clone necessary data for the async task
        let llm_clone = llm.clone();
        let request_clone = request.clone();
        let model_name_clone = model_name.to_owned();
        let timeout_duration = Duration::from_millis(request.timeout_ms);

        // Spawn the subagent task
        let handle = tokio::spawn(async move {
            let result = timeout(
                timeout_duration,
                execute_subagent(subagent_id.clone(), request_clone, llm_clone, model_name_clone, tokenizer)
            ).await;

            match result {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(error)) => Err(error),
                Err(_) => Err(SubagentError::TimeoutExceeded {
                    timeout_ms: timeout_duration.as_millis() as u64,
                }),
            }
        });

        active_handles.push(handle);
        drop(active_handles); // Release the lock

        // For this implementation, we'll wait for completion
        // In a more advanced implementation, this could return immediately
        // and allow polling of results
        let result_handle = {
            let mut handles = self.active.lock().map_err(|_| SubagentError::ExecutionFailed {
                source: cps_llm::LlmError::Timeout { timeout_ms: 0 },
            })?;
            handles.pop().unwrap() // We just pushed it
        };

        result_handle.await.map_err(|_| SubagentError::ExecutionFailed {
            source: cps_llm::LlmError::Timeout { timeout_ms: 0 },
        })?
    }

    /// Validate a spawn request
    fn validate_request(
        &self,
        request: &SpawnRequest,
        max_subagent_context: usize,
    ) -> Result<(), SubagentError> {
        // Check context budget
        if request.context_tokens > max_subagent_context {
            return Err(SubagentError::ContextBudgetExceeded {
                requested: request.context_tokens,
                max: max_subagent_context,
            });
        }

        // Check that all requested tools are allowed for this role
        let role_tools = request.role.allowed_tools();
        for tool in &request.allowed_tools {
            if !role_tools.contains(&tool.as_str()) {
                return Err(SubagentError::ToolNotAllowed {
                    role: request.role,
                    tool: tool.clone(),
                    allowed: role_tools.iter().map(|&s| s.to_owned()).collect(),
                });
            }
        }

        Ok(())
    }

    /// Get the count of active subagents
    pub fn active_count(&self) -> usize {
        self.active
            .lock()
            .map(|mut handles| {
                handles.retain(|handle| !handle.is_finished());
                handles.len()
            })
            .unwrap_or(0)
    }

    /// Cancel all active subagents
    pub async fn cancel_all(&self) {
        if let Ok(mut active_handles) = self.active.lock() {
            for handle in active_handles.drain(..) {
                handle.abort();
            }
        }
    }
}

/// Execute a subagent with the given parameters
async fn execute_subagent<L>(
    subagent_id: String,
    request: SpawnRequest,
    llm: L,
    model_name: String,
    _tokenizer: Arc<dyn Tokenizer>,
) -> Result<SubagentResult, SubagentError>
where
    L: CompletionClient,
{
    // Build the conversation
    let mut messages = Vec::new();

    // Add system prompt
    messages.push(Message::system(request.role.system_prompt()));

    // Add goal and input
    let user_content = format!(
        "GOAL: {}\n\nINPUT: {}\n\nPlease analyze and provide findings according to your role.",
        request.goal,
        serde_json::to_string_pretty(&request.input)?
    );
    messages.push(Message::user(user_content));

    // Create the chat request
    let thinking_budget = request.thinking_budget.unwrap_or(4096);
    let chat_request = ChatRequest {
        model: model_name,
        messages,
        tools: Some(build_tool_subset(&request.allowed_tools)),
        max_completion_tokens: Some(thinking_budget),
        stream: false,
        temperature: Some(0.1),
        tool_choice: Some(serde_json::Value::String("auto".to_owned())),
    };

    // Execute the LLM call
    let response = llm.complete(&chat_request).await?;

    // Parse the response and extract findings
    let findings = parse_subagent_findings(&response, &subagent_id)?;

    Ok(SubagentResult {
        subagent_id,
        status: SubagentStatus::Completed,
        findings: findings.findings,
        candidate_fragments: findings.candidate_fragments,
        open_questions: findings.open_questions,
    })
}

/// Build a subset of tools for the subagent
fn build_tool_subset(_allowed_tools: &[String]) -> Vec<Value> {
    // For MVP, return empty tools list
    // In full implementation, this would build actual tool definitions
    // based on the allowed_tools list
    Vec::new()
}

/// Findings extracted from subagent response
#[derive(Debug)]
struct ParsedFindings {
    findings: Vec<String>,
    candidate_fragments: Vec<String>,
    open_questions: Vec<String>,
}

/// Parse findings from subagent response
fn parse_subagent_findings(
    response: &ChatResponse,
    subagent_id: &str,
) -> Result<ParsedFindings, SubagentError> {
    // For MVP, extract findings from the response content
    // In full implementation, this would parse structured output

    let content = &response.message.content;
    let mut findings = Vec::new();
    let mut candidate_fragments = Vec::new();
    let mut open_questions = Vec::new();

    // Simple parsing - look for structured sections
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("FINDING:") {
            findings.push(line.strip_prefix("FINDING:").unwrap_or(line).trim().to_owned());
        } else if line.starts_with("FRAGMENT:") {
            candidate_fragments.push(line.strip_prefix("FRAGMENT:").unwrap_or(line).trim().to_owned());
        } else if line.starts_with("QUESTION:") {
            open_questions.push(line.strip_prefix("QUESTION:").unwrap_or(line).trim().to_owned());
        } else if !line.is_empty() && !line.starts_with("GOAL:") && !line.starts_with("INPUT:") {
            // Treat other non-empty lines as findings
            findings.push(line.to_owned());
        }
    }

    // If we didn't get any structured output, use the whole content as a finding
    if findings.is_empty() && candidate_fragments.is_empty() && open_questions.is_empty() {
        findings.push(format!("Subagent {} completed: {}", subagent_id, content));
    }

    Ok(ParsedFindings {
        findings,
        candidate_fragments,
        open_questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn subagent_role_allowed_tools() {
        let doc_explorer_tools = SubagentRole::DocExplorer.allowed_tools();
        assert!(doc_explorer_tools.contains(&"read_help"));
        assert!(doc_explorer_tools.contains(&"web_search"));
        assert_eq!(doc_explorer_tools.len(), 10);

        let risk_reviewer_tools = SubagentRole::RiskReviewer.allowed_tools();
        assert!(risk_reviewer_tools.contains(&"doc_grep"));
        assert!(!risk_reviewer_tools.contains(&"read_help"));
        assert!(!risk_reviewer_tools.contains(&"web_search"));
        assert_eq!(risk_reviewer_tools.len(), 6);
    }

    #[test]
    fn spawn_request_validation() {
        let pool = SubagentPool::new(2, 60000);

        let valid_request = SpawnRequest {
            role: SubagentRole::DocExplorer,
            goal: "Find kubectl syntax".to_owned(),
            input: json!({"intent": "list pods"}),
            allowed_tools: vec!["read_help".to_owned(), "doc_grep".to_owned()],
            context_tokens: 1000,
            thinking_budget: Some(2048),
            timeout_ms: 30000,
            output_schema: "SubagentFindingV1".to_owned(),
        };

        assert!(pool.validate_request(&valid_request, 5000).is_ok());

        // Test context budget exceeded
        let budget_exceeded = SpawnRequest {
            context_tokens: 10000,
            ..valid_request.clone()
        };
        assert!(matches!(
            pool.validate_request(&budget_exceeded, 5000),
            Err(SubagentError::ContextBudgetExceeded { .. })
        ));

        // Test tool not allowed
        let invalid_tool = SpawnRequest {
            allowed_tools: vec!["read_help".to_owned(), "execute".to_owned()],
            ..valid_request.clone()
        };
        assert!(matches!(
            pool.validate_request(&invalid_tool, 5000),
            Err(SubagentError::ToolNotAllowed { .. })
        ));
    }

    #[test]
    fn pool_capacity_management() {
        let pool = SubagentPool::new(2, 60000);
        assert_eq!(pool.active_count(), 0);

        // In a real test, we would spawn actual tasks and verify capacity limits
        // For now, just verify the pool structure is correct
        assert_eq!(pool.max_parallel, 2);
    }

    #[test]
    fn parse_subagent_findings_structured() {
        let response = ChatResponse {
            message: Message::assistant("FINDING: kubectl get is read-only\nFRAGMENT: kubectl get pods\nQUESTION: Which namespace?"),
            usage: None,
            finish_reason: None,
        };

        let findings = parse_subagent_findings(&response, "test-1").unwrap();
        assert_eq!(findings.findings, vec!["kubectl get is read-only"]);
        assert_eq!(findings.candidate_fragments, vec!["kubectl get pods"]);
        assert_eq!(findings.open_questions, vec!["Which namespace?"]);
    }

    #[test]
    fn parse_subagent_findings_unstructured() {
        let response = ChatResponse {
            message: Message::assistant("The command appears to be safe for read-only operations."),
            usage: None,
            finish_reason: None,
        };

        let findings = parse_subagent_findings(&response, "test-1").unwrap();
        assert_eq!(findings.findings.len(), 1);
        assert!(findings.findings[0].contains("The command appears to be safe"));
        assert!(findings.candidate_fragments.is_empty());
        assert!(findings.open_questions.is_empty());
    }
}
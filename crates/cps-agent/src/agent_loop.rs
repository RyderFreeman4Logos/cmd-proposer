//! Main-agent conversation loop and evidence compression.
//!
//! The loop keeps the LLM prefix stable: `system_prompt` and
//! `tool_definitions` are set once, while `messages` only grows by appending
//! user, assistant, tool, and evidence-summary messages.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use cps_budget::{BudgetEngine, Layer};
use cps_config::Config;
use cps_doc_runner::{DocRunner, HelpStyle};
use cps_doc_store::{DocStore, DocStoreError, GrepMatch, SourceKind};
use cps_llm::{ChatRequest, ChatResponse, LlmClient, Message, StreamChunk, ToolCall};
use cps_policy::{
    DocLookup, PolicyConfig, PolicyError, PolicyFinding, PolicyFindingSeverity, PolicyGate,
    ToolCall as PolicyToolCall,
};
use cps_proposal::CommandProposal;
use cps_tokenizer::{create_tokenizer, Tokenizer};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::budget::{
    classify_message_layer, count_message_tokens, is_evidence_message, layer_tokens, BudgetStatus,
    BudgetTracker,
};
use crate::session::SessionInit;
use crate::subagent::{SubagentPool, SpawnRequest, SubagentRole};

const MAX_TOOL_ROUNDS: usize = 8;
const DOC_EXPLORER_ROLE: &str = "doc_explorer";
const RISK_REVIEWER_ROLE: &str = "risk_reviewer";

/// Agent workflow state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    Idle,
    ReceiveIntent,
    Plan,
    ExploreDocs,
    MergeFindings,
    GenerateProposal,
    PolicyCheck,
    HumanReview,
}

/// Result returned after one human input turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    pub state: AgentState,
    pub proposal: Option<CommandProposal>,
    pub needs_input: bool,
}

/// Token usage tracked by logical context layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerTokenUsage {
    pub layer1_conversation: usize,
    pub layer2_evidence: usize,
    pub layer3_temp_output: usize,
}

/// Error type for the agent loop.
pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Llm(#[from] cps_llm::LlmError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("doc store lock poisoned while accessing {0}")]
    LockPoisoned(&'static str),

    #[error("tool round limit exceeded: {limit}")]
    ToolRoundLimit { limit: usize },

    #[error("proposal rejected by policy: {messages}")]
    ProposalRejected { messages: String },

    #[error("conversation token budget exceeded after compression: {messages}")]
    BudgetExceeded { messages: String },
}

/// Minimal LLM boundary used by [`AgentLoop`].
///
/// Production uses [`LlmClient`]; tests supply a deterministic mock with no
/// network endpoint.
pub trait CompletionClient: Send + Sync {
    fn complete_streaming<'a, F>(
        &'a self,
        request: &'a ChatRequest,
        callback: F,
    ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>>
    where
        F: FnMut(StreamChunk) + Send + 'a;

    fn complete<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>>;
}

impl CompletionClient for LlmClient {
    fn complete_streaming<'a, F>(
        &'a self,
        request: &'a ChatRequest,
        callback: F,
    ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>>
    where
        F: FnMut(StreamChunk) + Send + 'a,
    {
        Box::pin(async move { LlmClient::complete_streaming(self, request, callback).await })
    }

    fn complete<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { LlmClient::complete(self, request).await })
    }
}

/// Main-agent conversation runtime.
pub struct AgentLoop<L = LlmClient> {
    pub state: AgentState,
    pub llm: L,
    pub doc_store: Arc<RwLock<DocStore>>,
    pub doc_runner: DocRunner,
    pub policy: PolicyGate,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub budget: BudgetEngine,
    pub budget_tracker: BudgetTracker,
    pub messages: Vec<Message>,
    pub system_prompt: String,
    pub tool_definitions: Vec<Value>,
    pub subagent_pool: SubagentPool,

    model_name: String,
    layer_tokens: LayerTokenUsage,
    max_tool_rounds: usize,
}

/// Dependencies needed to construct an [`AgentLoop`].
pub struct AgentLoopParts<L> {
    pub llm: L,
    pub model_name: String,
    pub doc_store: Arc<RwLock<DocStore>>,
    pub doc_runner: DocRunner,
    pub policy: PolicyGate,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub budget: BudgetEngine,
    pub session: SessionInit,
    pub subagent_pool: SubagentPool,
}

impl AgentLoop<LlmClient> {
    /// Build a production loop from validated configuration.
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let llm = LlmClient::new(&cps_llm::ClientConfig::from_config(config))?;
        let doc_runner = DocRunner::new(config.doc_runner.clone());
        let doc_store = Arc::new(RwLock::new(DocStore::new()));
        let tokenizer: Arc<dyn Tokenizer> = Arc::from(create_tokenizer(&config.model.tokenizer)?);
        let budget = BudgetEngine::from_config(config);
        let policy = PolicyGate::new(policy_config_from_config(config, budget));
        let session = SessionInit::from_config(config);
        let subagent_pool = SubagentPool::new(
            config.subagents.max_parallel as usize,
            config.subagents.timeout_ms,
        );

        Ok(Self::new(AgentLoopParts {
            llm,
            model_name: config.model.model_name.clone(),
            doc_store,
            doc_runner,
            policy,
            tokenizer,
            budget,
            session,
            subagent_pool,
        }))
    }
}

impl<L> AgentLoop<L> {
    /// Build a loop from explicit dependencies.
    pub fn new(parts: AgentLoopParts<L>) -> Self {
        let tokenizer = parts.tokenizer;
        let budget = parts.budget;
        let system_prompt = parts.session.system_prompt;
        let tool_definitions = parts.session.tool_defs;
        let budget_tracker = BudgetTracker::new(
            budget,
            Arc::clone(&tokenizer),
            &system_prompt,
            &tool_definitions,
        );
        let layer_tokens = layer_token_usage_from_budget_status(&BudgetStatus {
            within_budget: true,
            layer_usage: budget_tracker.layer_usage.clone(),
            suggestions: Vec::new(),
        });

        Self {
            state: AgentState::Idle,
            llm: parts.llm,
            doc_store: parts.doc_store,
            doc_runner: parts.doc_runner,
            policy: parts.policy,
            tokenizer,
            budget,
            budget_tracker,
            messages: Vec::new(),
            system_prompt,
            tool_definitions,
            subagent_pool: parts.subagent_pool,
            model_name: parts.model_name,
            layer_tokens,
            max_tool_rounds: MAX_TOOL_ROUNDS,
        }
    }

    #[must_use]
    pub fn layer_token_usage(&self) -> LayerTokenUsage {
        self.layer_tokens
    }

    #[must_use]
    pub fn layer_budget(&self, layer: Layer) -> usize {
        self.budget.layer(layer)
    }
}

impl<L> AgentLoop<L>
where
    L: CompletionClient + Clone + 'static,
{
    /// Run one human input turn.
    ///
    /// The method appends the user message, calls the LLM with streaming,
    /// dispatches tool calls through the Rust runtime, compresses tool output
    /// into a Layer 2 evidence summary, and returns once a proposal or human
    /// input boundary is reached.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<TurnResult> {
        self.append_user_message(user_input);

        if is_cancel(user_input) {
            self.state = AgentState::Idle;
            return Ok(TurnResult {
                state: self.state,
                proposal: None,
                needs_input: true,
            });
        }

        self.state = AgentState::ReceiveIntent;

        for round in 0..self.max_tool_rounds {
            self.state = if round == 0 {
                AgentState::Plan
            } else {
                AgentState::MergeFindings
            };

            let request = self.chat_request()?;
            let response = self
                .llm
                .complete_streaming(&request, |chunk| {
                    if let Some(delta) = &chunk.delta_content {
                        tracing::trace!(bytes = delta.len(), "streaming assistant delta");
                    }
                })
                .await?;

            let assistant = response.message;
            self.append_assistant_message(assistant.clone());

            if let Some(tool_calls) = non_empty_tool_calls(&assistant) {
                if let Some(result) = self.handle_tool_round(tool_calls).await? {
                    return Ok(result);
                }
                continue;
            }

            if let Some(proposal) = parse_proposal_content(&assistant.content)? {
                self.state = AgentState::GenerateProposal;
                return self.finish_proposal(proposal);
            }

            self.state = AgentState::HumanReview;
            return Ok(TurnResult {
                state: self.state,
                proposal: None,
                needs_input: true,
            });
        }

        Err(AgentError::ToolRoundLimit {
            limit: self.max_tool_rounds,
        })
    }

    async fn handle_tool_round(&mut self, tool_calls: &[ToolCall]) -> Result<Option<TurnResult>> {
        self.state = AgentState::ExploreDocs;
        let mut observations = Vec::with_capacity(tool_calls.len());

        for call in tool_calls {
            let outcome = self.dispatch_tool_call(call).await;
            self.append_tool_result(&outcome.message);

            match outcome.control {
                ToolControl::Continue => observations.push(outcome.observation),
                ToolControl::Done => {
                    self.state = AgentState::Idle;
                    return Ok(Some(TurnResult {
                        state: self.state,
                        proposal: None,
                        needs_input: true,
                    }));
                }
                ToolControl::Proposal(proposal) => {
                    self.state = AgentState::GenerateProposal;
                    return self.finish_proposal(*proposal).map(Some);
                }
            }
        }

        self.append_evidence_summary(&observations);
        self.state = AgentState::MergeFindings;
        Ok(None)
    }

    async fn dispatch_tool_call(&self, call: &ToolCall) -> ToolOutcome {
        let name = call.function.name.as_str();

        if name == "done" {
            return ToolOutcome::done(call, "done");
        }

        if name == "propose_command" {
            return self.dispatch_propose_command(call);
        }

        if let Err(error) = self.validate_runtime_tool(call) {
            return ToolOutcome::error(call, name, error);
        }

        let result = match name {
            "read_help" => self.dispatch_read_help(call).await,
            "read_man" => self.dispatch_read_man(call).await,
            "read_info" => self.dispatch_read_info(call).await,
            "doc_token_count" => self.dispatch_doc_token_count(call),
            "doc_preview" => self.dispatch_doc_preview(call),
            "doc_grep" => self.dispatch_doc_grep(call),
            "doc_section" => self.dispatch_doc_section(call),
            "doc_lines" => self.dispatch_doc_lines(call),
            "doc_expand_around" => self.dispatch_doc_expand_around(call),
            "web_search" => Ok(json!({
                "ok": true,
                "message": "search not implemented yet",
                "results": [],
                "trust": "untrusted_web"
            })),
            "spawn" => self.dispatch_spawn(call).await,
            "execute" => Ok(json!({
                "ok": true,
                "message": "execution not implemented yet"
            })),
            other => Err(format!("unknown tool: {other}")),
        };

        match result {
            Ok(value) => ToolOutcome::success(call, name, value),
            Err(error) => ToolOutcome::error(call, name, error),
        }
    }

    fn dispatch_propose_command(&self, call: &ToolCall) -> ToolOutcome {
        let proposal = match serde_json::from_str::<CommandProposal>(&call.function.arguments) {
            Ok(proposal) => proposal,
            Err(error) => {
                return ToolOutcome::error(call, "propose_command", error.to_string());
            }
        };

        let findings = self.policy.check_proposal(&proposal);
        if contains_rejection(&findings) {
            return ToolOutcome::error(
                call,
                "propose_command",
                format!(
                    "proposal rejected by policy: {}",
                    finding_messages(&findings)
                ),
            );
        }

        ToolOutcome {
            message: Message::tool_result(
                &call.id,
                json_string(json!({
                    "ok": true,
                    "proposal": "accepted"
                })),
            ),
            observation: ToolObservation::success("propose_command", "proposal accepted"),
            control: ToolControl::Proposal(Box::new(proposal)),
        }
    }

    async fn dispatch_read_help(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: ReadHelpArgs = parse_args(call)?;
        let subcommands: Vec<&str> = args.subcommands.iter().map(String::as_str).collect();
        let store = self.doc_store_clone()?;
        let result = self
            .doc_runner
            .read_help(
                &args.program,
                &subcommands,
                args.style.into(),
                &store,
                self.tokenizer.as_ref(),
            )
            .await
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "ok": true,
            "doc_id": result.doc_id,
            "token_estimate": result.token_estimate,
            "preview": result.preview,
            "truncated": result.truncated
        }))
    }

    async fn dispatch_read_man(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: ReadManArgs = parse_args(call)?;
        let store = self.doc_store_clone()?;
        let result = self
            .doc_runner
            .read_man(&args.topic, args.section, &store, self.tokenizer.as_ref())
            .await
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "ok": true,
            "doc_id": result.doc_id,
            "token_estimate": result.token_estimate,
            "preview": result.preview,
            "truncated": result.truncated
        }))
    }

    async fn dispatch_read_info(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: ReadInfoArgs = parse_args(call)?;
        let store = self.doc_store_clone()?;
        let result = self
            .doc_runner
            .read_info(&args.topic, &store, self.tokenizer.as_ref())
            .await
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "ok": true,
            "doc_id": result.doc_id,
            "token_estimate": result.token_estimate,
            "preview": result.preview,
            "truncated": result.truncated
        }))
    }

    fn dispatch_doc_token_count(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocRefArgs = parse_args(call)?;
        let meta = self
            .doc_store_clone()?
            .doc_token_count(&args.doc)
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc_id": meta.doc_id,
            "token_estimate": meta.token_estimate,
            "line_count": meta.line_count,
            "byte_len": meta.byte_len,
            "source_kind": source_kind_name(meta.source_kind)
        }))
    }

    fn dispatch_doc_preview(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocPreviewArgs = parse_args(call)?;
        let text = self
            .doc_store_clone()?
            .doc_preview(&args.doc, args.max_tokens, self.tokenizer.as_ref())
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc": args.doc,
            "text": text
        }))
    }

    fn dispatch_doc_grep(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocGrepArgs = parse_args(call)?;
        let matches = self
            .doc_store_clone()?
            .doc_grep(
                &args.doc,
                &args.pattern,
                args.case_insensitive,
                args.context_lines,
                args.max_matches,
            )
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc": args.doc,
            "matches": grep_matches_json(&matches)
        }))
    }

    fn dispatch_doc_section(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocSectionArgs = parse_args(call)?;
        let text = self
            .doc_store_clone()?
            .doc_section(
                &args.doc,
                &args.heading_regex,
                args.max_tokens,
                self.tokenizer.as_ref(),
            )
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc": args.doc,
            "text": text
        }))
    }

    fn dispatch_doc_lines(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocLinesArgs = parse_args(call)?;
        let text = self
            .doc_store_clone()?
            .doc_lines(&args.doc, args.start, args.end)
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc": args.doc,
            "start": args.start,
            "end": args.end,
            "text": text
        }))
    }

    fn dispatch_doc_expand_around(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: DocExpandAroundArgs = parse_args(call)?;
        let text = self
            .doc_store_clone()?
            .doc_expand_around(&args.doc, &args.match_id, args.before, args.after)
            .map_err(format_doc_store_error)?;

        Ok(json!({
            "ok": true,
            "doc": args.doc,
            "match_id": args.match_id,
            "text": text
        }))
    }

    async fn dispatch_spawn(&self, call: &ToolCall) -> std::result::Result<Value, String> {
        let args: SpawnArgs = parse_args(call)?;

        // Parse the role
        let role = match args.role.as_str() {
            "doc_explorer" => SubagentRole::DocExplorer,
            "risk_reviewer" => SubagentRole::RiskReviewer,
            _ => return Err(format!("invalid role: {}", args.role)),
        };

        // Create the spawn request
        let request = SpawnRequest {
            role,
            goal: args.goal,
            input: args.input,
            allowed_tools: args.allowed_tools,
            context_tokens: args.context_tokens,
            thinking_budget: args.thinking_budget,
            timeout_ms: args.timeout_ms,
            output_schema: args.output_schema,
        };

        // Execute the subagent
        match self
            .subagent_pool
            .spawn(
                request,
                &self.llm,
                &self.model_name,
                self.budget.subagent_context(),
                Arc::clone(&self.tokenizer),
            )
            .await
        {
            Ok(result) => Ok(json!({
                "ok": true,
                "subagent_id": result.subagent_id,
                "status": result.status,
                "findings": result.findings,
                "candidate_fragments": result.candidate_fragments,
                "open_questions": result.open_questions
            })),
            Err(error) => Err(error.to_string()),
        }
    }

    fn finish_proposal(&mut self, proposal: CommandProposal) -> Result<TurnResult> {
        self.state = AgentState::PolicyCheck;
        let findings = self.policy.check_proposal(&proposal);
        if contains_rejection(&findings) {
            return Err(AgentError::ProposalRejected {
                messages: finding_messages(&findings),
            });
        }

        self.state = AgentState::HumanReview;
        Ok(TurnResult {
            state: self.state,
            proposal: Some(proposal),
            needs_input: true,
        })
    }

    fn validate_runtime_tool(&self, call: &ToolCall) -> std::result::Result<(), String> {
        if !requires_policy_validation(call.function.name.as_str()) {
            return Ok(());
        }

        let policy_call =
            PolicyToolCall::from_json_arguments(&call.function.name, &call.function.arguments)
                .map_err(|error| error.to_string())?;
        let lookup = DocStoreLookup {
            store: self.doc_store_clone()?,
        };
        self.policy
            .validate_tool_call(&policy_call, &lookup)
            .map_err(format_policy_error)
    }

    fn append_user_message(&mut self, content: &str) {
        self.messages.push(Message::user(content));
        self.refresh_budget_usage();
    }

    fn append_assistant_message(&mut self, message: Message) {
        self.messages.push(message);
        self.refresh_budget_usage();
    }

    fn append_tool_result(&mut self, message: &Message) {
        self.messages.push(message.clone());
        self.refresh_budget_usage();
    }

    fn append_evidence_summary(&mut self, observations: &[ToolObservation]) {
        if observations.is_empty() {
            return;
        }

        let summary = evidence_summary(observations);
        self.messages.push(Message::assistant(summary));
        self.refresh_budget_usage();
    }

    fn chat_request(&mut self) -> Result<ChatRequest> {
        self.enforce_budget_before_llm_request()?;
        Ok(ChatRequest {
            model: self.model_name.clone(),
            messages: self.request_messages(),
            tools: Some(self.tool_definitions.clone()),
            max_completion_tokens: Some(self.budget.thinking_budget()),
            stream: true,
            temperature: None,
        })
    }

    fn request_messages(&self) -> Vec<Message> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.push(Message::system(self.system_prompt.clone()));
        messages.extend(self.messages.clone());
        messages
    }

    fn enforce_budget_before_llm_request(&mut self) -> Result<()> {
        let mut status = self.refresh_budget_usage();
        log_budget_suggestions(&status);
        if status.within_budget {
            return Ok(());
        }

        tracing::warn!("conversation token budget exceeded; compressing Layer 3 tool output");
        while !status.within_budget && self.compress_oldest_layer3_content() {
            status = self.refresh_budget_usage();
        }

        if status.within_budget {
            return Ok(());
        }

        tracing::warn!(
            "conversation token budget still exceeded after Layer 3 compression; truncating oldest Layer 2 evidence"
        );
        while !status.within_budget && self.truncate_oldest_evidence() {
            status = self.refresh_budget_usage();
        }

        if !status.within_budget {
            log_budget_suggestions(&status);
            return Err(AgentError::BudgetExceeded {
                messages: status.suggestions.join("; "),
            });
        }

        Ok(())
    }

    fn refresh_budget_usage(&mut self) -> BudgetStatus {
        let status = self.budget_tracker.check_budget(&self.messages);
        self.layer_tokens = layer_token_usage_from_budget_status(&status);
        status
    }

    fn compress_oldest_layer3_content(&mut self) -> bool {
        for message in &mut self.messages {
            if classify_message_layer(message) != Layer::Layer3TempOutput
                || message
                    .content
                    .starts_with("Layer 3 compressed tool output:")
            {
                continue;
            }

            let original_tokens = count_message_tokens(self.tokenizer.as_ref(), message);
            let compressed =
                compressed_layer3_content(self.tokenizer.as_ref(), self.budget, message);
            let mut candidate = message.clone();
            candidate.content = compressed;
            let compressed_tokens = count_message_tokens(self.tokenizer.as_ref(), &candidate);
            if compressed_tokens < original_tokens {
                *message = candidate;
                return true;
            }
        }
        false
    }

    fn truncate_oldest_evidence(&mut self) -> bool {
        for message in &mut self.messages {
            if !is_evidence_message(message) || message.content == truncated_evidence_content() {
                continue;
            }

            let original_tokens = count_message_tokens(self.tokenizer.as_ref(), message);
            let mut candidate = message.clone();
            candidate.content = truncated_evidence_content().to_owned();
            let truncated_tokens = count_message_tokens(self.tokenizer.as_ref(), &candidate);
            if truncated_tokens < original_tokens {
                *message = candidate;
                return true;
            }
        }
        false
    }

    fn doc_store_clone(&self) -> std::result::Result<DocStore, String> {
        self.doc_store
            .read()
            .map(|store| store.clone())
            .map_err(|_| AgentError::LockPoisoned("doc_store").to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolObservation {
    tool: String,
    ok: bool,
    detail: String,
}

impl ToolObservation {
    fn success(tool: &str, detail: impl Into<String>) -> Self {
        Self {
            tool: tool.to_owned(),
            ok: true,
            detail: detail.into(),
        }
    }

    fn error(tool: &str, detail: impl Into<String>) -> Self {
        Self {
            tool: tool.to_owned(),
            ok: false,
            detail: detail.into(),
        }
    }
}

#[derive(Debug)]
struct ToolOutcome {
    message: Message,
    observation: ToolObservation,
    control: ToolControl,
}

impl ToolOutcome {
    fn success(call: &ToolCall, tool: &str, value: Value) -> Self {
        Self {
            message: Message::tool_result(&call.id, json_string(value.clone())),
            observation: ToolObservation::success(tool, summarize_tool_value(tool, &value)),
            control: ToolControl::Continue,
        }
    }

    fn error(call: &ToolCall, tool: &str, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            message: Message::tool_result(
                &call.id,
                json_string(json!({
                    "ok": false,
                    "error": error
                })),
            ),
            observation: ToolObservation::error(tool, error),
            control: ToolControl::Continue,
        }
    }

    fn done(call: &ToolCall, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        Self {
            message: Message::tool_result(
                &call.id,
                json_string(json!({
                    "ok": true,
                    "done": true
                })),
            ),
            observation: ToolObservation::success("done", detail),
            control: ToolControl::Done,
        }
    }
}

#[derive(Debug)]
enum ToolControl {
    Continue,
    Done,
    Proposal(Box<CommandProposal>),
}

struct DocStoreLookup {
    store: DocStore,
}

impl DocLookup for DocStoreLookup {
    fn contains_doc(&self, doc_id: &str) -> bool {
        self.store.doc_token_count(doc_id).is_ok()
    }
}

#[derive(Debug, Deserialize)]
struct ReadHelpArgs {
    program: String,
    #[serde(default)]
    subcommands: Vec<String>,
    #[serde(default)]
    style: HelpStyleArg,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HelpStyleArg {
    #[default]
    Long,
    Short,
}

impl From<HelpStyleArg> for HelpStyle {
    fn from(value: HelpStyleArg) -> Self {
        match value {
            HelpStyleArg::Long => Self::Long,
            HelpStyleArg::Short => Self::Short,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadManArgs {
    topic: String,
    section: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct ReadInfoArgs {
    topic: String,
}

#[derive(Debug, Deserialize)]
struct DocRefArgs {
    #[serde(alias = "doc_id")]
    doc: String,
}

#[derive(Debug, Deserialize)]
struct DocPreviewArgs {
    #[serde(alias = "doc_id")]
    doc: String,
    max_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct DocGrepArgs {
    #[serde(alias = "doc_id")]
    doc: String,
    pattern: String,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default = "default_context_lines")]
    context_lines: usize,
    #[serde(default = "default_max_matches")]
    max_matches: usize,
}

#[derive(Debug, Deserialize)]
struct DocSectionArgs {
    #[serde(alias = "doc_id")]
    doc: String,
    heading_regex: String,
    max_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct DocLinesArgs {
    #[serde(alias = "doc_id")]
    doc: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Deserialize)]
struct DocExpandAroundArgs {
    #[serde(alias = "doc_id")]
    doc: String,
    match_id: String,
    before: usize,
    after: usize,
}

#[derive(Debug, Deserialize)]
struct SpawnArgs {
    role: String,
    goal: String,
    input: Value,
    allowed_tools: Vec<String>,
    context_tokens: usize,
    thinking_budget: Option<u32>,
    timeout_ms: u64,
    output_schema: String,
}

fn default_context_lines() -> usize {
    2
}

fn default_max_matches() -> usize {
    20
}

fn parse_args<T>(call: &ToolCall) -> std::result::Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(&call.function.arguments).map_err(|error| error.to_string())
}

fn non_empty_tool_calls(message: &Message) -> Option<&[ToolCall]> {
    message
        .tool_calls
        .as_deref()
        .filter(|calls| !calls.is_empty())
}

fn parse_proposal_content(content: &str) -> Result<Option<CommandProposal>> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    match serde_json::from_str::<CommandProposal>(trimmed) {
        Ok(proposal) => Ok(Some(proposal)),
        Err(error) if trimmed.starts_with('{') => Err(error.into()),
        Err(_) => Ok(None),
    }
}

fn is_cancel(input: &str) -> bool {
    matches!(input.trim(), "/cancel" | "/Cancel" | "cancel" | "Cancel")
}

fn requires_policy_validation(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_help"
            | "doc_token_count"
            | "doc_preview"
            | "doc_grep"
            | "doc_section"
            | "doc_lines"
            | "doc_expand_around"
            | "web_search"
            | "spawn"
            | "execute"
    )
}

fn contains_rejection(findings: &[PolicyFinding]) -> bool {
    findings
        .iter()
        .any(|finding| finding.severity == PolicyFindingSeverity::Reject)
}

fn finding_messages(findings: &[PolicyFinding]) -> String {
    findings
        .iter()
        .map(|finding| finding.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_policy_error(error: PolicyError) -> String {
    error.to_string()
}

fn format_doc_store_error(error: DocStoreError) -> String {
    error.to_string()
}

fn source_kind_name(source_kind: SourceKind) -> &'static str {
    match source_kind {
        SourceKind::LocalSchema => "local_schema",
        SourceKind::LocalDoc => "local_doc",
        SourceKind::UntrustedWeb => "untrusted_web",
    }
}

fn grep_matches_json(matches: &[GrepMatch]) -> Value {
    Value::Array(
        matches
            .iter()
            .map(|hit| {
                json!({
                    "match_id": hit.match_id,
                    "line_number": hit.line_number,
                    "line_text": hit.line_text,
                    "context_before": hit.context_before,
                    "context_after": hit.context_after
                })
            })
            .collect(),
    )
}

fn summarize_tool_value(tool: &str, value: &Value) -> String {
    match tool {
        "read_help" | "read_man" | "read_info" => {
            let doc_id = value
                .get("doc_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let tokens = value
                .get("token_estimate")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            format!("stored {doc_id} ({tokens} tokens)")
        }
        "doc_token_count" => {
            let doc_id = value
                .get("doc_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let tokens = value
                .get("token_estimate")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            format!("{doc_id} has {tokens} tokens")
        }
        "doc_grep" => {
            let count = value
                .get("matches")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            format!("{count} matches")
        }
        "doc_preview" | "doc_section" | "doc_lines" | "doc_expand_around" => {
            let bytes = value
                .get("text")
                .and_then(Value::as_str)
                .map_or(0, str::len);
            format!("returned {bytes} bytes")
        }
        "web_search" => "search not implemented yet".to_owned(),
        "spawn" => {
            let status = value
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let findings_count = value
                .get("findings")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            format!("subagent {} ({} findings)", status, findings_count)
        }
        "execute" => "execution not implemented yet".to_owned(),
        _ => "completed".to_owned(),
    }
}

fn evidence_summary(observations: &[ToolObservation]) -> String {
    let mut summary = String::from("Evidence summary (Layer 2):\n");
    for observation in observations {
        let status = if observation.ok { "ok" } else { "error" };
        summary.push_str("- ");
        summary.push_str(&observation.tool);
        summary.push_str(": ");
        summary.push_str(status);
        summary.push_str(" - ");
        summary.push_str(&observation.detail);
        summary.push('\n');
    }
    summary
}

fn layer_token_usage_from_budget_status(status: &BudgetStatus) -> LayerTokenUsage {
    LayerTokenUsage {
        layer1_conversation: layer_tokens(&status.layer_usage, Layer::Layer1Conversation),
        layer2_evidence: layer_tokens(&status.layer_usage, Layer::Layer2Evidence),
        layer3_temp_output: layer_tokens(&status.layer_usage, Layer::Layer3TempOutput),
    }
}

fn log_budget_suggestions(status: &BudgetStatus) {
    for suggestion in &status.suggestions {
        tracing::warn!(suggestion = %suggestion, "conversation token budget pressure");
    }
}

fn compressed_layer3_content(
    tokenizer: &dyn Tokenizer,
    budget: BudgetEngine,
    message: &Message,
) -> String {
    let original_tokens = count_message_tokens(tokenizer, message);
    let preview_budget = original_tokens
        .saturating_div(4)
        .min(budget.layer(Layer::Layer3TempOutput).saturating_div(20))
        .max(1);
    let preview = token_limited_prefix(&message.content, preview_budget, tokenizer);
    format!("Layer 3 compressed tool output: original_tokens={original_tokens}; preview={preview}")
}

fn token_limited_prefix(text: &str, max_tokens: usize, tokenizer: &dyn Tokenizer) -> String {
    if max_tokens == 0 || text.is_empty() {
        return String::new();
    }
    if tokenizer.count_tokens(text) <= max_tokens {
        return text.to_owned();
    }

    let char_indices: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
    let mut low = 0;
    let mut high = char_indices.len();
    while low < high {
        let mid = (low + high).div_ceil(2);
        let end = if mid == char_indices.len() {
            text.len()
        } else {
            char_indices[mid]
        };
        if tokenizer.count_tokens(&text[..end]) <= max_tokens {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    let end = if low == char_indices.len() {
        text.len()
    } else {
        char_indices[low]
    };
    text[..end].to_owned()
}

fn truncated_evidence_content() -> &'static str {
    "Evidence summary (Layer 2):\n- truncated: oldest evidence removed to restore token budget\n"
}

fn json_string(value: Value) -> String {
    serde_json::to_string(&value).unwrap_or_else(|error| {
        format!(
            "{{\"ok\":false,\"error\":\"failed to serialize tool result: {}\"}}",
            error
        )
    })
}

fn policy_config_from_config(config: &Config, budget: BudgetEngine) -> PolicyConfig {
    let mut role_tool_permissions = HashMap::new();
    role_tool_permissions.insert(
        DOC_EXPLORER_ROLE.to_owned(),
        vec![
            "read_help".to_owned(),
            "read_man".to_owned(),
            "read_info".to_owned(),
            "doc_token_count".to_owned(),
            "doc_preview".to_owned(),
            "doc_grep".to_owned(),
            "doc_section".to_owned(),
            "doc_lines".to_owned(),
            "doc_expand_around".to_owned(),
            "web_search".to_owned(),
        ],
    );
    role_tool_permissions.insert(
        RISK_REVIEWER_ROLE.to_owned(),
        vec![
            "doc_token_count".to_owned(),
            "doc_preview".to_owned(),
            "doc_grep".to_owned(),
            "doc_section".to_owned(),
            "doc_lines".to_owned(),
            "doc_expand_around".to_owned(),
        ],
    );

    PolicyConfig {
        allow_programs: config.doc_runner.allow_programs.clone(),
        allowed_roles: vec![DOC_EXPLORER_ROLE.to_owned(), RISK_REVIEWER_ROLE.to_owned()],
        role_tool_permissions,
        search_enabled: config.search.default_enabled,
        execute_enabled: config.approval.execute_enabled,
        max_subagent_context: budget.subagent_context(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use cps_config::DocRunnerConfig;
    use cps_doc_store::SourceKind;
    use cps_llm::{FunctionCall, Role, ToolCall};
    use cps_policy::PolicyConfig;
    use cps_proposal::{Confidence, Evidence, EvidenceKind, EvidenceSource, PreflightCmd, Risk};
    use cps_tokenizer::FallbackTokenizer;

    use super::*;
    use crate::ToolFeatureFlags;

    #[derive(Clone)]
    struct MockLlm {
        responses: Arc<Mutex<VecDeque<ChatResponse>>>,
        requests: Arc<Mutex<Vec<ChatRequest>>>,
    }

    impl MockLlm {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into())),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl CompletionClient for MockLlm {
        fn complete_streaming<'a, F>(
            &'a self,
            request: &'a ChatRequest,
            _callback: F,
        ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>>
        where
            F: FnMut(StreamChunk) + Send + 'a,
        {
            Box::pin(async move {
                self.requests
                    .lock()
                    .expect("mock request lock")
                    .push(request.clone());
                let response = self
                    .responses
                    .lock()
                    .expect("mock response lock")
                    .pop_front()
                    .expect("mock response queued");
                Ok(response)
            })
        }

        fn complete<'a>(
            &'a self,
            request: &'a ChatRequest,
        ) -> Pin<Box<dyn Future<Output = cps_llm::Result<ChatResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.requests
                    .lock()
                    .expect("mock request lock")
                    .push(request.clone());
                let response = self
                    .responses
                    .lock()
                    .expect("mock response lock")
                    .pop_front()
                    .expect("mock response queued");
                Ok(response)
            })
        }
    }

    fn loop_with(responses: Vec<ChatResponse>) -> AgentLoop<MockLlm> {
        loop_with_budget(responses, BudgetEngine::new(100_000, 10_000))
    }

    fn loop_with_budget(responses: Vec<ChatResponse>, budget: BudgetEngine) -> AgentLoop<MockLlm> {
        let doc_store = Arc::new(RwLock::new(DocStore::new()));
        let doc_runner = DocRunner::new(DocRunnerConfig {
            allow_programs: vec!["kubectl".to_owned()],
            sandbox: "true".to_owned(),
            timeout_ms: 5_000,
            max_output_bytes: 16 * 1024,
            allow_doc_actions: vec!["help".to_owned(), "man".to_owned(), "info".to_owned()],
            extra_bind_ro: Vec::new(),
        });
        let policy = PolicyGate::new(test_policy_config());
        let session = SessionInit {
            system_prompt: "system prompt".to_owned(),
            tool_defs: crate::tools::build(ToolFeatureFlags {
                search_enabled: true,
                subagents_enabled: true,
            }),
        };

        AgentLoop::new(AgentLoopParts {
            llm: MockLlm::new(responses),
            model_name: "test-model".to_owned(),
            doc_store,
            doc_runner,
            policy,
            tokenizer: Arc::new(FallbackTokenizer::new()),
            budget,
            session,
            subagent_pool: SubagentPool::new(2, 60000),
        })
    }

    fn test_policy_config() -> PolicyConfig {
        let mut role_tool_permissions = HashMap::new();
        role_tool_permissions.insert(
            DOC_EXPLORER_ROLE.to_owned(),
            vec![
                "read_help".to_owned(),
                "doc_token_count".to_owned(),
                "doc_preview".to_owned(),
                "doc_grep".to_owned(),
                "doc_lines".to_owned(),
            ],
        );
        PolicyConfig {
            allow_programs: vec!["kubectl".to_owned()],
            allowed_roles: vec![DOC_EXPLORER_ROLE.to_owned()],
            role_tool_permissions,
            search_enabled: true,
            execute_enabled: false,
            max_subagent_context: 4_000,
        }
    }

    fn response(message: Message) -> ChatResponse {
        ChatResponse {
            message,
            usage: None,
            finish_reason: None,
        }
    }

    fn assistant_tool(name: &str, arguments: Value) -> Message {
        Message {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall {
                id: format!("call_{name}"),
                call_type: "function".to_owned(),
                function: FunctionCall {
                    name: name.to_owned(),
                    arguments: serde_json::to_string(&arguments).expect("arguments serialize"),
                },
            }]),
        }
    }

    fn assistant_tool_with_id(id: &str, name: &str, arguments: Value) -> Message {
        Message {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: Some(vec![ToolCall {
                id: id.to_owned(),
                call_type: "function".to_owned(),
                function: FunctionCall {
                    name: name.to_owned(),
                    arguments: serde_json::to_string(&arguments).expect("arguments serialize"),
                },
            }]),
        }
    }

    fn sample_proposal() -> CommandProposal {
        CommandProposal {
            summary: "List pods".to_owned(),
            argv: vec![
                "kubectl".to_owned(),
                "--context".to_owned(),
                "prod".to_owned(),
                "-n".to_owned(),
                "payments".to_owned(),
                "get".to_owned(),
                "pods".to_owned(),
            ],
            display: "kubectl --context prod -n payments get pods".to_owned(),
            risk: Risk::Low,
            risk_reasons: Vec::new(),
            assumptions: Vec::new(),
            preflight: vec![PreflightCmd {
                argv: vec![
                    "kubectl".to_owned(),
                    "--context".to_owned(),
                    "prod".to_owned(),
                    "-n".to_owned(),
                    "payments".to_owned(),
                    "get".to_owned(),
                    "pods".to_owned(),
                ],
                reason: "same read-only command".to_owned(),
            }],
            rollback: None,
            evidence: vec![
                Evidence {
                    claim: "kubectl get is read-only".to_owned(),
                    source: EvidenceSource {
                        kind: EvidenceKind::LocalDoc,
                        doc: "help:kubectl:".to_owned(),
                        lines: Some((1, 3)),
                    },
                    confidence: Confidence::High,
                },
                Evidence {
                    claim: "--context selects the cluster context".to_owned(),
                    source: EvidenceSource {
                        kind: EvidenceKind::LocalDoc,
                        doc: "help:kubectl:".to_owned(),
                        lines: Some((4, 6)),
                    },
                    confidence: Confidence::High,
                },
                Evidence {
                    claim: "-n selects the namespace".to_owned(),
                    source: EvidenceSource {
                        kind: EvidenceKind::LocalDoc,
                        doc: "help:kubectl:".to_owned(),
                        lines: Some((7, 9)),
                    },
                    confidence: Confidence::High,
                },
            ],
            missing_confirmations: Vec::new(),
        }
    }

    #[tokio::test]
    async fn state_machine_transitions_to_human_review_for_proposal_content() {
        let proposal = sample_proposal();
        let mut agent = loop_with(vec![response(Message::assistant(
            serde_json::to_string(&proposal).expect("proposal serialize"),
        ))]);

        let result = agent.run_turn("list pods").await.expect("turn ok");

        assert_eq!(result.state, AgentState::HumanReview);
        assert_eq!(agent.state, AgentState::HumanReview);
        assert_eq!(result.proposal, Some(proposal));
        assert!(result.needs_input);
    }

    #[tokio::test]
    async fn tool_dispatch_routes_read_help_and_then_done() {
        let mut agent = loop_with(vec![
            response(assistant_tool(
                "read_help",
                json!({
                    "program": "kubectl",
                    "subcommands": ["get"],
                    "style": "long"
                }),
            )),
            response(assistant_tool("done", json!({}))),
        ]);

        let result = agent
            .run_turn("read kubectl get help")
            .await
            .expect("turn ok");

        assert_eq!(result.state, AgentState::Idle);
        let tool_outputs: Vec<&Message> = agent
            .messages
            .iter()
            .filter(|message| message.role == Role::Tool)
            .collect();
        assert!(
            tool_outputs
                .iter()
                .any(|message| message.content.contains("\"doc_id\":\"help:kubectl:get\"")),
            "read_help should route through DocRunner and store a help doc"
        );
    }

    #[tokio::test]
    async fn doc_store_tool_dispatch_reads_existing_doc() {
        let mut agent = loop_with(vec![
            response(assistant_tool(
                "doc_grep",
                json!({
                    "doc": "d1",
                    "pattern": "restart",
                    "case_insensitive": false,
                    "context_lines": 0,
                    "max_matches": 10
                }),
            )),
            response(assistant_tool("done", json!({}))),
        ]);
        agent.doc_store.read().expect("doc store lock").insert(
            "d1",
            "rollout restart\nget pods\n",
            SourceKind::LocalDoc,
            agent.tokenizer.as_ref(),
        );

        let result = agent.run_turn("grep restart").await.expect("turn ok");

        assert_eq!(result.state, AgentState::Idle);
        assert!(agent
            .messages
            .iter()
            .any(|message| message.role == Role::Tool
                && message.content.contains("\"match_id\":\"d1:L1:I0\"")));
    }

    #[tokio::test]
    async fn evidence_compression_appends_summary_without_rewriting_history() {
        let mut agent = loop_with(vec![
            response(assistant_tool(
                "web_search",
                json!({
                    "query": "kubectl docs",
                    "max_results": 1
                }),
            )),
            response(assistant_tool("done", json!({}))),
        ]);

        let before = agent.messages.clone();
        let result = agent.run_turn("search docs").await.expect("turn ok");

        assert_eq!(result.state, AgentState::Idle);
        assert_eq!(&agent.messages[..before.len()], before.as_slice());
        assert!(agent
            .messages
            .iter()
            .any(|message| message.role == Role::Assistant
                && message.content.starts_with("Evidence summary (Layer 2):")
                && message.content.contains("web_search: ok")));
        assert!(agent.layer_token_usage().layer2_evidence > 0);
    }

    #[tokio::test]
    async fn conversation_is_append_only_across_turns() {
        let proposal = sample_proposal();
        let mut agent = loop_with(vec![response(Message::assistant(
            serde_json::to_string(&proposal).expect("proposal serialize"),
        ))]);

        agent.messages.push(Message::assistant("prior summary"));
        let before = agent.messages.clone();

        let _ = agent.run_turn("list pods").await.expect("turn ok");

        assert_eq!(&agent.messages[..before.len()], before.as_slice());
        assert!(agent.messages.len() > before.len());
    }

    #[tokio::test]
    async fn policy_gate_rejection_becomes_error_tool_result() {
        let mut agent = loop_with(vec![
            response(assistant_tool_with_id(
                "bad_help",
                "read_help",
                json!({
                    "program": "terraform"
                }),
            )),
            response(assistant_tool("done", json!({}))),
        ]);

        let result = agent
            .run_turn("read terraform help")
            .await
            .expect("turn ok");

        assert_eq!(result.state, AgentState::Idle);
        assert!(agent
            .messages
            .iter()
            .any(|message| message.role == Role::Tool
                && message.tool_call_id.as_deref() == Some("bad_help")
                && message.content.contains("\"ok\":false")
                && message.content.contains("program not allowed")));
    }

    #[tokio::test]
    async fn user_cancel_returns_to_idle_without_calling_llm() {
        let mut agent = loop_with(Vec::new());
        agent.state = AgentState::HumanReview;

        let result = agent.run_turn("/cancel").await.expect("cancel ok");

        assert_eq!(result.state, AgentState::Idle);
        assert_eq!(agent.state, AgentState::Idle);
        assert!(result.needs_input);
        assert_eq!(agent.messages.len(), 1);
    }

    #[tokio::test]
    async fn propose_command_tool_call_returns_proposal() {
        let proposal = sample_proposal();
        let mut agent = loop_with(vec![response(assistant_tool(
            "propose_command",
            serde_json::to_value(&proposal).expect("proposal value"),
        ))]);

        let result = agent.run_turn("list pods").await.expect("turn ok");

        assert_eq!(result.state, AgentState::HumanReview);
        assert_eq!(result.proposal, Some(proposal));
        assert!(agent
            .messages
            .iter()
            .any(|message| message.role == Role::Tool
                && message.content.contains("\"proposal\":\"accepted\"")));
    }

    #[test]
    fn layer3_compression_reduces_token_count() {
        let mut agent = loop_with_budget(Vec::new(), BudgetEngine::new(20_000, 2_000));
        let message = Message::tool_result("call_1", "x".repeat(40_000));
        let original_tokens = count_message_tokens(agent.tokenizer.as_ref(), &message);
        agent.messages.push(message);

        assert!(agent.compress_oldest_layer3_content());

        let compressed = agent
            .messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .expect("compressed tool message exists");
        let compressed_tokens = count_message_tokens(agent.tokenizer.as_ref(), compressed);
        assert!(compressed
            .content
            .starts_with("Layer 3 compressed tool output:"));
        assert!(compressed_tokens < original_tokens);
    }

    #[tokio::test]
    async fn budget_enforcement_compresses_raw_tool_output_before_followup_llm_call() {
        let mut agent = loop_with_budget(
            vec![
                response(assistant_tool(
                    "doc_lines",
                    json!({
                        "doc": "d1",
                        "start": 1,
                        "end": 2
                    }),
                )),
                response(assistant_tool("done", json!({}))),
            ],
            BudgetEngine::new(20_000, 2_000),
        );
        agent.doc_store.read().expect("doc store lock").insert(
            "d1",
            "x".repeat(40_000),
            SourceKind::LocalDoc,
            agent.tokenizer.as_ref(),
        );

        let result = agent.run_turn("read a large line").await.expect("turn ok");

        assert_eq!(result.state, AgentState::Idle);
        let requests = agent.llm.requests.lock().expect("mock request lock");
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|message| {
            message.role == Role::Tool
                && message
                    .content
                    .starts_with("Layer 3 compressed tool output:")
        }));
    }
}

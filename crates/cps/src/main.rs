use std::future::Future;
use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cps_agent::{AgentLoop, TurnResult};
use cps_config::{
    ApprovalConfig, Config, DocRunnerConfig, DocStoreConfig, ExecutionRunnerConfig, ModelConfig,
    ProposalConfig, RiskConfig, RuntimeConfig, SearchConfig, SubagentsConfig, ThinkingConfig,
    TokenizerConfig,
};
use cps_proposal::{CommandProposal, EvidenceKind, Risk};
use tokio::io::{AsyncBufReadExt, BufReader, Lines, Stdin};

#[derive(Parser, Debug)]
#[command(
    name = "cps",
    version,
    about = "cmd-proposer — sandboxed argv proposer for production ops",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// Override config file path.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Disable web search for this invocation.
    #[arg(long, conflicts_with = "online")]
    offline: bool,

    /// Enable web search for this invocation.
    #[arg(long, conflicts_with = "offline")]
    online: bool,

    #[command(subcommand)]
    cmd: Option<Command>,

    /// Non-interactive intent. Quote it, or pass multiple words and cps will join them.
    #[arg(value_name = "INTENT", num_args = 0.., trailing_var_arg = true, allow_hyphen_values = true)]
    intent: Vec<String>,
}

#[derive(Subcommand, Debug, PartialEq, Eq)]
enum Command {
    /// Print a config template to stdout. Pipe into ./.cmd-proposer.yaml to bootstrap.
    Init {
        /// Print the full reference (all settings shown with defaults commented out).
        #[arg(long)]
        full: bool,
    },
}

impl Cli {
    fn intent_text(&self) -> Option<String> {
        let intent = self.intent.join(" ");
        let trimmed = intent.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    if let Some(Command::Init { full }) = cli.cmd {
        let template = if full {
            cps_config::full_template()
        } else {
            cps_config::minimal_template()
        };
        print!("{template}");
        return Ok(());
    }

    let config = load_runtime_config(&cli)?;
    let mut agent = AgentLoop::from_config(&config).context("failed to initialize agent loop")?;

    if let Some(intent) = cli.intent_text() {
        let mut stdout = io::stdout();
        run_non_interactive(&mut agent, &intent, &mut stdout).await?;
        return Ok(());
    }

    run_interactive(agent, &config).await
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .without_time()
        .try_init();
}

trait TurnRunner {
    fn run_turn_boxed<'a>(
        &'a mut self,
        user_input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult>> + 'a>>;
}

impl TurnRunner for AgentLoop {
    fn run_turn_boxed<'a>(
        &'a mut self,
        user_input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult>> + 'a>> {
        Box::pin(async move { self.run_turn(user_input).await.map_err(anyhow::Error::from) })
    }
}

async fn run_non_interactive<R, W>(runner: &mut R, intent: &str, output: &mut W) -> Result<()>
where
    R: TurnRunner,
    W: IoWrite,
{
    let result = runner.run_turn_boxed(intent).await?;
    let proposal = result
        .proposal
        .context("agent turn completed without a command proposal")?;
    writeln!(output, "{}", format_proposal_plain(&proposal))?;
    Ok(())
}

async fn run_interactive(mut agent: AgentLoop, config: &Config) -> Result<()> {
    println!("cps {}", env!("CARGO_PKG_VERSION"));
    println!("Enter an intent. Ctrl+C or q exits.");

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    loop {
        print!("cps> ");
        io::stdout().flush()?;

        let Some(input) = read_line_or_interrupt(&mut lines).await? else {
            println!();
            return Ok(());
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "q" {
            return Ok(());
        }

        let Some(result) = run_turn_or_interrupt(&mut agent, input).await? else {
            println!();
            return Ok(());
        };

        if let Some(proposal) = result.proposal {
            println!("{}", format_proposal_plain(&proposal));
            match prompt_review(&mut lines).await? {
                ReviewChoice::Accept => {
                    if config.approval.execute_enabled {
                        println!("Accepted. Execution runner is not available in v0.1; command was not executed.");
                    } else {
                        println!("Accepted. Execution is disabled; command was not executed.");
                    }
                }
                ReviewChoice::Reject => {
                    println!("Rejected. Enter a revised intent.");
                }
                ReviewChoice::Quit => return Ok(()),
            }
        } else if result.needs_input {
            println!("No command proposal yet. Enter more detail or q to quit.");
        }
    }
}

async fn read_line_or_interrupt(lines: &mut Lines<BufReader<Stdin>>) -> Result<Option<String>> {
    tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            signal.context("failed to listen for Ctrl+C")?;
            Ok(None)
        }
        line = lines.next_line() => {
            line.context("failed to read stdin")
        }
    }
}

async fn run_turn_or_interrupt(agent: &mut AgentLoop, input: &str) -> Result<Option<TurnResult>> {
    tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            signal.context("failed to listen for Ctrl+C")?;
            Ok(None)
        }
        result = agent.run_turn(input) => {
            Ok(Some(result?))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewChoice {
    Accept,
    Reject,
    Quit,
}

async fn prompt_review(lines: &mut Lines<BufReader<Stdin>>) -> Result<ReviewChoice> {
    loop {
        print!("Accept proposal? [y/n/q] ");
        io::stdout().flush()?;

        let Some(input) = read_line_or_interrupt(lines).await? else {
            return Ok(ReviewChoice::Quit);
        };
        match input.trim() {
            "y" | "Y" => return Ok(ReviewChoice::Accept),
            "n" | "N" => return Ok(ReviewChoice::Reject),
            "q" | "Q" => return Ok(ReviewChoice::Quit),
            _ => println!("Enter y, n, or q."),
        }
    }
}

fn format_proposal_plain(proposal: &CommandProposal) -> String {
    use std::fmt::Write as FmtWrite;

    let mut output = String::new();
    let (risk, marker) = risk_display(proposal.risk);

    let _ = writeln!(output, "═══ Command Proposal ═══");
    let _ = writeln!(output, "Risk: [{risk}] {marker}");
    let _ = writeln!(output);
    let _ = writeln!(output, "$ {}", proposal.display_command());
    let _ = writeln!(output);
    let _ = writeln!(output, "Summary: {}", proposal.summary);
    let _ = writeln!(output);
    push_evidence(&mut output, proposal);
    let _ = writeln!(output);
    push_assumptions(&mut output, proposal);

    output.trim_end().to_owned()
}

fn risk_display(risk: Risk) -> (&'static str, &'static str) {
    match risk {
        Risk::Low => ("LOW", "✓"),
        Risk::Medium => ("MEDIUM", "!"),
        Risk::High => ("HIGH", "!"),
        Risk::Critical => ("CRITICAL", "!"),
    }
}

fn push_evidence(output: &mut String, proposal: &CommandProposal) {
    use std::fmt::Write as FmtWrite;

    let _ = writeln!(output, "Evidence:");
    if proposal.evidence.is_empty() {
        let _ = writeln!(output, "  • none");
        return;
    }

    for item in &proposal.evidence {
        let _ = writeln!(
            output,
            "  • [{}] {}",
            evidence_kind_label(item.source.kind),
            item.claim
        );
    }
}

fn push_assumptions(output: &mut String, proposal: &CommandProposal) {
    use std::fmt::Write as FmtWrite;

    let _ = writeln!(output, "Assumptions:");
    if proposal.assumptions.is_empty() {
        let _ = writeln!(output, "  • none");
        return;
    }

    for assumption in &proposal.assumptions {
        let _ = writeln!(output, "  • {assumption}");
    }
}

fn evidence_kind_label(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::OtherWeb => "other_web",
        EvidenceKind::OfficialWebDoc => "official_web",
        EvidenceKind::LocalDoc => "local_doc",
        EvidenceKind::LocalSchema => "local_schema",
    }
}

fn load_runtime_config(cli: &Cli) -> Result<Config> {
    load_runtime_config_with_paths(cli, &ConfigPaths::default(), &ProcessEnv)
}

fn load_runtime_config_with_paths(
    cli: &Cli,
    paths: &ConfigPaths,
    env: &dyn EnvLookup,
) -> Result<Config> {
    let mut config = match resolve_config_path(cli, paths, env) {
        Some(path) => {
            let (config, _) = cps_config::load_from_path(&path)
                .with_context(|| format!("failed to load config from `{}`", path.display()))?;
            config
        }
        None => default_config(env),
    };

    apply_env_overrides(&mut config, env);
    apply_cli_overrides(&mut config, cli);
    config.validate()?;
    Ok(config)
}

#[derive(Debug, Clone)]
struct ConfigPaths {
    local: PathBuf,
    user: Option<PathBuf>,
}

impl Default for ConfigPaths {
    fn default() -> Self {
        Self {
            local: PathBuf::from(".cmd-proposer.yaml"),
            user: dirs::config_dir().map(|dir| dir.join("cmd-proposer").join("config.yaml")),
        }
    }
}

fn resolve_config_path(cli: &Cli, paths: &ConfigPaths, env: &dyn EnvLookup) -> Option<PathBuf> {
    if let Some(path) = &cli.config {
        return Some(path.clone());
    }
    if let Some(path) = first_non_empty(env, &["CMD_PROPOSER_CONFIG", "CPS_CONFIG"]) {
        return Some(PathBuf::from(path));
    }
    if paths.local.exists() {
        return Some(paths.local.clone());
    }
    paths.user.as_ref().filter(|path| path.exists()).cloned()
}

trait EnvLookup {
    fn get(&self, key: &str) -> Option<String>;
}

struct ProcessEnv;

impl EnvLookup for ProcessEnv {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

fn default_config(env: &dyn EnvLookup) -> Config {
    Config {
        version: 1,
        model: ModelConfig {
            base_url: first_non_empty(
                env,
                &["CMD_PROPOSER_MODEL_BASE_URL", "LOCAL_ROUTER_BASEURL"],
            )
            .unwrap_or_else(|| "http://localhost:8317/v1".to_owned()),
            api_key: first_value(env, &["CMD_PROPOSER_MODEL_API_KEY", "LOCAL_ROUTER_API_KEY"])
                .unwrap_or_default(),
            model_name: first_non_empty(env, &["CMD_PROPOSER_MODEL_NAME", "LLM_MODEL"])
                .unwrap_or_else(|| "qwen3.6-27b".to_owned()),
            provider: "local_openai_compatible".to_owned(),
            max_context_tokens: 200_000,
            tokenizer: TokenizerConfig {
                path: first_non_empty(env, &["CMD_PROPOSER_TOKENIZER_PATH"])
                    .unwrap_or_else(|| "~/.local/share/models/qwen/tokenizer.json".to_owned()),
                tokenizer_type: "huggingface_tokenizer_json".to_owned(),
            },
        },
        thinking: ThinkingConfig::default(),
        runtime: RuntimeConfig::default(),
        search: SearchConfig::default(),
        doc_runner: DocRunnerConfig {
            allow_programs: env_list(env, "CMD_PROPOSER_DOC_RUNNER_ALLOW_PROGRAMS")
                .unwrap_or_else(default_allow_programs),
            sandbox: "bwrap".to_owned(),
            timeout_ms: 60_000,
            max_output_bytes: 10_485_760,
            allow_doc_actions: vec!["help".to_owned(), "man".to_owned(), "info".to_owned()],
            extra_bind_ro: Vec::new(),
        },
        doc_store: DocStoreConfig::default(),
        subagents: SubagentsConfig::default(),
        proposal: ProposalConfig::default(),
        approval: ApprovalConfig::default(),
        execution_runner: ExecutionRunnerConfig::default(),
        risk: RiskConfig::default(),
    }
}

fn apply_env_overrides(config: &mut Config, env: &dyn EnvLookup) {
    if let Some(value) = first_non_empty(
        env,
        &["CMD_PROPOSER_MODEL_BASE_URL", "LOCAL_ROUTER_BASEURL"],
    ) {
        config.model.base_url = value;
    }
    if let Some(value) = first_value(env, &["CMD_PROPOSER_MODEL_API_KEY", "LOCAL_ROUTER_API_KEY"]) {
        config.model.api_key = value;
    }
    if let Some(value) = first_non_empty(env, &["CMD_PROPOSER_MODEL_NAME", "LLM_MODEL"]) {
        config.model.model_name = value;
    }
    if let Some(value) = first_non_empty(env, &["CMD_PROPOSER_TOKENIZER_PATH"]) {
        config.model.tokenizer.path = value;
    }
    if let Some(value) = env_list(env, "CMD_PROPOSER_DOC_RUNNER_ALLOW_PROGRAMS") {
        config.doc_runner.allow_programs = value;
    }
    if let Some(value) = env_bool(env, "CMD_PROPOSER_SEARCH_DEFAULT_ENABLED") {
        config.search.default_enabled = value;
    }
}

fn apply_cli_overrides(config: &mut Config, cli: &Cli) {
    if cli.offline {
        config.search.default_enabled = false;
    }
    if cli.online {
        config.search.default_enabled = true;
    }
}

fn first_value(env: &dyn EnvLookup, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| env.get(key))
}

fn first_non_empty(env: &dyn EnvLookup, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| env.get(key))
        .find(|value| !value.trim().is_empty())
}

fn env_list(env: &dyn EnvLookup, key: &str) -> Option<Vec<String>> {
    env.get(key).and_then(|value| {
        let values: Vec<String> = value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        (!values.is_empty()).then_some(values)
    })
}

fn env_bool(env: &dyn EnvLookup, key: &str) -> Option<bool> {
    let value = env.get(key)?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn default_allow_programs() -> Vec<String> {
    ["kubectl", "helm", "terraform", "systemctl", "journalctl"]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use cps_proposal::{Confidence, Evidence, EvidenceSource};

    #[test]
    fn missing_config_file_returns_defaults() {
        let cli = Cli::try_parse_from(["cps"]).expect("parse");
        let paths = ConfigPaths {
            local: PathBuf::from("/definitely/missing/.cmd-proposer.yaml"),
            user: Some(PathBuf::from(
                "/definitely/missing/cmd-proposer/config.yaml",
            )),
        };
        let env = MapEnv::default();

        let config =
            load_runtime_config_with_paths(&cli, &paths, &env).expect("default config loads");

        assert_eq!(config.model.base_url, "http://localhost:8317/v1");
        assert_eq!(config.model.model_name, "qwen3.6-27b");
        assert!(config
            .doc_runner
            .allow_programs
            .contains(&"kubectl".to_owned()));
        config.validate().expect("defaults validate");
    }

    #[test]
    fn cli_arg_parsing_works() {
        let init = Cli::try_parse_from(["cps", "init", "--full"]).expect("parse init");
        assert_eq!(init.cmd, Some(Command::Init { full: true }));

        let run = Cli::try_parse_from([
            "cps",
            "--config",
            "custom.yaml",
            "--offline",
            "kubectl",
            "get",
            "pods",
        ])
        .expect("parse run");

        assert_eq!(run.config, Some(PathBuf::from("custom.yaml")));
        assert!(run.offline);
        assert_eq!(run.intent_text().as_deref(), Some("kubectl get pods"));
    }

    #[tokio::test]
    async fn non_interactive_mode_exits_after_one_turn() {
        let proposal = sample_proposal();
        let mut runner = FakeRunner {
            calls: 0,
            result: TurnResult {
                state: cps_agent::AgentState::HumanReview,
                proposal: Some(proposal),
                needs_input: true,
            },
        };
        let mut output = Vec::new();

        run_non_interactive(&mut runner, "list pods", &mut output)
            .await
            .expect("non-interactive run");

        assert_eq!(runner.calls, 1);
        let text = String::from_utf8(output).expect("utf8");
        assert!(text.contains("═══ Command Proposal ═══"));
        assert!(text.contains("$ kubectl get pods -n default"));
    }

    #[test]
    fn offline_and_online_override_config_search_flag() {
        let mut env = MapEnv::default();
        env.set("CMD_PROPOSER_SEARCH_DEFAULT_ENABLED", "false");
        let cli = Cli::try_parse_from(["cps", "--online"]).expect("parse");
        let paths = ConfigPaths {
            local: PathBuf::from("/definitely/missing/.cmd-proposer.yaml"),
            user: None,
        };

        let config = load_runtime_config_with_paths(&cli, &paths, &env).expect("load");

        assert!(config.search.default_enabled);
    }

    #[derive(Default)]
    struct MapEnv {
        values: HashMap<String, String>,
    }

    impl MapEnv {
        fn set(&mut self, key: &str, value: &str) {
            self.values.insert(key.to_owned(), value.to_owned());
        }
    }

    impl EnvLookup for MapEnv {
        fn get(&self, key: &str) -> Option<String> {
            self.values.get(key).cloned()
        }
    }

    struct FakeRunner {
        calls: usize,
        result: TurnResult,
    }

    impl TurnRunner for FakeRunner {
        fn run_turn_boxed<'a>(
            &'a mut self,
            _user_input: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<TurnResult>> + 'a>> {
            self.calls += 1;
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }
    }

    fn sample_proposal() -> CommandProposal {
        CommandProposal {
            summary: "列出 default 命名空间下的所有 pods".to_owned(),
            argv: vec![
                "kubectl".to_owned(),
                "get".to_owned(),
                "pods".to_owned(),
                "-n".to_owned(),
                "default".to_owned(),
            ],
            display: "kubectl get pods -n default".to_owned(),
            risk: Risk::Low,
            risk_reasons: Vec::new(),
            assumptions: vec!["Current kubeconfig context is correct".to_owned()],
            preflight: Vec::new(),
            rollback: None,
            evidence: vec![Evidence {
                claim: "kubectl help confirms 'get pods' syntax".to_owned(),
                source: EvidenceSource {
                    kind: EvidenceKind::LocalDoc,
                    doc: "kubectl-help".to_owned(),
                    lines: None,
                },
                confidence: Confidence::High,
            }],
            missing_confirmations: Vec::new(),
        }
    }
}

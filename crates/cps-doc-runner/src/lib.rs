//! bwrap sandbox runner for executing `--help`/`man`/`info` documentation commands.
//!
//! The runner is intentionally narrow: it accepts structured tool arguments,
//! validates them before process construction, runs only argv vectors inside
//! a networkless bwrap sandbox, and stores captured output in [`DocStore`] as
//! [`SourceKind::LocalDoc`].
//!
//! For programs that support `__schema --format=json`, the runner can also
//! retrieve a structured [`CliSchema`] (see [`schema`] module) before falling
//! back to `--help` exploration. Schema evidence is tagged as
//! [`SourceKind::LocalSchema`] — the highest trust tier.

pub mod schema;

pub use schema::{ArgSchema, CliSchema, FlagSchema, FlagType, SubcommandSchema};

use std::ffi::OsString;
use std::io;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use cps_config::DocRunnerConfig;
use cps_doc_store::{DocStore, SourceKind};
use cps_tokenizer::Tokenizer;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;
use tokio::task::JoinHandle;
use tokio::time::timeout;

const PREVIEW_TOKEN_LIMIT: usize = 200;
const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
const DOC_ACTION_HELP: &str = "help";
const DOC_ACTION_MAN: &str = "man";
const DOC_ACTION_INFO: &str = "info";
const DEFAULT_RO_BINDS: &[&str] = &[
    "/usr",
    "/bin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
    "/usr/share/man",
    "/usr/local/share/man",
    "/usr/share/info",
    "/usr/local/share/info",
];

pub type Result<T> = std::result::Result<T, DocRunnerError>;

#[derive(Debug, Clone)]
pub struct DocRunner {
    config: DocRunnerConfig,
}

impl DocRunner {
    pub fn new(config: DocRunnerConfig) -> Self {
        Self { config }
    }

    pub async fn read_help(
        &self,
        program: &str,
        subcommands: &[&str],
        style: HelpStyle,
        doc_store: &DocStore,
        tokenizer: &dyn Tokenizer,
    ) -> Result<DocResult> {
        self.ensure_action_allowed(DOC_ACTION_HELP)?;
        self.ensure_program_allowed(program)?;
        for subcommand in subcommands {
            validate_subcommand(subcommand)?;
        }

        let mut argv = Vec::with_capacity(subcommands.len() + 2);
        argv.push(program.to_string());
        argv.extend(subcommands.iter().map(|s| (*s).to_string()));
        argv.push(style.flag().to_string());

        self.capture_to_store(
            help_doc_id(program, subcommands),
            &argv,
            doc_store,
            tokenizer,
        )
        .await
    }

    pub async fn read_man(
        &self,
        topic: &str,
        section: Option<u8>,
        doc_store: &DocStore,
        tokenizer: &dyn Tokenizer,
    ) -> Result<DocResult> {
        self.ensure_action_allowed(DOC_ACTION_MAN)?;
        validate_topic(topic)?;

        let mut argv = Vec::with_capacity(3);
        argv.push("man".to_string());
        if let Some(section) = section {
            argv.push(section.to_string());
        }
        argv.push(topic.to_string());

        self.capture_to_store(man_doc_id(topic, section), &argv, doc_store, tokenizer)
            .await
    }

    pub async fn read_info(
        &self,
        topic: &str,
        doc_store: &DocStore,
        tokenizer: &dyn Tokenizer,
    ) -> Result<DocResult> {
        self.ensure_action_allowed(DOC_ACTION_INFO)?;
        validate_topic(topic)?;

        let argv = vec!["info".to_string(), topic.to_string()];
        self.capture_to_store(info_doc_id(topic), &argv, doc_store, tokenizer)
            .await
    }

    /// Attempt to retrieve a structured CLI schema from `program` by running
    /// `<program> __schema --format=json` inside the bwrap sandbox.
    ///
    /// Returns `Some(CliSchema)` on success. If the command fails or the
    /// output is not valid schema JSON, returns `None` — the caller should
    /// fall back to `--help`/`man`/`info` exploration.
    ///
    /// When a schema is successfully parsed, its JSON representation is also
    /// stored in `doc_store` under doc_id `schema:<program>` with
    /// [`SourceKind::LocalSchema`] (highest trust).
    pub async fn try_schema(
        &self,
        program: &str,
        doc_store: &DocStore,
        tokenizer: &dyn Tokenizer,
    ) -> Option<schema::CliSchema> {
        self.ensure_program_allowed(program).ok()?;

        let argv = vec![
            program.to_string(),
            "__schema".to_string(),
            "--format=json".to_string(),
        ];

        let captured = match self.run_in_bwrap(&argv).await {
            Ok(output) => output,
            Err(error) => {
                tracing::debug!(
                    program = %program,
                    error = %error,
                    "schema retrieval failed; will fall back to --help"
                );
                return None;
            }
        };

        let parsed = schema::parse_schema_json(&captured.text)?;

        // Store the raw JSON as LocalSchema evidence in the doc store.
        let doc_id = schema::schema_doc_id(program);
        doc_store.insert(doc_id, captured.text, SourceKind::LocalSchema, tokenizer);

        tracing::info!(
            program = %program,
            version = %parsed.version,
            subcommands = parsed.subcommands.len(),
            "parsed CLI schema successfully"
        );

        Some(parsed)
    }

    /// Build the std command used by tests and by callers that need to inspect
    /// the exact sandbox argv without spawning it.
    pub fn build_bwrap_command(&self, argv: &[String]) -> Command {
        let mut command = Command::new(&self.config.sandbox);
        command.args(self.build_bwrap_args(argv));
        command
    }

    fn ensure_program_allowed(&self, program: &str) -> Result<()> {
        if self
            .config
            .allow_programs
            .iter()
            .any(|p| p == "*" || p == program)
        {
            return Ok(());
        }
        Err(DocRunnerError::ProgramNotAllowed(program.to_string()))
    }

    fn ensure_action_allowed(&self, action: &str) -> Result<()> {
        if self
            .config
            .allow_doc_actions
            .iter()
            .any(|allowed| allowed == action)
        {
            return Ok(());
        }
        Err(DocRunnerError::ActionNotAllowed(action.to_string()))
    }

    async fn capture_to_store(
        &self,
        doc_id: String,
        argv: &[String],
        doc_store: &DocStore,
        tokenizer: &dyn Tokenizer,
    ) -> Result<DocResult> {
        let captured = self.run_in_bwrap(argv).await?;
        let preview = preview_text(&captured.text, tokenizer, PREVIEW_TOKEN_LIMIT);
        let meta = doc_store.insert(doc_id, captured.text, SourceKind::LocalDoc, tokenizer);

        Ok(DocResult {
            doc_id: meta.doc_id,
            token_estimate: meta.token_estimate,
            preview,
            truncated: captured.truncated,
        })
    }

    async fn run_in_bwrap(&self, argv: &[String]) -> Result<CapturedOutput> {
        let max_output_bytes = self.max_output_bytes()?;
        let mut command = self.build_tokio_bwrap_command(argv);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        tracing::debug!(argv = ?argv, "running doc command inside bwrap sandbox");

        let mut child = command.spawn().map_err(map_spawn_error)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| DocRunnerError::ProcessFailed {
                exit_code: None,
                stderr: "failed to capture child stdout".to_string(),
            })?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| DocRunnerError::ProcessFailed {
                exit_code: None,
                stderr: "failed to capture child stderr".to_string(),
            })?;

        let stdout_task = tokio::spawn(read_limited(stdout, max_output_bytes));
        let stderr_task = tokio::spawn(read_limited(stderr, max_output_bytes));

        let status =
            match timeout(Duration::from_millis(self.config.timeout_ms), child.wait()).await {
                Ok(Ok(status)) => status,
                Ok(Err(error)) => {
                    abort_reader(stdout_task);
                    abort_reader(stderr_task);
                    return Err(DocRunnerError::ProcessFailed {
                        exit_code: None,
                        stderr: error.to_string(),
                    });
                }
                Err(_) => {
                    abort_reader(stdout_task);
                    abort_reader(stderr_task);
                    if let Err(error) = child.kill().await {
                        tracing::warn!(error = %error, "failed to kill timed-out doc command");
                    }
                    return Err(DocRunnerError::Timeout);
                }
            };

        let stdout = join_reader(stdout_task).await?;
        let stderr = join_reader(stderr_task).await?;

        if !status.success() {
            let stderr = select_preferred(stderr, stdout).text;
            return Err(DocRunnerError::ProcessFailed {
                exit_code: status.code(),
                stderr,
            });
        }

        Ok(select_output(stdout, stderr))
    }

    fn build_tokio_bwrap_command(&self, argv: &[String]) -> TokioCommand {
        let mut command = TokioCommand::new(&self.config.sandbox);
        command.args(self.build_bwrap_args(argv));
        command
    }

    fn build_bwrap_args(&self, argv: &[String]) -> Vec<OsString> {
        let mut args = Vec::new();

        push_arg(&mut args, "--unshare-all");
        push_arg(&mut args, "--unshare-net");
        push_arg(&mut args, "--new-session");
        push_arg(&mut args, "--die-with-parent");
        push_arg(&mut args, "--clearenv");

        for path in DEFAULT_RO_BINDS {
            if Path::new(path).exists() {
                push_ro_bind(&mut args, path);
            }
        }
        for path in &self.config.extra_bind_ro {
            if Path::new(path).exists() {
                push_ro_bind(&mut args, path);
            } else {
                tracing::warn!(
                    path = %path,
                    "skipping missing extra doc runner read-only bind path"
                );
            }
        }

        push_arg(&mut args, "--tmpfs");
        push_arg(&mut args, "/tmp");

        push_env(&mut args, "PATH", DEFAULT_PATH);
        push_env(&mut args, "HOME", "/tmp");
        push_env(&mut args, "PAGER", "cat");
        push_env(&mut args, "MANPAGER", "cat");
        push_env(&mut args, "LESSSECURE", "1");

        args.extend(argv.iter().map(OsString::from));
        args
    }

    fn max_output_bytes(&self) -> Result<usize> {
        usize::try_from(self.config.max_output_bytes)
            .map_err(|_| DocRunnerError::OutputTooLarge(usize::MAX))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocResult {
    pub doc_id: String,
    pub token_estimate: usize,
    pub preview: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpStyle {
    Long,
    Short,
}

impl HelpStyle {
    fn flag(self) -> &'static str {
        match self {
            Self::Long => "--help",
            Self::Short => "-h",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DocRunnerError {
    #[error("program is not allowed for doc runner: {0}")]
    ProgramNotAllowed(String),

    #[error("doc action is not allowed for doc runner: {0}")]
    ActionNotAllowed(String),

    #[error("invalid subcommand: {0}")]
    InvalidSubcommand(String),

    #[error("invalid doc topic: {0}")]
    InvalidTopic(String),

    #[error("bwrap executable not found")]
    BwrapNotFound,

    #[error("doc command timed out")]
    Timeout,

    #[error("doc command failed with exit code {exit_code:?}: {stderr}")]
    ProcessFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("output is too large to handle on this platform: {0} bytes")]
    OutputTooLarge(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedOutput {
    text: String,
    truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LimitedBytes {
    bytes: Vec<u8>,
    truncated: bool,
}

impl LimitedBytes {
    fn has_output(&self) -> bool {
        !self.bytes.is_empty() || self.truncated
    }
}

fn push_arg(args: &mut Vec<OsString>, arg: &str) {
    args.push(OsString::from(arg));
}

fn push_ro_bind(args: &mut Vec<OsString>, path: &str) {
    push_arg(args, "--ro-bind");
    args.push(OsString::from(path));
    args.push(OsString::from(path));
}

fn push_env(args: &mut Vec<OsString>, key: &str, value: &str) {
    push_arg(args, "--setenv");
    args.push(OsString::from(key));
    args.push(OsString::from(value));
}

fn validate_subcommand(subcommand: &str) -> Result<()> {
    if is_valid_identifier(subcommand) {
        return Ok(());
    }
    Err(DocRunnerError::InvalidSubcommand(subcommand.to_string()))
}

fn validate_topic(topic: &str) -> Result<()> {
    if !topic.is_empty()
        && topic
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_'))
    {
        return Ok(());
    }
    Err(DocRunnerError::InvalidTopic(topic.to_string()))
}

fn is_valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

fn help_doc_id(program: &str, subcommands: &[&str]) -> String {
    format!("help:{program}:{}", subcommands.join(":"))
}

fn man_doc_id(topic: &str, section: Option<u8>) -> String {
    let section = section
        .map(|s| s.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!("man:{topic}:{section}")
}

fn info_doc_id(topic: &str) -> String {
    format!("info:{topic}")
}

async fn read_limited<R>(mut reader: R, byte_limit: usize) -> io::Result<LimitedBytes>
where
    R: AsyncRead + Unpin,
{
    let mut out = Vec::new();
    let mut truncated = false;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        let remaining = byte_limit.saturating_sub(out.len());
        if remaining == 0 {
            truncated = true;
            continue;
        }
        let kept = remaining.min(n);
        out.extend_from_slice(&buf[..kept]);
        if kept < n {
            truncated = true;
        }
    }

    Ok(LimitedBytes {
        bytes: out,
        truncated,
    })
}

async fn join_reader(handle: JoinHandle<io::Result<LimitedBytes>>) -> Result<LimitedBytes> {
    match handle.await {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(error)) => Err(DocRunnerError::ProcessFailed {
            exit_code: None,
            stderr: error.to_string(),
        }),
        Err(error) => Err(DocRunnerError::ProcessFailed {
            exit_code: None,
            stderr: error.to_string(),
        }),
    }
}

fn abort_reader(handle: JoinHandle<io::Result<LimitedBytes>>) {
    handle.abort();
}

fn map_spawn_error(error: io::Error) -> DocRunnerError {
    if error.kind() == io::ErrorKind::NotFound {
        return DocRunnerError::BwrapNotFound;
    }
    DocRunnerError::ProcessFailed {
        exit_code: None,
        stderr: error.to_string(),
    }
}

fn select_output(stdout: LimitedBytes, stderr: LimitedBytes) -> CapturedOutput {
    let output = select_preferred(stdout, stderr);
    if output.truncated {
        tracing::warn!("doc runner output exceeded byte limit; storing truncated output");
    }
    output
}

fn select_preferred(primary: LimitedBytes, fallback: LimitedBytes) -> CapturedOutput {
    let selected = if primary.has_output() {
        primary
    } else {
        fallback
    };
    CapturedOutput {
        text: String::from_utf8_lossy(&selected.bytes).into_owned(),
        truncated: selected.truncated,
    }
}

fn preview_text(text: &str, tokenizer: &dyn Tokenizer, max_tokens: usize) -> String {
    if max_tokens == 0 || text.is_empty() {
        return String::new();
    }
    if tokenizer.count_tokens(text) <= max_tokens {
        return text.to_string();
    }

    let mut preview = String::new();
    let mut used = 0usize;
    for line in text.split_inclusive('\n') {
        let token_count = tokenizer
            .count_tokens(line)
            .max(if line.is_empty() { 0 } else { 1 });
        if used + token_count > max_tokens {
            break;
        }
        preview.push_str(line);
        used += token_count;
    }

    if !preview.is_empty() {
        return preview;
    }

    for ch in text.chars() {
        preview.push(ch);
        if tokenizer.count_tokens(&preview) > max_tokens {
            preview.pop();
            break;
        }
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::*;
    use cps_tokenizer::FallbackTokenizer;

    fn config() -> DocRunnerConfig {
        DocRunnerConfig {
            allow_programs: vec!["kubectl".to_string(), "helm".to_string()],
            sandbox: "bwrap".to_string(),
            timeout_ms: 60_000,
            max_output_bytes: 16,
            allow_doc_actions: vec!["help".to_string(), "man".to_string(), "info".to_string()],
            extra_bind_ro: vec![existing_extra_bind()],
        }
    }

    fn existing_extra_bind() -> String {
        std::env::current_dir()
            .expect("current directory")
            .to_string_lossy()
            .into_owned()
    }

    fn args(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn build_bwrap_command_produces_expected_argv() {
        let runner = DocRunner::new(config());
        let command = runner.build_bwrap_command(&[
            "kubectl".to_string(),
            "rollout".to_string(),
            "--help".to_string(),
        ]);

        assert_eq!(command.get_program(), "bwrap");
        let args = args(&command);
        assert!(args.starts_with(&[
            "--unshare-all".to_string(),
            "--unshare-net".to_string(),
            "--new-session".to_string(),
            "--die-with-parent".to_string(),
            "--clearenv".to_string(),
        ]));
        assert!(contains_sequence(&args, &["--ro-bind", "/usr", "/usr"]));
        for path in DEFAULT_RO_BINDS {
            assert_eq!(
                contains_sequence(&args, &["--ro-bind", path, path]),
                Path::new(path).exists(),
                "default bind path {path} should match host existence"
            );
        }
        let extra_bind = existing_extra_bind();
        assert!(contains_sequence(
            &args,
            &["--ro-bind", extra_bind.as_str(), extra_bind.as_str()]
        ));
        assert!(contains_sequence(&args, &["--tmpfs", "/tmp"]));
        assert!(contains_sequence(
            &args,
            &["--setenv", "PATH", DEFAULT_PATH]
        ));
        assert!(contains_sequence(&args, &["--setenv", "HOME", "/tmp"]));
        assert!(contains_sequence(&args, &["--setenv", "PAGER", "cat"]));
        assert!(contains_sequence(&args, &["--setenv", "MANPAGER", "cat"]));
        assert!(contains_sequence(&args, &["--setenv", "LESSSECURE", "1"]));
        assert!(args.ends_with(&[
            "kubectl".to_string(),
            "rollout".to_string(),
            "--help".to_string(),
        ]));
    }

    #[test]
    fn build_bwrap_command_skips_missing_extra_bind_paths() {
        let missing = "/definitely/missing/cps-doc-runner-extra-bind";
        assert!(!Path::new(missing).exists());

        let mut cfg = config();
        cfg.extra_bind_ro = vec![missing.to_string()];
        let runner = DocRunner::new(cfg);
        let command = runner.build_bwrap_command(&["kubectl".to_string(), "--help".to_string()]);
        let args = args(&command);

        assert!(!contains_sequence(&args, &["--ro-bind", missing, missing]));
    }

    #[test]
    fn subcommand_validation_rejects_shell_metacharacters() {
        for subcommand in [
            "rollout;rm",
            "rollout|restart",
            "rollout&restart",
            "rollout/restart",
            "rollout restart",
        ] {
            assert!(matches!(
                validate_subcommand(subcommand),
                Err(DocRunnerError::InvalidSubcommand(_))
            ));
        }
    }

    #[test]
    fn subcommand_validation_accepts_valid_identifiers() {
        for subcommand in ["rollout", "restart_foo", "v1-beta", "A1_B2-C3"] {
            validate_subcommand(subcommand).expect("valid subcommand");
        }
    }

    #[test]
    fn topic_validation_rejects_shell_metacharacters() {
        for topic in [
            "kubectl;rm",
            "kubectl|less",
            "kubectl&less",
            "kubectl/rollout",
            "kubectl rollout",
        ] {
            assert!(matches!(
                validate_topic(topic),
                Err(DocRunnerError::InvalidTopic(_))
            ));
        }
    }

    #[test]
    fn topic_validation_accepts_valid_topics() {
        for topic in ["kubectl", "kubectl-rollout", "systemd.service", "foo_bar.1"] {
            validate_topic(topic).expect("valid topic");
        }
    }

    #[test]
    fn program_allow_list_check_works() {
        let runner = DocRunner::new(config());

        runner.ensure_program_allowed("kubectl").expect("allowed");
        assert!(matches!(
            runner.ensure_program_allowed("sh"),
            Err(DocRunnerError::ProgramNotAllowed(program)) if program == "sh"
        ));
    }

    #[tokio::test]
    async fn read_methods_reject_disallowed_doc_actions() {
        let store = DocStore::new();
        let tokenizer = FallbackTokenizer::new();

        let mut cfg = config();
        cfg.allow_doc_actions = vec!["man".to_string(), "info".to_string()];
        let runner = DocRunner::new(cfg);
        assert!(matches!(
            runner
                .read_help("kubectl", &[], HelpStyle::Long, &store, &tokenizer)
                .await,
            Err(DocRunnerError::ActionNotAllowed(action)) if action == "help"
        ));

        let mut cfg = config();
        cfg.allow_doc_actions = vec!["help".to_string()];
        let runner = DocRunner::new(cfg);
        assert!(matches!(
            runner.read_man("kubectl", None, &store, &tokenizer).await,
            Err(DocRunnerError::ActionNotAllowed(action)) if action == "man"
        ));
        assert!(matches!(
            runner.read_info("kubectl", &store, &tokenizer).await,
            Err(DocRunnerError::ActionNotAllowed(action)) if action == "info"
        ));
    }

    #[test]
    fn doc_id_generation_is_deterministic() {
        assert_eq!(
            help_doc_id("kubectl", &["rollout", "restart"]),
            "help:kubectl:rollout:restart"
        );
        assert_eq!(help_doc_id("helm", &[]), "help:helm:");
        assert_eq!(man_doc_id("systemctl", Some(1)), "man:systemctl:1");
        assert_eq!(man_doc_id("systemctl", None), "man:systemctl:none");
        assert_eq!(info_doc_id("coreutils"), "info:coreutils");
    }

    #[test]
    fn output_truncates_at_max_output_bytes() {
        let output = select_output(
            limited_bytes(b"abcdefghij".to_vec(), true),
            limited_bytes(Vec::new(), false),
        );

        assert_eq!(output.text, "abcdefghij");
        assert!(output.truncated);
    }

    #[tokio::test]
    async fn read_limited_truncates_at_max_output_bytes() {
        let input = std::io::Cursor::new(b"abcdefghijklmnopqrstuvwxyz".as_slice());
        let output = read_limited(input, 10).await.expect("read");

        assert_eq!(output.bytes, b"abcdefghij");
        assert!(output.truncated);
    }

    #[test]
    fn stderr_is_used_when_stdout_is_empty() {
        let output = select_output(
            limited_bytes(Vec::new(), false),
            limited_bytes(b"stderr text".to_vec(), false),
        );

        assert_eq!(output.text, "stderr text");
        assert!(!output.truncated);
    }

    #[test]
    fn truncated_empty_stdout_still_counts_as_stdout() {
        let output = select_output(
            limited_bytes(Vec::new(), true),
            limited_bytes(b"stderr text".to_vec(), false),
        );

        assert_eq!(output.text, "");
        assert!(output.truncated);
    }

    fn contains_sequence(args: &[String], needle: &[&str]) -> bool {
        args.windows(needle.len())
            .any(|window| window.iter().map(String::as_str).eq(needle.iter().copied()))
    }

    fn limited_bytes(bytes: Vec<u8>, truncated: bool) -> LimitedBytes {
        LimitedBytes { bytes, truncated }
    }
}

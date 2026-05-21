//! Argv-only execution runner with audit logging; disabled by default.
//!
//! All commands are `Vec<String>` argv executed via `Command::new(argv[0]).args(&argv[1..])`.
//! Shell interpretation (`sh -c`) is never used.

use cps_proposal::Risk;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("execution runner is disabled; set runtime.exec_enabled = true to enable")]
    Disabled,

    #[error("empty argv: nothing to execute")]
    EmptyArgv,

    #[error("approval required for {0:?} risk command")]
    NotApproved(Risk),

    #[error("high-risk command requires explicit confirmation string \"yes\", got {0:?}")]
    HighRiskNoConfirmation(Option<String>),

    #[error("critical-risk command requires typing the full command as confirmation")]
    CriticalRiskNoConfirmation,

    #[error("critical-risk confirmation does not match the command")]
    CriticalRiskMismatch,

    #[error("process spawn failed: {0}")]
    Spawn(#[from] std::io::Error),

    #[error("audit log write failed: {0}")]
    AuditWrite(String),
}

// ---------------------------------------------------------------------------
// ExecResult
// ---------------------------------------------------------------------------

/// Outcome of a command execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
    pub timed_out: bool,
}

// ---------------------------------------------------------------------------
// AuditEntry (JSONL)
// ---------------------------------------------------------------------------

/// Single audit log record, serialized as one JSON line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub argv: Vec<String>,
    pub risk: String,
    pub approved: bool,
    pub executed: bool,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    pub timed_out: bool,
}

// ---------------------------------------------------------------------------
// ExecutionRunner
// ---------------------------------------------------------------------------

/// Argv-only execution runner with risk-gated approval and audit logging.
///
/// When `enabled` is false (the default), every call to [`execute`](Self::execute)
/// returns [`ExecError::Disabled`].
pub struct ExecutionRunner {
    enabled: bool,
    timeout: Duration,
    audit_log_path: PathBuf,
}

impl ExecutionRunner {
    /// Create a new runner.
    ///
    /// * `enabled` -- from config, default `false`.
    /// * `timeout` -- from `runtime.global_timeout_ms`, converted to `Duration`.
    /// * `audit_log_path` -- if `None`, defaults to
    ///   `~/.local/state/cmd-proposer/audit.log`.
    pub fn new(enabled: bool, timeout: Duration, audit_log_path: Option<PathBuf>) -> Self {
        let audit_log_path = audit_log_path.unwrap_or_else(default_audit_log_path);
        Self {
            enabled,
            timeout,
            audit_log_path,
        }
    }

    /// Execute an argv command after risk-gated approval validation.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError`] when the runner is disabled, the argv is empty,
    /// approval is insufficient for the given risk level, or execution fails.
    pub fn execute(
        &self,
        argv: &[String],
        risk: Risk,
        approved: bool,
        confirmation: Option<&str>,
    ) -> Result<ExecResult, ExecError> {
        if !self.enabled {
            return Err(ExecError::Disabled);
        }

        if argv.is_empty() {
            return Err(ExecError::EmptyArgv);
        }

        validate_approval(argv, risk, approved, confirmation)?;

        let start = Instant::now();
        let result = run_argv(argv, self.timeout);
        let duration = start.elapsed();

        let exec_result = match &result {
            Ok(output) => ExecResult {
                exit_code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                duration,
                timed_out: false,
            },
            Err(e) if is_timeout_error(e) => ExecResult {
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration,
                timed_out: true,
            },
            Err(e) => return Err(ExecError::Spawn(copy_io_error(e))),
        };

        // Best-effort audit logging -- execution already happened, so we log
        // but convert write errors to a non-fatal warning via the result.
        if let Err(e) = self.write_audit_entry(argv, risk, approved, &exec_result) {
            // In production we would use tracing::warn! here; for now the
            // error is surfaced if callers inspect it.  We still return the
            // exec result because the command already ran.
            eprintln!("audit log write failed: {e}");
        }

        Ok(exec_result)
    }

    // -- private helpers ----------------------------------------------------

    fn write_audit_entry(
        &self,
        argv: &[String],
        risk: Risk,
        approved: bool,
        result: &ExecResult,
    ) -> Result<(), ExecError> {
        let entry = AuditEntry {
            timestamp: now_iso8601(),
            argv: argv.to_vec(),
            risk: format!("{risk:?}").to_lowercase(),
            approved,
            executed: true,
            exit_code: result.exit_code,
            duration_ms: result.duration.as_millis() as u64,
            timed_out: result.timed_out,
        };

        let line =
            serde_json::to_string(&entry).map_err(|e| ExecError::AuditWrite(e.to_string()))?;

        // Create parent directories if needed.
        if let Some(parent) = self.audit_log_path.parent() {
            fs::create_dir_all(parent).map_err(|e| ExecError::AuditWrite(e.to_string()))?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_log_path)
            .map_err(|e| ExecError::AuditWrite(e.to_string()))?;

        writeln!(file, "{line}").map_err(|e| ExecError::AuditWrite(e.to_string()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Approval validation
// ---------------------------------------------------------------------------

fn validate_approval(
    argv: &[String],
    risk: Risk,
    approved: bool,
    confirmation: Option<&str>,
) -> Result<(), ExecError> {
    match risk {
        Risk::Low | Risk::Medium => {
            if !approved {
                return Err(ExecError::NotApproved(risk));
            }
        }
        Risk::High => {
            if !approved {
                return Err(ExecError::NotApproved(risk));
            }
            match confirmation {
                Some(c) if c == "yes" => {}
                other => return Err(ExecError::HighRiskNoConfirmation(other.map(String::from))),
            }
        }
        Risk::Critical => {
            if !approved {
                return Err(ExecError::NotApproved(risk));
            }
            let full_command = argv.join(" ");
            match confirmation {
                None => return Err(ExecError::CriticalRiskNoConfirmation),
                Some(c) if c != full_command => return Err(ExecError::CriticalRiskMismatch),
                Some(_) => {}
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Process execution (argv-only, never shell)
// ---------------------------------------------------------------------------

fn run_argv(argv: &[String], timeout: Duration) -> std::io::Result<std::process::Output> {
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    // Blocking wait with timeout.
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(_status) => return child.wait_with_output(),
            None => {
                if start.elapsed() >= timeout {
                    // Kill the child and return a timeout indicator.
                    let _ = child.kill();
                    let _ = child.wait(); // reap zombie
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "command timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn is_timeout_error(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::TimedOut
}

/// Create a new `io::Error` with the same kind and message (io::Error is not Clone).
fn copy_io_error(e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), e.to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_audit_log_path() -> PathBuf {
    dirs::state_dir()
        .or_else(|| {
            dirs::home_dir().map(|h| h.join(".local").join("state"))
        })
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cmd-proposer")
        .join("audit.log")
}

fn now_iso8601() -> String {
    // Minimal ISO 8601 without pulling in chrono; uses seconds since epoch.
    // Format: 1970-01-01T00:00:00Z  (good enough for audit logs).
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    // Manual UTC breakdown.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01 to (year, month, day).
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm adapted from Howard Hinnant's `civil_from_days`.
    let era_offset = days + 719_468;
    let era = era_offset / 146_097;
    let doe = era_offset - era * 146_097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn runner_disabled(dir: &std::path::Path) -> ExecutionRunner {
        ExecutionRunner::new(false, Duration::from_secs(10), Some(dir.join("audit.log")))
    }

    fn runner_enabled(dir: &std::path::Path) -> ExecutionRunner {
        ExecutionRunner::new(true, Duration::from_secs(10), Some(dir.join("audit.log")))
    }

    fn runner_short_timeout(dir: &std::path::Path) -> ExecutionRunner {
        ExecutionRunner::new(
            true,
            Duration::from_millis(200),
            Some(dir.join("audit.log")),
        )
    }

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_owned()).collect()
    }

    // -- disabled runner ----------------------------------------------------

    #[test]
    fn disabled_runner_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_disabled(tmp.path());
        let result = runner.execute(&argv(&["echo", "hello"]), Risk::Low, true, None);
        assert!(matches!(result, Err(ExecError::Disabled)));
    }

    // -- empty argv ---------------------------------------------------------

    #[test]
    fn empty_argv_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(&[], Risk::Low, true, None);
        assert!(matches!(result, Err(ExecError::EmptyArgv)));
    }

    // -- argv-only execution ------------------------------------------------

    #[test]
    fn executes_simple_command() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner
            .execute(&argv(&["echo", "hello world"]), Risk::Low, true, None)
            .unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert_eq!(result.stdout.trim(), "hello world");
        assert!(!result.timed_out);
    }

    #[test]
    fn captures_exit_code_and_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        // `false` exits with code 1
        let result = runner
            .execute(&argv(&["false"]), Risk::Low, true, None)
            .unwrap();
        assert_eq!(result.exit_code, Some(1));
        assert!(!result.timed_out);
    }

    #[test]
    fn argv_does_not_interpret_shell_metacharacters() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        // If shell interpretation happened, `$(whoami)` would expand.
        let result = runner
            .execute(
                &argv(&["echo", "$(whoami)", "&&", "echo", "injected"]),
                Risk::Low,
                true,
                None,
            )
            .unwrap();
        // echo receives the literal strings as separate arguments
        assert!(result.stdout.contains("$(whoami)"));
        assert!(result.stdout.contains("&&"));
        assert!(result.stdout.contains("injected"));
    }

    // -- risk-based approval ------------------------------------------------

    #[test]
    fn low_risk_unapproved_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(&argv(&["true"]), Risk::Low, false, None);
        assert!(matches!(result, Err(ExecError::NotApproved(Risk::Low))));
    }

    #[test]
    fn medium_risk_unapproved_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(&argv(&["true"]), Risk::Medium, false, None);
        assert!(matches!(result, Err(ExecError::NotApproved(Risk::Medium))));
    }

    #[test]
    fn high_risk_without_yes_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        // Approved but no confirmation string
        let result = runner.execute(&argv(&["true"]), Risk::High, true, None);
        assert!(matches!(
            result,
            Err(ExecError::HighRiskNoConfirmation(None))
        ));
    }

    #[test]
    fn high_risk_with_wrong_confirmation_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(&argv(&["true"]), Risk::High, true, Some("nope"));
        assert!(matches!(
            result,
            Err(ExecError::HighRiskNoConfirmation(Some(_)))
        ));
    }

    #[test]
    fn high_risk_with_yes_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner
            .execute(&argv(&["true"]), Risk::High, true, Some("yes"))
            .unwrap();
        assert_eq!(result.exit_code, Some(0));
    }

    #[test]
    fn critical_risk_without_confirmation_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(&argv(&["rm", "-rf", "/"]), Risk::Critical, true, None);
        assert!(matches!(
            result,
            Err(ExecError::CriticalRiskNoConfirmation)
        ));
    }

    #[test]
    fn critical_risk_with_wrong_confirmation_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        let result = runner.execute(
            &argv(&["rm", "-rf", "/"]),
            Risk::Critical,
            true,
            Some("rm -rf"),
        );
        assert!(matches!(result, Err(ExecError::CriticalRiskMismatch)));
    }

    #[test]
    fn critical_risk_with_full_command_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        // Use a safe command for actual execution
        let result = runner
            .execute(
                &argv(&["echo", "critical"]),
                Risk::Critical,
                true,
                Some("echo critical"),
            )
            .unwrap();
        assert_eq!(result.exit_code, Some(0));
    }

    // -- audit log ----------------------------------------------------------

    #[test]
    fn audit_log_entry_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());
        runner
            .execute(&argv(&["echo", "audit-test"]), Risk::Low, true, None)
            .unwrap();

        let log_path = tmp.path().join("audit.log");
        assert!(log_path.exists(), "audit log file should exist");

        let content = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1, "should have exactly one log entry");

        let entry: AuditEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(entry.argv, vec!["echo", "audit-test"]);
        assert_eq!(entry.risk, "low");
        assert!(entry.approved);
        assert!(entry.executed);
        assert_eq!(entry.exit_code, Some(0));
        assert!(!entry.timed_out);
    }

    #[test]
    fn audit_log_is_append_only() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_enabled(tmp.path());

        runner
            .execute(&argv(&["echo", "first"]), Risk::Low, true, None)
            .unwrap();
        runner
            .execute(&argv(&["echo", "second"]), Risk::Low, true, None)
            .unwrap();

        let log_path = tmp.path().join("audit.log");
        let content = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "should have two log entries");

        let first: AuditEntry = serde_json::from_str(lines[0]).unwrap();
        let second: AuditEntry = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(first.argv, vec!["echo", "first"]);
        assert_eq!(second.argv, vec!["echo", "second"]);
    }

    // -- timeout detection --------------------------------------------------

    #[test]
    fn timeout_is_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = runner_short_timeout(tmp.path());
        let result = runner
            .execute(&argv(&["sleep", "10"]), Risk::Low, true, None)
            .unwrap();
        assert!(result.timed_out);
        assert!(result.exit_code.is_none());
        // Duration should be close to the 200ms timeout, not 10s.
        assert!(result.duration < Duration::from_secs(2));
    }

    // -- ISO 8601 timestamp -------------------------------------------------

    #[test]
    fn timestamp_format_is_iso8601() {
        let ts = now_iso8601();
        // Pattern: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20);
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
        assert!(ts.ends_with('Z'));
    }

    // -- default audit log path ---------------------------------------------

    #[test]
    fn default_path_ends_with_expected_suffix() {
        let path = default_audit_log_path();
        assert!(
            path.ends_with("cmd-proposer/audit.log"),
            "expected path ending with cmd-proposer/audit.log, got: {path:?}"
        );
    }
}

//! In-memory doc storage with token-counted grep/section/lines/expand retrieval.
//!
//! See `drafts/SPEC.md` §4.2.
//!
//! [`DocStore`] holds tool outputs (help text, man pages, web pages) keyed by
//! `doc_id`. The main agent and subagents NEVER carry raw doc bytes in their
//! conversation; instead they call the retrieval operations on this store and
//! pull back small spans. The store is thread-safe because multiple subagents
//! may read concurrently while the main agent writes.
//!
//! Match IDs returned by [`DocStore::doc_grep`] can later be passed to
//! [`DocStore::doc_expand_around`] to widen the context window around a hit
//! without re-running the regex.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use cps_tokenizer::Tokenizer;
use regex::RegexBuilder;

// ---------- safety limits ----------

/// Hard cap on `max_matches` accepted from a tool call (SPEC §4.2).
///
/// Callers may request fewer; values above this are clamped. The cap exists
/// to prevent a runaway regex on a huge doc from producing tens of thousands
/// of result rows that the agent then has to filter.
pub const MAX_MATCHES_CAP: usize = 50;

/// `regex::RegexBuilder::size_limit` for compiled patterns.
///
/// `regex` already rejects catastrophic backtracking via its automaton-based
/// engine, but a malicious pattern can still inflate the compiled NFA. 1 MiB
/// is generous for any human-written --help regex and tiny vs total RSS.
const REGEX_SIZE_LIMIT_BYTES: usize = 1 << 20;

// ---------- public types ----------

/// Provenance tag for a stored document.
///
/// `LocalSchema` and `LocalDoc` are trusted; `UntrustedWeb` is NOT — the
/// agent's safety rules forbid web content from overriding system prompts
/// or invoking tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// `kubectl explain`, OpenAPI schema, etc. — highest trust.
    LocalSchema,
    /// `--help`, `man`, `info` output produced inside the bwrap sandbox.
    LocalDoc,
    /// Anything fetched via the web runner; always treated as untrusted.
    UntrustedWeb,
}

/// Metadata returned by [`DocStore::insert`] and [`DocStore::doc_token_count`].
///
/// `token_estimate` comes from the supplied [`Tokenizer`] at insert time, so
/// downstream budget checks are deterministic for the rest of the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocMeta {
    pub doc_id: String,
    pub byte_len: usize,
    pub line_count: usize,
    pub token_estimate: usize,
    pub source_kind: SourceKind,
}

/// One hit returned by [`DocStore::doc_grep`].
///
/// `match_id` is opaque — the caller may pass it back into
/// [`DocStore::doc_expand_around`] to widen the context window without
/// re-running the regex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepMatch {
    pub match_id: String,
    pub line_number: usize,
    pub line_text: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

/// Failure modes for the public API.
#[derive(Debug, thiserror::Error)]
pub enum DocStoreError {
    #[error("doc not found: {0}")]
    DocNotFound(String),

    #[error("invalid pattern {pattern:?}: {source}")]
    InvalidPattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("match not found: {0}")]
    MatchNotFound(String),

    #[error("invalid line range: start={start} end={end} (1-indexed, end >= start)")]
    InvalidLineRange { start: usize, end: usize },
}

pub type Result<T> = std::result::Result<T, DocStoreError>;

// ---------- store internals ----------

#[derive(Debug)]
struct StoredDoc {
    raw_text: String,
    byte_len: usize,
    line_count: usize,
    token_estimate: usize,
    source_kind: SourceKind,
}

impl StoredDoc {
    fn meta(&self, doc_id: &str) -> DocMeta {
        DocMeta {
            doc_id: doc_id.to_string(),
            byte_len: self.byte_len,
            line_count: self.line_count,
            token_estimate: self.token_estimate,
            source_kind: self.source_kind,
        }
    }
}

/// Where a previously-emitted [`GrepMatch::match_id`] points to.
///
/// `doc_id` is captured so that [`DocStore::doc_expand_around`] cannot be
/// tricked into reading line N of a different doc.
#[derive(Debug, Clone)]
struct MatchPosition {
    doc_id: String,
    line: usize,
}

#[derive(Debug, Default)]
struct Inner {
    docs: HashMap<String, StoredDoc>,
    matches: HashMap<String, MatchPosition>,
}

/// Thread-safe in-memory store of tool-output documents.
///
/// Cheap to `clone`: shares the same backing storage via `Arc`. Multiple
/// subagents read concurrently while the main agent writes; the global
/// `RwLock` is held only for the duration of a single read or insert and
/// never across `.await`.
#[derive(Debug, Clone, Default)]
pub struct DocStore {
    inner: Arc<RwLock<Inner>>,
    match_counter: Arc<AtomicU64>,
}

impl DocStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `text` under `doc_id`, replacing any prior content for that id.
    ///
    /// `token_estimate` is computed from `tokenizer` and snapshotted on the
    /// stored record — re-inserting the same `doc_id` with a different
    /// tokenizer will overwrite. `byte_len`/`line_count` are computed from the
    /// raw text and do NOT depend on the tokenizer.
    pub fn insert(
        &self,
        doc_id: impl Into<String>,
        text: impl Into<String>,
        source_kind: SourceKind,
        tokenizer: &dyn Tokenizer,
    ) -> DocMeta {
        let doc_id = doc_id.into();
        let raw_text = text.into();
        let byte_len = raw_text.len();
        let line_count = count_lines(&raw_text);
        let token_estimate = tokenizer.count_tokens(&raw_text);

        let doc = StoredDoc {
            raw_text,
            byte_len,
            line_count,
            token_estimate,
            source_kind,
        };
        let meta = doc.meta(&doc_id);
        let mut guard = self.inner.write().expect("DocStore lock poisoned");
        guard.docs.insert(doc_id, doc);
        meta
    }

    /// Returns metadata only — never the raw text.
    pub fn doc_token_count(&self, doc_id: &str) -> Result<DocMeta> {
        let guard = self.inner.read().expect("DocStore lock poisoned");
        let doc = guard
            .docs
            .get(doc_id)
            .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;
        Ok(doc.meta(doc_id))
    }

    /// Return a prefix of the doc no larger than `max_tokens`.
    ///
    /// Truncation is performed at line boundaries by walking lines until the
    /// running token estimate would exceed `max_tokens`. The supplied
    /// `tokenizer` MUST match the one used at insert; passing a different
    /// tokenizer is sound but yields a slightly different cut-off.
    pub fn doc_preview(
        &self,
        doc_id: &str,
        max_tokens: usize,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String> {
        let guard = self.inner.read().expect("DocStore lock poisoned");
        let doc = guard
            .docs
            .get(doc_id)
            .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;

        if max_tokens == 0 || doc.raw_text.is_empty() {
            return Ok(String::new());
        }
        if doc.token_estimate <= max_tokens {
            return Ok(doc.raw_text.clone());
        }

        let mut out = String::new();
        let mut used = 0usize;
        for line in doc.raw_text.split_inclusive('\n') {
            let next = tokenizer.count_tokens(line).max(if line.is_empty() { 0 } else { 1 });
            if used + next > max_tokens {
                break;
            }
            out.push_str(line);
            used += next;
        }
        Ok(out)
    }

    /// Search `doc_id` for `pattern` and return up to `max_matches` hits.
    ///
    /// `case_insensitive` toggles `(?i)`. `context_lines` is applied
    /// symmetrically (clamped to doc bounds). `max_matches` is capped at
    /// [`MAX_MATCHES_CAP`] regardless of caller intent.
    ///
    /// Each returned [`GrepMatch::match_id`] is registered internally so a
    /// later [`DocStore::doc_expand_around`] call can find the same line
    /// without re-running the regex. Match IDs are globally unique within
    /// a `DocStore` instance.
    pub fn doc_grep(
        &self,
        doc_id: &str,
        pattern: &str,
        case_insensitive: bool,
        context_lines: usize,
        max_matches: usize,
    ) -> Result<Vec<GrepMatch>> {
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(case_insensitive)
            .size_limit(REGEX_SIZE_LIMIT_BYTES)
            .build()
            .map_err(|source| DocStoreError::InvalidPattern {
                pattern: pattern.to_string(),
                source,
            })?;

        let cap = max_matches.min(MAX_MATCHES_CAP);
        if cap == 0 {
            return Ok(Vec::new());
        }

        // Collect owned line copies under the read lock, then release it
        // before acquiring the write lock for the match registry — a single
        // thread holding both shared and exclusive on the same `RwLock` is
        // implementation-defined behavior in `std::sync::RwLock`.
        let owned_lines: Vec<String> = {
            let guard = self.inner.read().expect("DocStore lock poisoned");
            let doc = guard
                .docs
                .get(doc_id)
                .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;
            doc.raw_text.lines().map(str::to_string).collect()
        };

        let mut hits: Vec<(usize, String)> = Vec::new();
        for (idx, line) in owned_lines.iter().enumerate() {
            if regex.is_match(line) {
                hits.push((idx + 1, line.clone()));
                if hits.len() >= cap {
                    break;
                }
            }
        }
        let lines: Vec<&str> = owned_lines.iter().map(String::as_str).collect();

        let mut out = Vec::with_capacity(hits.len());
        let mut guard = self.inner.write().expect("DocStore lock poisoned");
        for (line_number, line_text) in hits {
            let match_id = self.next_match_id();
            let (before, after) = window_context(&lines, line_number, context_lines);
            guard.matches.insert(
                match_id.clone(),
                MatchPosition {
                    doc_id: doc_id.to_string(),
                    line: line_number,
                },
            );
            out.push(GrepMatch {
                match_id,
                line_number,
                line_text,
                context_before: before,
                context_after: after,
            });
        }
        Ok(out)
    }

    /// Extract the section whose heading matches `heading_regex`.
    ///
    /// A line is recognised as a heading when ANY of these holds:
    ///   - markdown ATX header: `^#{1,6}\s` (`# Foo`, `### Bar`);
    ///   - trailing colon line (non-indented): `^[^\s].*:\s*$` (`SYNOPSIS:`);
    ///   - all-caps line: every alphabetic char uppercase, contains ≥1
    ///     letter, no leading whitespace.
    ///
    /// The returned slice runs from the matching heading up to (but not
    /// including) the next heading or EOF, then is truncated at line
    /// boundaries to fit `max_tokens`. If no heading matches, returns
    /// `DocNotFound` is NOT raised — an empty string is returned because the
    /// section semantically does not exist, distinct from a missing doc.
    pub fn doc_section(
        &self,
        doc_id: &str,
        heading_regex: &str,
        max_tokens: usize,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String> {
        let regex = RegexBuilder::new(heading_regex)
            .size_limit(REGEX_SIZE_LIMIT_BYTES)
            .build()
            .map_err(|source| DocStoreError::InvalidPattern {
                pattern: heading_regex.to_string(),
                source,
            })?;

        let guard = self.inner.read().expect("DocStore lock poisoned");
        let doc = guard
            .docs
            .get(doc_id)
            .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;

        let lines: Vec<&str> = doc.raw_text.lines().collect();
        let Some(start) = lines.iter().position(|l| is_heading(l) && regex.is_match(l)) else {
            return Ok(String::new());
        };
        let end = lines
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, l)| is_heading(l))
            .map(|(i, _)| i)
            .unwrap_or(lines.len());

        let mut out = String::new();
        let mut used = 0usize;
        for line in &lines[start..end] {
            let chunk = format!("{}\n", line);
            let next = tokenizer.count_tokens(&chunk).max(if chunk.trim().is_empty() { 0 } else { 1 });
            if max_tokens > 0 && used + next > max_tokens {
                break;
            }
            out.push_str(&chunk);
            used += next;
        }
        Ok(out)
    }

    /// Return lines `[start, end)` half-open (1-indexed).
    ///
    /// Returns `InvalidLineRange` if `start == 0` or `end < start`. Out-of-
    /// range tails are silently truncated to the doc end.
    pub fn doc_lines(&self, doc_id: &str, start: usize, end: usize) -> Result<String> {
        if start == 0 || end < start {
            return Err(DocStoreError::InvalidLineRange { start, end });
        }
        let guard = self.inner.read().expect("DocStore lock poisoned");
        let doc = guard
            .docs
            .get(doc_id)
            .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;

        let mut out = String::new();
        for (idx, line) in doc.raw_text.lines().enumerate() {
            let n = idx + 1;
            if n < start {
                continue;
            }
            if n >= end {
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
        Ok(out)
    }

    /// Widen the window around a prior grep hit by `before` lines above and
    /// `after` lines below, returning the joined text with trailing newlines.
    ///
    /// `match_id` MUST have come from a prior [`DocStore::doc_grep`] on the
    /// same `doc_id`; mismatched ids raise [`DocStoreError::MatchNotFound`].
    pub fn doc_expand_around(
        &self,
        doc_id: &str,
        match_id: &str,
        before: usize,
        after: usize,
    ) -> Result<String> {
        let guard = self.inner.read().expect("DocStore lock poisoned");
        let position = guard
            .matches
            .get(match_id)
            .ok_or_else(|| DocStoreError::MatchNotFound(match_id.to_string()))?;
        if position.doc_id != doc_id {
            return Err(DocStoreError::MatchNotFound(match_id.to_string()));
        }
        let doc = guard
            .docs
            .get(doc_id)
            .ok_or_else(|| DocStoreError::DocNotFound(doc_id.to_string()))?;

        let lines: Vec<&str> = doc.raw_text.lines().collect();
        let line_idx = position.line.saturating_sub(1);
        let start = line_idx.saturating_sub(before);
        let end = (line_idx + after + 1).min(lines.len());
        let mut out = String::new();
        for line in &lines[start..end] {
            out.push_str(line);
            out.push('\n');
        }
        Ok(out)
    }

    fn next_match_id(&self) -> String {
        format!("m{}", self.match_counter.fetch_add(1, Ordering::Relaxed))
    }
}

// ---------- helpers ----------

fn count_lines(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    // `lines()` ignores a trailing newline; we want logical line count which
    // counts the final line whether or not it ends with `\n`.
    s.lines().count()
}

/// Return `(context_before, context_after)` around `line_number` (1-indexed),
/// each capped to `radius` lines and clamped to the doc bounds.
fn window_context(lines: &[&str], line_number: usize, radius: usize) -> (Vec<String>, Vec<String>) {
    let idx = line_number.saturating_sub(1);
    let before_start = idx.saturating_sub(radius);
    let after_end = idx.saturating_add(radius).saturating_add(1).min(lines.len());
    let before = lines[before_start..idx].iter().map(|s| s.to_string()).collect();
    let after = lines[(idx + 1).min(lines.len())..after_end]
        .iter()
        .map(|s| s.to_string())
        .collect();
    (before, after)
}

/// Heuristic heading detector — see [`DocStore::doc_section`] doc-comment for
/// the precise rules.
fn is_heading(line: &str) -> bool {
    let trimmed_end = line.trim_end();
    if trimmed_end.is_empty() {
        return false;
    }
    // Markdown ATX header.
    if let Some(rest) = trimmed_end.strip_prefix('#') {
        // Allow up to 6 leading hashes, then a space.
        let extra = rest.chars().take_while(|c| *c == '#').count();
        if extra <= 5 {
            let after = &rest[extra..];
            if after.starts_with(' ') {
                return true;
            }
        }
    }
    // Indented lines are body text, not headings.
    let starts_indented = line.starts_with(|c: char| c == ' ' || c == '\t');
    if starts_indented {
        return false;
    }
    // Trailing-colon line, e.g. `SYNOPSIS:` or `Options:`.
    if trimmed_end.ends_with(':') && trimmed_end.len() > 1 {
        return true;
    }
    // ALL-CAPS line: every alphabetic char uppercase, at least one letter.
    let mut has_alpha = false;
    for c in trimmed_end.chars() {
        if c.is_alphabetic() {
            has_alpha = true;
            if !c.is_uppercase() {
                return false;
            }
        }
    }
    has_alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use cps_tokenizer::FallbackTokenizer;

    fn store_with(text: &str) -> (DocStore, FallbackTokenizer, DocMeta) {
        let store = DocStore::new();
        let tok = FallbackTokenizer::new();
        let meta = store.insert("d1", text, SourceKind::LocalDoc, &tok);
        (store, tok, meta)
    }

    #[test]
    fn insert_records_byte_and_line_counts() {
        let text = "one\ntwo\nthree\n";
        let (_store, _tok, meta) = store_with(text);
        assert_eq!(meta.doc_id, "d1");
        assert_eq!(meta.byte_len, text.len());
        assert_eq!(meta.line_count, 3);
        // Fallback tokenizer: bytes / 4
        assert_eq!(meta.token_estimate, text.len() / 4);
        assert_eq!(meta.source_kind, SourceKind::LocalDoc);
    }

    #[test]
    fn insert_counts_trailing_unterminated_line() {
        let (_, _, meta) = store_with("a\nb\nc");
        assert_eq!(meta.line_count, 3);
    }

    #[test]
    fn empty_doc_has_zero_lines() {
        let (_, _, meta) = store_with("");
        assert_eq!(meta.line_count, 0);
        assert_eq!(meta.byte_len, 0);
    }

    #[test]
    fn doc_token_count_returns_inserted_meta() {
        let (store, _, meta) = store_with("hello world\n");
        let got = store.doc_token_count("d1").expect("present");
        assert_eq!(got, meta);
    }

    #[test]
    fn doc_token_count_missing_returns_doc_not_found() {
        let store = DocStore::new();
        let err = store.doc_token_count("missing").expect_err("must error");
        assert!(matches!(err, DocStoreError::DocNotFound(id) if id == "missing"));
    }

    #[test]
    fn doc_preview_returns_whole_doc_when_within_budget() {
        let text = "hello world";
        let (store, tok, meta) = store_with(text);
        let out = store
            .doc_preview("d1", meta.token_estimate + 10, &tok)
            .expect("ok");
        assert_eq!(out, text);
    }

    #[test]
    fn doc_preview_truncates_at_line_boundary() {
        // Each line is 16 chars = 4 fallback tokens.
        let line = "abcdefghijklmnop\n";
        let text = line.repeat(10); // 160 bytes -> 40 tokens
        let (store, tok, _) = store_with(&text);
        // Ask for 8 tokens -> expect the first 2 lines (8 tokens).
        let out = store.doc_preview("d1", 8, &tok).expect("ok");
        assert_eq!(out, line.repeat(2));
    }

    #[test]
    fn doc_preview_zero_budget_returns_empty() {
        let (store, tok, _) = store_with("anything\n");
        assert_eq!(store.doc_preview("d1", 0, &tok).expect("ok"), "");
    }

    #[test]
    fn doc_grep_finds_matches_with_context() {
        let text = "alpha\nbeta\ngamma\ndelta\nepsilon\n";
        let (store, _, _) = store_with(text);
        let hits = store.doc_grep("d1", "gamma", false, 1, 10).expect("ok");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line_number, 3);
        assert_eq!(hits[0].line_text, "gamma");
        assert_eq!(hits[0].context_before, vec!["beta".to_string()]);
        assert_eq!(hits[0].context_after, vec!["delta".to_string()]);
    }

    #[test]
    fn doc_grep_case_insensitive() {
        let (store, _, _) = store_with("FOO\nBar\nfoo\n");
        let hits = store.doc_grep("d1", "foo", true, 0, 10).expect("ok");
        assert_eq!(hits.iter().map(|h| h.line_number).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn doc_grep_invalid_regex_errors() {
        let (store, _, _) = store_with("anything");
        let err = store.doc_grep("d1", "(", false, 0, 10).expect_err("must error");
        match err {
            DocStoreError::InvalidPattern { pattern, .. } => assert_eq!(pattern, "("),
            other => panic!("expected InvalidPattern, got {other:?}"),
        }
    }

    #[test]
    fn doc_grep_caps_max_matches() {
        let text = "x\n".repeat(MAX_MATCHES_CAP + 25);
        let (store, _, _) = store_with(&text);
        let hits = store
            .doc_grep("d1", "x", false, 0, MAX_MATCHES_CAP + 100)
            .expect("ok");
        assert_eq!(hits.len(), MAX_MATCHES_CAP);
    }

    #[test]
    fn doc_grep_respects_caller_limit_when_smaller_than_cap() {
        let text = "x\n".repeat(20);
        let (store, _, _) = store_with(&text);
        let hits = store.doc_grep("d1", "x", false, 0, 3).expect("ok");
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn doc_grep_missing_doc_errors() {
        let store = DocStore::new();
        let err = store
            .doc_grep("nope", "x", false, 0, 1)
            .expect_err("must error");
        assert!(matches!(err, DocStoreError::DocNotFound(_)));
    }

    #[test]
    fn doc_section_extracts_markdown_header_section() {
        let text = "\
# Title

intro paragraph

## Section A
body a1
body a2

## Section B
body b1
";
        let (store, tok, _) = store_with(text);
        let out = store
            .doc_section("d1", "^## Section A", 1000, &tok)
            .expect("ok");
        assert!(out.contains("## Section A"));
        assert!(out.contains("body a1"));
        assert!(out.contains("body a2"));
        assert!(!out.contains("Section B"));
        assert!(!out.contains("body b1"));
    }

    #[test]
    fn doc_section_detects_trailing_colon_heading() {
        let text = "\
SYNOPSIS:
  cmd [options]

DESCRIPTION:
  does things
";
        let (store, tok, _) = store_with(text);
        let out = store
            .doc_section("d1", "^SYNOPSIS:", 1000, &tok)
            .expect("ok");
        assert!(out.contains("SYNOPSIS:"));
        assert!(out.contains("cmd [options]"));
        assert!(!out.contains("DESCRIPTION:"));
    }

    #[test]
    fn doc_section_detects_all_caps_heading() {
        let text = "\
NAME
    foo - widget

OPTIONS
    --bar    do bar
";
        let (store, tok, _) = store_with(text);
        let out = store
            .doc_section("d1", "^OPTIONS$", 1000, &tok)
            .expect("ok");
        assert!(out.contains("OPTIONS"));
        assert!(out.contains("--bar"));
        assert!(!out.contains("NAME"));
    }

    #[test]
    fn doc_section_missing_heading_returns_empty() {
        let (store, tok, _) = store_with("# Foo\nbody\n");
        let out = store
            .doc_section("d1", "^# Bar", 1000, &tok)
            .expect("ok");
        assert_eq!(out, "");
    }

    #[test]
    fn doc_section_respects_max_tokens() {
        // Each line is 16 chars -> 4 fallback tokens. Heading itself is
        // "## Section\n" = 11 bytes -> 2 tokens.
        let mut text = String::from("## Section\n");
        for _ in 0..10 {
            text.push_str("abcdefghijklmnop\n");
        }
        let (store, tok, _) = store_with(&text);
        let out = store
            .doc_section("d1", "^## Section", 6, &tok) // heading (2) + 1 body line (4) = 6
            .expect("ok");
        assert!(out.starts_with("## Section\n"));
        // Only one body line should fit.
        let body_lines = out.lines().filter(|l| l.starts_with("abc")).count();
        assert_eq!(body_lines, 1);
    }

    #[test]
    fn doc_lines_returns_half_open_range() {
        let (store, _, _) = store_with("L1\nL2\nL3\nL4\nL5\n");
        let out = store.doc_lines("d1", 2, 4).expect("ok");
        assert_eq!(out, "L2\nL3\n");
    }

    #[test]
    fn doc_lines_clamps_past_end() {
        let (store, _, _) = store_with("L1\nL2\nL3\n");
        let out = store.doc_lines("d1", 2, 100).expect("ok");
        assert_eq!(out, "L2\nL3\n");
    }

    #[test]
    fn doc_lines_rejects_zero_start() {
        let (store, _, _) = store_with("x\n");
        let err = store.doc_lines("d1", 0, 5).expect_err("must error");
        assert!(matches!(err, DocStoreError::InvalidLineRange { .. }));
    }

    #[test]
    fn doc_lines_rejects_inverted_range() {
        let (store, _, _) = store_with("x\n");
        let err = store.doc_lines("d1", 5, 2).expect_err("must error");
        assert!(matches!(err, DocStoreError::InvalidLineRange { .. }));
    }

    #[test]
    fn doc_expand_around_widens_grep_window() {
        let text = (1..=20)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (store, _, _) = store_with(&text);
        let hits = store.doc_grep("d1", "^line 10$", false, 0, 1).expect("ok");
        let id = &hits[0].match_id;
        let out = store.doc_expand_around("d1", id, 2, 3).expect("ok");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["line 8", "line 9", "line 10", "line 11", "line 12", "line 13"]);
    }

    #[test]
    fn doc_expand_around_unknown_id_errors() {
        let (store, _, _) = store_with("x\n");
        let err = store
            .doc_expand_around("d1", "m9999", 1, 1)
            .expect_err("must error");
        assert!(matches!(err, DocStoreError::MatchNotFound(_)));
    }

    #[test]
    fn doc_expand_around_rejects_mismatched_doc() {
        let store = DocStore::new();
        let tok = FallbackTokenizer::new();
        store.insert("a", "alpha\n", SourceKind::LocalDoc, &tok);
        store.insert("b", "beta\n", SourceKind::LocalDoc, &tok);
        let hits = store.doc_grep("a", "alpha", false, 0, 1).expect("ok");
        let id = &hits[0].match_id;
        // Match id belongs to doc "a"; querying via doc "b" must fail.
        let err = store
            .doc_expand_around("b", id, 0, 0)
            .expect_err("must error");
        assert!(matches!(err, DocStoreError::MatchNotFound(_)));
    }

    #[test]
    fn doc_grep_zero_max_matches_returns_empty() {
        let (store, _, _) = store_with("x\nx\nx\n");
        let hits = store.doc_grep("d1", "x", false, 0, 0).expect("ok");
        assert!(hits.is_empty());
    }

    #[test]
    fn match_ids_are_unique_across_calls() {
        let (store, _, _) = store_with("x\ny\nx\ny\n");
        let h1 = store.doc_grep("d1", "x", false, 0, 10).expect("ok");
        let h2 = store.doc_grep("d1", "y", false, 0, 10).expect("ok");
        let all: Vec<_> = h1.iter().chain(h2.iter()).map(|h| h.match_id.clone()).collect();
        let dedup: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(all.len(), dedup.len());
    }

    #[test]
    fn store_is_send_and_sync() {
        // Spawn a thread that reads while the main thread writes — this
        // wouldn't compile if DocStore weren't Send + Sync, and would
        // deadlock if the locking strategy were wrong.
        let store = DocStore::new();
        let tok = FallbackTokenizer::new();
        store.insert("shared", "alpha\nbeta\ngamma\n", SourceKind::LocalDoc, &tok);
        let reader = store.clone();
        let handle = std::thread::spawn(move || {
            reader.doc_token_count("shared").expect("present");
        });
        store.insert("shared2", "delta\n", SourceKind::LocalDoc, &tok);
        handle.join().expect("reader thread");
    }
}

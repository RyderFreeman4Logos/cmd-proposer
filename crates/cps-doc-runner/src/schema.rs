//! CLI schema types and JSON parsing for custom CLI introspection.
//!
//! Programs that support `__schema --format=json` emit a structured
//! [`CliSchema`] describing their subcommands, arguments, flags, and
//! risk annotations. This module defines the schema types, the JSON
//! deserialisation, and a helper that attempts schema retrieval inside
//! the bwrap sandbox before falling back to `--help`.
//!
//! Evidence produced from a successfully parsed schema is tagged
//! [`SourceKind::LocalSchema`] — the highest trust tier.

use serde::Deserialize;

// ---------- public schema types ----------

/// Top-level schema describing an entire CLI program.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CliSchema {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub subcommands: Vec<SubcommandSchema>,
}

/// One subcommand (or nested sub-sub-command) within the CLI.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SubcommandSchema {
    /// Path segments, e.g. `["rollout", "restart"]` for `kubectl rollout restart`.
    pub path: Vec<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub args: Vec<ArgSchema>,
    #[serde(default)]
    pub flags: Vec<FlagSchema>,
    /// Risk level declared by the CLI author (e.g. "high", "critical").
    pub risk: Option<String>,
    #[serde(default)]
    pub examples: Vec<String>,
}

/// A positional argument within a subcommand.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ArgSchema {
    pub name: String,
    #[serde(default)]
    pub required: bool,
    pub description: Option<String>,
}

/// A flag (option) within a subcommand.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FlagSchema {
    pub name: String,
    #[serde(default = "default_flag_type")]
    pub flag_type: FlagType,
    #[serde(default)]
    pub required: bool,
    /// Risk annotation for this specific flag (e.g. "bypass-confirmation").
    pub risk: Option<String>,
    pub description: Option<String>,
}

/// The value type accepted by a flag.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", content = "values")]
pub enum FlagType {
    Bool,
    #[serde(alias = "String")]
    StringVal,
    Enum(Vec<String>),
}

fn default_flag_type() -> FlagType {
    FlagType::Bool
}

// ---------- parsing ----------

/// Attempt to parse raw JSON bytes into a [`CliSchema`].
///
/// Returns `None` on any parse failure — the caller should fall back
/// to `--help` exploration when this happens.
pub fn parse_schema_json(json: &str) -> Option<CliSchema> {
    serde_json::from_str(json).ok()
}

/// Build the doc_id for a schema entry.
pub(crate) fn schema_doc_id(program: &str) -> String {
    format!("schema:{program}")
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema_json() -> &'static str {
        r#"{
            "name": "myctl",
            "version": "1.2.0",
            "subcommands": [
                {
                    "path": ["deploy"],
                    "description": "Deploy the application",
                    "args": [
                        {
                            "name": "target",
                            "required": true,
                            "description": "Deployment target"
                        }
                    ],
                    "flags": [
                        {
                            "name": "--dry-run",
                            "flag_type": { "kind": "Bool" },
                            "required": false,
                            "description": "Simulate the deployment"
                        },
                        {
                            "name": "--env",
                            "flag_type": {
                                "kind": "Enum",
                                "values": ["staging", "production"]
                            },
                            "required": true,
                            "description": "Target environment"
                        },
                        {
                            "name": "--tag",
                            "flag_type": { "kind": "StringVal" },
                            "required": false,
                            "description": "Image tag to deploy"
                        }
                    ],
                    "risk": "high",
                    "examples": [
                        "myctl deploy web --env=staging --tag=v1.0"
                    ]
                },
                {
                    "path": ["deploy", "rollback"],
                    "description": "Rollback a deployment",
                    "args": [],
                    "flags": [
                        {
                            "name": "--yes",
                            "flag_type": { "kind": "Bool" },
                            "required": false,
                            "risk": "bypass-confirmation",
                            "description": "Skip confirmation prompt"
                        }
                    ],
                    "risk": "critical",
                    "examples": []
                },
                {
                    "path": ["status"],
                    "description": "Show deployment status",
                    "args": [],
                    "flags": [],
                    "risk": null,
                    "examples": ["myctl status"]
                }
            ]
        }"#
    }

    #[test]
    fn parse_sample_schema_json_into_cli_schema() {
        let schema = parse_schema_json(sample_schema_json()).expect("valid JSON");
        assert_eq!(schema.name, "myctl");
        assert_eq!(schema.version, "1.2.0");
        assert_eq!(schema.subcommands.len(), 3);

        let deploy = &schema.subcommands[0];
        assert_eq!(deploy.path, vec!["deploy"]);
        assert_eq!(deploy.description.as_deref(), Some("Deploy the application"));
        assert_eq!(deploy.args.len(), 1);
        assert_eq!(deploy.flags.len(), 3);
        assert_eq!(deploy.examples.len(), 1);
    }

    #[test]
    fn schema_risk_levels_extracted_correctly() {
        let schema = parse_schema_json(sample_schema_json()).expect("valid JSON");

        let deploy = &schema.subcommands[0];
        assert_eq!(deploy.risk.as_deref(), Some("high"));

        let rollback = &schema.subcommands[1];
        assert_eq!(rollback.risk.as_deref(), Some("critical"));

        let status = &schema.subcommands[2];
        assert_eq!(status.risk, None);
    }

    #[test]
    fn flag_type_parsing_bool_string_enum() {
        let schema = parse_schema_json(sample_schema_json()).expect("valid JSON");
        let flags = &schema.subcommands[0].flags;

        assert_eq!(flags[0].name, "--dry-run");
        assert_eq!(flags[0].flag_type, FlagType::Bool);

        assert_eq!(flags[1].name, "--env");
        assert_eq!(
            flags[1].flag_type,
            FlagType::Enum(vec!["staging".into(), "production".into()])
        );

        assert_eq!(flags[2].name, "--tag");
        assert_eq!(flags[2].flag_type, FlagType::StringVal);
    }

    #[test]
    fn flag_risk_annotations_parsed() {
        let schema = parse_schema_json(sample_schema_json()).expect("valid JSON");
        let rollback = &schema.subcommands[1];
        let yes_flag = &rollback.flags[0];

        assert_eq!(yes_flag.name, "--yes");
        assert_eq!(yes_flag.risk.as_deref(), Some("bypass-confirmation"));

        // Flags without risk should be None.
        let deploy_dry_run = &schema.subcommands[0].flags[0];
        assert_eq!(deploy_dry_run.risk, None);
    }

    #[test]
    fn invalid_json_returns_none_gracefully() {
        assert!(parse_schema_json("").is_none());
        assert!(parse_schema_json("{").is_none());
        assert!(parse_schema_json("null").is_none());
        assert!(parse_schema_json("\"just a string\"").is_none());
        assert!(parse_schema_json("42").is_none());
        assert!(parse_schema_json(r#"{"name": "x"}"#).is_none()); // missing `version`
    }

    #[test]
    fn minimal_schema_parses_with_defaults() {
        let json = r#"{"name": "simple", "version": "0.1.0"}"#;
        let schema = parse_schema_json(json).expect("valid JSON");
        assert_eq!(schema.name, "simple");
        assert_eq!(schema.version, "0.1.0");
        assert!(schema.subcommands.is_empty());
    }

    #[test]
    fn flag_type_defaults_to_bool() {
        let json = r#"{
            "name": "t",
            "version": "0.1.0",
            "subcommands": [{
                "path": ["cmd"],
                "flags": [{ "name": "--verbose" }]
            }]
        }"#;
        let schema = parse_schema_json(json).expect("valid JSON");
        let flag = &schema.subcommands[0].flags[0];
        assert_eq!(flag.flag_type, FlagType::Bool);
        assert!(!flag.required);
        assert_eq!(flag.risk, None);
        assert_eq!(flag.description, None);
    }

    #[test]
    fn schema_doc_id_format() {
        assert_eq!(schema_doc_id("kubectl"), "schema:kubectl");
        assert_eq!(schema_doc_id("myctl"), "schema:myctl");
    }

    #[test]
    fn string_alias_for_flag_type() {
        // Accept "String" as an alias for "StringVal" for convenience.
        let json = r#"{
            "name": "t",
            "version": "0.1.0",
            "subcommands": [{
                "path": ["cmd"],
                "flags": [{
                    "name": "--output",
                    "flag_type": { "kind": "String" }
                }]
            }]
        }"#;
        let schema = parse_schema_json(json).expect("valid JSON");
        assert_eq!(schema.subcommands[0].flags[0].flag_type, FlagType::StringVal);
    }

    #[test]
    fn arg_required_defaults_to_false() {
        let json = r#"{
            "name": "t",
            "version": "0.1.0",
            "subcommands": [{
                "path": ["cmd"],
                "args": [{ "name": "file" }]
            }]
        }"#;
        let schema = parse_schema_json(json).expect("valid JSON");
        let arg = &schema.subcommands[0].args[0];
        assert!(!arg.required);
        assert_eq!(arg.description, None);
    }
}

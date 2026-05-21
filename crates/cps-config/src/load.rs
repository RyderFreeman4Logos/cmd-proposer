//! Config file lookup and loading.
//!
//! Resolution order (highest priority first):
//! 1. Explicit path passed via [`load_from_path`].
//! 2. `./.cmd-proposer.yaml` relative to the current working directory.
//! 3. `~/.config/cmd-proposer/config.yaml`.
//!
//! A `.env` file in the current working directory is loaded into the process
//! environment (without overriding pre-set vars) before interpolation.

use std::path::{Path, PathBuf};

use crate::{interpolate_env, Config, InterpolateError, ValidateError};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("no config file found (tried CLI path, ./.cmd-proposer.yaml, ~/.config/cmd-proposer/config.yaml)")]
    NotFound,

    #[error("failed to read `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse `{path}`: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("env var interpolation failed in `{path}`: {source}")]
    Interpolate {
        path: PathBuf,
        #[source]
        source: InterpolateError,
    },

    #[error("validation failed: {0}")]
    Validate(#[from] ValidateError),
}

/// Where the loaded config came from. Mostly useful for logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSource {
    pub path: PathBuf,
}

/// Try to load config from the standard lookup chain. Use [`load_from_path`]
/// if a CLI flag supplies an explicit path.
///
/// Loads `.env` from `cwd` into the process environment before interpolation
/// (without overriding pre-set variables).
pub fn load() -> Result<(Config, ConfigSource), LoadError> {
    let _ = dotenvy::dotenv();
    let path = match resolve_path() {
        Some(p) => p,
        None => return Err(LoadError::NotFound),
    };
    load_from_path(&path)
}

/// Load config from a specific path. Used directly by callers that have a
/// CLI-supplied path; the standard chain in [`load`] also funnels here.
pub fn load_from_path(path: &Path) -> Result<(Config, ConfigSource), LoadError> {
    let raw = std::fs::read_to_string(path).map_err(|e| LoadError::Read {
        path: path.to_path_buf(),
        source: e,
    })?;
    let interpolated = interpolate_env(&raw).map_err(|e| LoadError::Interpolate {
        path: path.to_path_buf(),
        source: e,
    })?;
    let cfg: Config = serde_yaml::from_str(&interpolated).map_err(|e| LoadError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    cfg.validate()?;
    Ok((
        cfg,
        ConfigSource {
            path: path.to_path_buf(),
        },
    ))
}

fn resolve_path() -> Option<PathBuf> {
    // 1. ./.cmd-proposer.yaml
    let local = PathBuf::from(".cmd-proposer.yaml");
    if local.exists() {
        return Some(local);
    }
    // 2. ~/.config/cmd-proposer/config.yaml
    if let Some(home_cfg) = dirs::config_dir() {
        let p = home_cfg.join("cmd-proposer").join("config.yaml");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".yaml")
            .tempfile()
            .expect("tempfile");
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[test]
    fn load_from_path_with_env_interpolation() {
        // Use unique var names to avoid colliding with the host environment.
        std::env::set_var("CPS_TEST_BASE_URL", "http://x/v1");
        std::env::set_var("CPS_TEST_API_KEY", "sk-test");
        std::env::set_var("CPS_TEST_MODEL", "qwen-test");
        let yaml = r#"
model:
  base_url: ${CPS_TEST_BASE_URL}
  api_key: ${CPS_TEST_API_KEY}
  model_name: ${CPS_TEST_MODEL}
  tokenizer:
    path: "/tmp/tokenizer.json"

doc_runner:
  allow_programs: [kubectl, helm]
"#;
        let f = write_tmp(yaml);
        let (cfg, src) = load_from_path(f.path()).expect("load");
        assert_eq!(cfg.model.base_url, "http://x/v1");
        assert_eq!(cfg.model.api_key, "sk-test");
        assert_eq!(cfg.model.model_name, "qwen-test");
        assert_eq!(src.path, f.path());
    }

    #[test]
    fn missing_required_field_returns_parse_error() {
        let yaml = r#"
doc_runner:
  allow_programs: [kubectl]
"#;
        let f = write_tmp(yaml);
        let err = load_from_path(f.path()).unwrap_err();
        assert!(matches!(err, LoadError::Parse { .. }), "got {err:?}");
        // Error message should point at the missing field path.
        let s = err.to_string();
        assert!(s.contains("model"), "{s}");
    }

    #[test]
    fn unset_env_var_returns_interpolate_error() {
        let yaml = r#"
model:
  base_url: ${CPS_TEST_THIS_SHOULD_NOT_EXIST_xyzzy}
  api_key: ""
  model_name: m
  tokenizer:
    path: /tmp/t

doc_runner:
  allow_programs: [kubectl]
"#;
        let f = write_tmp(yaml);
        let err = load_from_path(f.path()).unwrap_err();
        assert!(matches!(err, LoadError::Interpolate { .. }), "got {err:?}");
    }
}

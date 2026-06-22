use std::env;
use std::path::PathBuf;

use crate::error::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub blok_home: PathBuf,
    pub model_root: PathBuf,
    pub report_root: PathBuf,
    pub strict_direct_io: bool,
}

impl RuntimeConfig {
    pub fn from_env() -> Result<Self> {
        let blok_home = env_path("BLOK_HOME")?.unwrap_or(default_home()?);
        let model_root = env_path("BLOK_MODEL_ROOT")?.unwrap_or_else(|| blok_home.join("models"));
        let report_root =
            env_path("BLOK_REPORT_ROOT")?.unwrap_or_else(|| blok_home.join("reports"));
        let strict_direct_io = env_bool("BLOK_STRICT_DIRECT_IO")?.unwrap_or(true);

        Ok(Self {
            blok_home,
            model_root,
            report_root,
            strict_direct_io,
        })
    }
}

fn env_path(name: &str) -> Result<Option<PathBuf>> {
    match env::var_os(name) {
        Some(value) if value.is_empty() => Err(Error::Config(format!("{name} cannot be empty"))),
        Some(value) => Ok(Some(PathBuf::from(value))),
        None => Ok(None),
    }
}

fn env_bool(name: &str) -> Result<Option<bool>> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| Error::Config(format!("{name} must be valid unicode")))?;
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(Error::Config(format!("{name} must be boolean"))),
    }
}

fn default_home() -> Result<PathBuf> {
    let Some(home) = env::var_os("HOME") else {
        return Err(Error::Config(
            "BLOK_HOME is required when HOME is unset".to_owned(),
        ));
    };
    if home.is_empty() {
        return Err(Error::Config("HOME cannot be empty".to_owned()));
    }
    Ok(PathBuf::from(home).join(".blok"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn false_values_are_accepted() {
        env::set_var("BLOK_TEST_BOOL_FALSE", "off");
        assert_eq!(env_bool("BLOK_TEST_BOOL_FALSE").unwrap(), Some(false));
        env::remove_var("BLOK_TEST_BOOL_FALSE");
    }

    #[test]
    fn invalid_bool_is_rejected() {
        env::set_var("BLOK_TEST_BOOL_INVALID", "maybe");
        let err = env_bool("BLOK_TEST_BOOL_INVALID").unwrap_err();
        env::remove_var("BLOK_TEST_BOOL_INVALID");

        assert_eq!(
            err.to_string(),
            "config error: BLOK_TEST_BOOL_INVALID must be boolean"
        );
    }
}

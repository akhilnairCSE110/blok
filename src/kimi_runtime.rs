use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{Error, Manifest, Result, KIMI_EXPERTS_PER_TOKEN};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiNativeRequest<'a> {
    pub manifest: &'a Manifest,
    pub manifest_path: &'a Path,
    pub prompt: &'a str,
    pub max_tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiNativeResponse {
    pub text: String,
    pub tokens: u64,
    pub predicted_tps: Option<f64>,
    pub watts: Option<f64>,
}

pub fn generate_native(request: KimiNativeRequest<'_>) -> Result<KimiNativeResponse> {
    validate_request(&request)?;
    run_executor(&request)
}

fn run_executor(request: &KimiNativeRequest<'_>) -> Result<KimiNativeResponse> {
    let bin = executor_bin();
    if !bin.is_file() {
        return Err(Error::Capability("blok_kimi_exec_binary_required"));
    }
    let output = Command::new(&bin)
        .args([
            "--manifest",
            &request.manifest_path.display().to_string(),
            "--prompt",
            request.prompt,
            "--tokens",
            &request.max_tokens.to_string(),
            "--router-top-k",
            &KIMI_EXPERTS_PER_TOKEN.to_string(),
        ])
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(Error::Native(if stderr.is_empty() {
            format!("{} exited with {}", bin.display(), output.status)
        } else {
            stderr
        }));
    }
    parse_executor_json(&String::from_utf8_lossy(&output.stdout))
}

fn executor_bin() -> PathBuf {
    std::env::var_os("BLOK_KIMI_EXEC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("build/blok-kimi-exec"))
}

fn parse_executor_json(text: &str) -> Result<KimiNativeResponse> {
    let text_value = json_string(text, "text")
        .ok_or_else(|| Error::Native("executor response missing text".to_owned()))?;
    let tokens = json_u64(text, "tokens")
        .ok_or_else(|| Error::Native("executor response missing tokens".to_owned()))?;
    let predicted_tps = json_f64(text, "predicted_tps");
    let watts = json_f64(text, "watts");
    Ok(KimiNativeResponse {
        text: text_value,
        tokens,
        predicted_tps,
        watts,
    })
}

fn json_u64(text: &str, key: &str) -> Option<u64> {
    json_number(text, key)?.parse().ok()
}

fn json_f64(text: &str, key: &str) -> Option<f64> {
    json_number(text, key)?.parse().ok()
}

fn json_number<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let i = text.find(&marker)?;
    let value = text[i + marker.len()..].split_once(':')?.1.trim_start();
    let end = value
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(value.len());
    Some(&value[..end])
}

fn json_string(text: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\"");
    let i = text.find(&marker)?;
    let value = text[i + marker.len()..].split_once(':')?.1.trim_start();
    let value = value.strip_prefix('"')?;
    let mut out = String::new();
    let mut escaped = false;
    for c in value.chars() {
        if escaped {
            out.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn validate_request(request: &KimiNativeRequest<'_>) -> Result<()> {
    if request.max_tokens == 0 {
        return Err(Error::Cli("--tokens must be greater than zero".to_owned()));
    }
    if request.prompt.trim().is_empty() {
        return Err(Error::Cli("--prompt must not be empty".to_owned()));
    }
    if request.manifest.tensors.is_empty() {
        return Err(Error::Manifest("manifest has no tensors".to_owned()));
    }
    if !request.manifest_path.is_file() {
        return Err(Error::Manifest(format!(
            "manifest path does not exist: {}",
            request.manifest_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_executor_json;

    #[test]
    fn parses_null_measurements_as_unknown() {
        let got = parse_executor_json(
            r#"{"status":"ok","text":"paris","tokens":1,"predicted_tps":null,"watts":null}"#,
        )
        .unwrap();
        assert_eq!(got.text, "paris");
        assert_eq!(got.tokens, 1);
        assert_eq!(got.predicted_tps, None);
        assert_eq!(got.watts, None);
    }

    #[test]
    fn parses_positive_measurements() {
        let got = parse_executor_json(
            r#"{"status":"ok","text":"x","tokens":2,"predicted_tps":7.5,"watts":199.0}"#,
        )
        .unwrap();
        assert_eq!(got.predicted_tps, Some(7.5));
        assert_eq!(got.watts, Some(199.0));
    }
}

use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerPlan {
    pub tokenizer_json: PathBuf,
    pub tokenizer_config_json: Option<PathBuf>,
    pub model_type: Option<String>,
    pub vocab_size: Option<u64>,
    pub merge_count: Option<u64>,
    pub bos_token_id: Option<u64>,
    pub eos_token_id: Option<u64>,
    pub prompt_bytes: u64,
    pub prompt_chars: u64,
    pub estimated_prompt_tokens: u64,
    pub prompt_roundtrip: &'static str,
}

impl TokenizerPlan {
    pub fn load(model: &Path, prompt: &str) -> Result<Self> {
        let root = if model.is_file() {
            model.parent().unwrap_or(model)
        } else {
            model
        };
        let tokenizer_json = find_under(root, "tokenizer.json", 5).ok_or(Error::Capability(
            "tokenizer_json_required_before_prompt_execution",
        ))?;
        let tokenizer_config_json = find_under(root, "tokenizer_config.json", 5);
        let tokenizer_text = fs::read_to_string(&tokenizer_json)?;
        let config_text = tokenizer_config_json
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok());
        let prompt_bytes = prompt.len() as u64;
        let prompt_chars = prompt.chars().count() as u64;
        Ok(Self {
            tokenizer_json,
            tokenizer_config_json,
            model_type: json_string(&tokenizer_text, "type"),
            vocab_size: json_number(&tokenizer_text, "vocab_size")
                .or_else(|| count_vocab_entries(&tokenizer_text)),
            merge_count: count_array_entries(&tokenizer_text, "merges"),
            bos_token_id: config_text
                .as_deref()
                .and_then(|text| json_number(text, "bos_token_id")),
            eos_token_id: config_text
                .as_deref()
                .and_then(|text| json_number(text, "eos_token_id")),
            prompt_bytes,
            prompt_chars,
            estimated_prompt_tokens: estimate_tokens(prompt),
            prompt_roundtrip: "metadata_only_tokenizer_execution_pending",
        })
    }
}

fn find_under(root: &Path, name: &str, depth: u8) -> Option<PathBuf> {
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    if depth == 0 {
        return None;
    }
    let mut dirs: Vec<_> = fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .find_map(|path| find_under(&path, name, depth - 1))
}

fn json_number(text: &str, key: &str) -> Option<u64> {
    let marker = format!("\"{key}\"");
    let i = text.find(&marker)?;
    let after_key = &text[i + marker.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let digits: String = value.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn json_string(text: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\"");
    let i = text.find(&marker)?;
    let after_key = &text[i + marker.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
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

fn count_vocab_entries(text: &str) -> Option<u64> {
    let marker = "\"vocab\"";
    let i = text.find(marker)?;
    let rest = &text[i + marker.len()..];
    let open = rest.find('{')?;
    let mut depth = 0_u32;
    let mut in_string = false;
    let mut escaped = false;
    let mut count = 0_u64;
    let mut could_be_key = false;
    for c in rest[open..].chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && in_string {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            if in_string && depth == 1 {
                could_be_key = true;
            }
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(count);
                }
            }
            ':' if depth == 1 && could_be_key => {
                count += 1;
                could_be_key = false;
            }
            ',' if depth == 1 => could_be_key = false,
            _ => {}
        }
    }
    None
}

fn count_array_entries(text: &str, key: &str) -> Option<u64> {
    let marker = format!("\"{key}\"");
    let i = text.find(&marker)?;
    let rest = &text[i + marker.len()..];
    let open = rest.find('[')?;
    let mut depth = 0_u32;
    let mut in_string = false;
    let mut escaped = false;
    let mut count = 0_u64;
    let mut has_value = false;
    for c in rest[open..].chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && in_string {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            if in_string && depth == 1 {
                has_value = true;
            }
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '[' => depth += 1,
            ']' => {
                if depth == 1 {
                    return Some(count + u64::from(has_value));
                }
                depth = depth.saturating_sub(1);
            }
            ',' if depth == 1 && has_value => {
                count += 1;
                has_value = false;
            }
            c if depth == 1 && !c.is_whitespace() => has_value = true,
            _ => {}
        }
    }
    None
}

fn estimate_tokens(prompt: &str) -> u64 {
    let words = prompt.split_whitespace().count() as u64;
    let byte_estimate = (prompt.len() as u64).div_ceil(4);
    words.max(byte_estimate).max(u64::from(!prompt.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn tokenizer_plan_reports_metadata_without_claiming_roundtrip() {
        let root = env::temp_dir().join(format!("blok-tokenizer-test-{}", std::process::id()));
        fs::create_dir_all(&root).expect("create tokenizer dir");
        fs::write(
            root.join("tokenizer.json"),
            r#"{"model":{"type":"BPE","vocab":{"H":0,"i":1,"Hi":2},"merges":["H i"]}}"#,
        )
        .expect("write tokenizer");
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"bos_token_id":0,"eos_token_id":1}"#,
        )
        .expect("write tokenizer config");

        let plan = TokenizerPlan::load(&root, "Hi").expect("load tokenizer plan");

        assert_eq!(plan.model_type.as_deref(), Some("BPE"));
        assert_eq!(plan.vocab_size, Some(3));
        assert_eq!(plan.merge_count, Some(1));
        assert_eq!(plan.bos_token_id, Some(0));
        assert_eq!(plan.eos_token_id, Some(1));
        assert_eq!(
            plan.prompt_roundtrip,
            "metadata_only_tokenizer_execution_pending"
        );

        let _ = fs::remove_dir_all(root);
    }
}

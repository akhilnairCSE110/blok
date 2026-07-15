#[path = "blok_runtime_error.rs"]
mod error;
#[path = "kimi_runtime.rs"]
mod kimi_runtime;
#[path = "tensor_manifest_parser.rs"]
mod manifest;
#[path = "primitives.rs"]
mod primitives;

use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub use error::{Error, Result};
use kimi_runtime::{generate_native, KimiNativeRequest};
use manifest::Manifest;
pub use primitives::KIMI_K26_TEXT;

const USAGE: &str = "usage: blok generate --model <manifest-or-dir> --prompt <text> --tokens <n>\n";

pub fn run<I, W>(args: I, out: &mut W) -> Result<()>
where
    I: IntoIterator<Item = OsString>,
    W: Write,
{
    let mut args = args.into_iter();
    let _program = args.next();
    match args.next().and_then(|s| s.into_string().ok()).as_deref() {
        Some("generate") => generate(parse_generate(args)?, out),
        Some("-h" | "--help" | "help") | None => {
            out.write_all(USAGE.as_bytes()).map_err(Error::from)
        }
        Some(other) => Err(Error::Cli(format!("unknown command: {other}"))),
    }
}

struct Generate {
    model: PathBuf,
    prompt: String,
    tokens: u64,
}

fn parse_generate<I>(args: I) -> Result<Generate>
where
    I: IntoIterator<Item = OsString>,
{
    let mut model = None;
    let mut prompt = None;
    let mut tokens = None;
    let mut args = args.into_iter();
    while let Some(flag) = args.next().and_then(|s| s.into_string().ok()) {
        let value = args
            .next()
            .ok_or_else(|| Error::Cli(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.to_string_lossy().into_owned()),
            "--tokens" => tokens = Some(parse_u64(&value.to_string_lossy(), "--tokens")?),
            other => return Err(Error::Cli(format!("unknown generate flag: {other}"))),
        }
    }
    Ok(Generate {
        model: model.ok_or_else(|| Error::Cli("--model is required".to_owned()))?,
        prompt: prompt.ok_or_else(|| Error::Cli("--prompt is required".to_owned()))?,
        tokens: tokens.ok_or_else(|| Error::Cli("--tokens is required".to_owned()))?,
    })
}

fn parse_u64(value: &str, name: &str) -> Result<u64> {
    let n = value
        .parse()
        .map_err(|_| Error::Cli(format!("{name} must be an integer")))?;
    if n == 0 {
        Err(Error::Cli(format!("{name} must be greater than zero")))
    } else {
        Ok(n)
    }
}

fn generate<W: Write>(g: Generate, out: &mut W) -> Result<()> {
    let manifest_path = find_manifest(&g.model)?;
    let manifest = Manifest::parse(&fs::read_to_string(&manifest_path)?)?;
    let response = generate_native(KimiNativeRequest {
        manifest: &manifest,
        manifest_path: &manifest_path,
        prompt: &g.prompt,
        max_tokens: g.tokens,
    })?;
    writeln!(
        out,
        "{{\"status\":\"ok\",\"text\":{},\"tokens\":{},\"predicted_tps\":{},\"watts\":{}}}",
        json(&response.text),
        response.tokens,
        response
            .predicted_tps
            .map_or_else(|| "null".to_owned(), |tps| tps.to_string()),
        response
            .watts
            .map_or_else(|| "null".to_owned(), |watts| watts.to_string())
    )?;
    Ok(())
}

fn find_manifest(model: &Path) -> Result<PathBuf> {
    if model.is_file() {
        return Ok(model.to_path_buf());
    }
    [
        model.join("manifest.blok"),
        model.join("blok").join("manifest.blok"),
        model.join("meta").join("manifest.blok"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| Error::Manifest(format!("no manifest.blok found under {}", model.display())))
}

fn json(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::run;
    use std::ffi::OsString;

    #[test]
    fn missing_prompt_is_cli_error() {
        let args = ["blok", "generate", "--model", "m", "--tokens", "1"].map(OsString::from);
        let err = run(args, &mut Vec::new()).unwrap_err().to_string();
        assert!(err.contains("--prompt is required"));
    }

    #[test]
    fn help_is_stable() {
        let args = ["blok", "--help"].map(OsString::from);
        let mut out = Vec::new();
        run(args, &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "usage: blok generate --model <manifest-or-dir> --prompt <text> --tokens <n>\n"
        );
    }
}

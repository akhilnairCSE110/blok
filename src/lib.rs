pub mod config;
pub mod error;
pub mod graph;
pub mod manifest;
pub mod observe;

use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

pub use config::RuntimeConfig;
pub use error::{Error, Result};
pub use graph::Graph;
pub use manifest::Manifest;
pub use observe::{CommandReport, GenerateIntent};

const USAGE: &str = "usage: blok {doctor|report|inspect --manifest <path>|generate --model <path> --prompt <text> --tokens <count>}\n";

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Doctor,
    Report,
    Inspect(PathBuf),
    Generate(GenerateIntent),
    Help,
    Version,
}

pub fn run<I, W>(args: I, out: &mut W) -> Result<()>
where
    I: IntoIterator<Item = OsString>,
    W: Write,
{
    match parse(args)? {
        Command::Help => out.write_all(USAGE.as_bytes()).map_err(Error::from),
        Command::Version => {
            writeln!(out, "blok {}", env!("CARGO_PKG_VERSION")).map_err(Error::from)
        }
        Command::Doctor => {
            write_report(out, CommandReport::ok("doctor", RuntimeConfig::from_env()?))
        }
        Command::Report => {
            write_report(out, CommandReport::ok("report", RuntimeConfig::from_env()?))
        }
        Command::Inspect(path) => {
            let manifest = Manifest::parse(&fs::read_to_string(path)?)?;
            let graph = Graph::first_token(&manifest);
            write_report(
                out,
                CommandReport::inspect(RuntimeConfig::from_env()?, &manifest, &graph),
            )
        }
        Command::Generate(intent) => {
            write_report(
                out,
                CommandReport::blocked_generate(intent, RuntimeConfig::from_env()?),
            )?;
            Err(Error::Capability(
                "generate_requires_manifest_layout_arena_io_descriptors",
            ))
        }
    }
}

fn write_report(out: &mut impl Write, report: CommandReport) -> Result<()> {
    writeln!(out, "{}", report.to_json())?;
    Ok(())
}

fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = OsString>,
{
    let mut args = args.into_iter();
    let _program = args.next();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };

    match command.to_string_lossy().as_ref() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "-V" | "--version" | "version" => Ok(Command::Version),
        "doctor" => no_trailing(args, Command::Doctor),
        "report" => no_trailing(args, Command::Report),
        "inspect" => parse_inspect(args),
        "generate" => parse_generate(args),
        other => Err(Error::Cli(format!("unknown command: {other}"))),
    }
}

fn no_trailing<I>(mut args: I, command: Command) -> Result<Command>
where
    I: Iterator<Item = OsString>,
{
    if let Some(extra) = args.next() {
        return Err(Error::Cli(format!(
            "unexpected argument: {}",
            extra.to_string_lossy()
        )));
    }
    Ok(command)
}

fn parse_inspect<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = OsString>,
{
    let mut args = args.into_iter();
    match (args.next(), args.next(), args.next()) {
        (Some(flag), Some(path), None) if flag == "--manifest" => Ok(Command::Inspect(path.into())),
        _ => Err(Error::Cli("inspect requires --manifest <path>".to_owned())),
    }
}

fn parse_generate<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = OsString>,
{
    let mut model = None;
    let mut prompt = None;
    let mut tokens = None;
    let mut args = args.into_iter();

    while let Some(flag) = args.next() {
        let flag = flag.to_string_lossy();
        let value = args
            .next()
            .ok_or_else(|| Error::Cli(format!("{flag} requires a value")))?;
        match flag.as_ref() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.to_string_lossy().into_owned()),
            "--tokens" => tokens = Some(parse_token_count(&value.to_string_lossy())?),
            other => return Err(Error::Cli(format!("unknown generate flag: {other}"))),
        }
    }

    Ok(Command::Generate(GenerateIntent {
        model: model.ok_or(Error::Cli("--model is required".to_owned()))?,
        prompt: prompt.ok_or(Error::Cli("--prompt is required".to_owned()))?,
        tokens: tokens.ok_or(Error::Cli("--tokens is required".to_owned()))?,
    }))
}

fn parse_token_count(value: &str) -> Result<u32> {
    let tokens = value
        .parse::<u32>()
        .map_err(|_| Error::Cli(format!("invalid token count: {value}")))?;
    if tokens == 0 {
        Err(Error::Cli("--tokens must be greater than zero".to_owned()))
    } else {
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os_args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn parse_generate_requires_positive_tokens() {
        let err = parse(os_args(&[
            "blok", "generate", "--model", "m", "--prompt", "hi", "--tokens", "0",
        ]))
        .unwrap_err();

        assert_eq!(
            err.to_string(),
            "cli error: --tokens must be greater than zero"
        );
    }

    #[test]
    fn parse_generate_accepts_unordered_required_flags() {
        let command = parse(os_args(&[
            "blok",
            "generate",
            "--tokens",
            "1",
            "--prompt",
            "Hi",
            "--model",
            "model.gguf",
        ]))
        .unwrap();

        assert_eq!(
            command,
            Command::Generate(GenerateIntent {
                model: PathBuf::from("model.gguf"),
                prompt: "Hi".to_owned(),
                tokens: 1,
            })
        );
    }
}

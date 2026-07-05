#[path = "arena.rs"]
pub mod arena;
#[path = "runtime_environment_config.rs"]
pub mod config;
#[path = "blok_runtime_error.rs"]
pub mod error;
#[path = "first_token_execution_graph.rs"]
pub mod graph;
#[path = "linux_hardware_probe_report.rs"]
pub mod hardware;
#[path = "io.rs"]
pub mod io;
#[path = "tensor_manifest_parser.rs"]
pub mod manifest;
#[path = "command_report_json.rs"]
pub mod observe;
#[path = "tokenizer.rs"]
pub mod tokenizer;

use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub use arena::ArenaPlan;
pub use config::RuntimeConfig;
pub use error::{Error, Result};
pub use graph::Graph;
pub use hardware::HardwareReport;
pub use io::{DirectIoProbe, TransferPlan};
pub use manifest::Manifest;
pub use observe::{CommandReport, GenerateIntent};
pub use tokenizer::TokenizerPlan;

const USAGE: &str = "usage: blok {doctor|report|inspect --manifest <path>|generate --model <path> --prompt <text> --tokens <count> [--agents <count>] [--context <tokens>]}\n";

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
            let config = RuntimeConfig::from_env()?;
            let manifest_path = find_manifest(&intent.model)?;
            let manifest = Manifest::parse(&fs::read_to_string(manifest_path)?)?;
            let graph = Graph::first_token(&manifest);
            let arena = ArenaPlan::first_token(&graph)?;
            let transfers = TransferPlan::first_token(&manifest, &graph)?;
            let tokenizer = TokenizerPlan::load(&intent.model, &intent.prompt)?;
            validate_schedule(&intent, &graph)?;
            DirectIoProbe::first_window(&transfers).run()?;
            write_report(
                out,
                CommandReport::generate_descriptors(
                    intent, config, &manifest, &graph, &arena, &transfers, &tokenizer,
                ),
            )?;
            Err(Error::Capability("generate_requires_cuda_token_emission"))
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
    let mut agents = 1;
    let mut context = None;
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
            "--agents" => agents = parse_agent_count(&value.to_string_lossy())?,
            "--context" => context = Some(parse_token_count(&value.to_string_lossy())?),
            other => return Err(Error::Cli(format!("unknown generate flag: {other}"))),
        }
    }

    let tokens = tokens.ok_or(Error::Cli("--tokens is required".to_owned()))?;
    Ok(Command::Generate(GenerateIntent {
        model: model.ok_or(Error::Cli("--model is required".to_owned()))?,
        prompt: prompt.ok_or(Error::Cli("--prompt is required".to_owned()))?,
        tokens,
        agents,
        context: context.unwrap_or(tokens),
    }))
}

fn validate_schedule(intent: &GenerateIntent, graph: &Graph) -> Result<()> {
    intent
        .context
        .checked_mul(u64::from(intent.agents))
        .and_then(|_| {
            graph
                .scheduled_bytes()
                .checked_mul(u64::from(intent.agents))
        })
        .ok_or(Error::Capability("schedule_counter_overflow"))?;
    Ok(())
}

fn parse_token_count(value: &str) -> Result<u64> {
    let tokens = value
        .parse::<u64>()
        .map_err(|_| Error::Cli(format!("invalid token count: {value}")))?;
    if tokens == 0 {
        Err(Error::Cli(
            "token counts must be greater than zero".to_owned(),
        ))
    } else {
        Ok(tokens)
    }
}

fn parse_agent_count(value: &str) -> Result<u32> {
    let agents = value
        .parse::<u32>()
        .map_err(|_| Error::Cli(format!("invalid agent count: {value}")))?;
    if agents == 0 {
        Err(Error::Cli("--agents must be greater than zero".to_owned()))
    } else {
        Ok(agents)
    }
}

fn find_manifest(model: &Path) -> Result<PathBuf> {
    if model.is_file() {
        return Ok(model.to_path_buf());
    }
    if !model.exists() {
        return Err(Error::Manifest(format!(
            "model path does not exist: {}",
            model.display()
        )));
    }

    [
        model.join("manifest.blok"),
        model.join("blok").join("manifest.blok"),
        model.join("blok").join("repacked").join("manifest.blok"),
        model.join("meta").join("manifest.blok"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| {
        Error::Manifest(format!(
            "no manifest.blok found under {}; run scripts/model_fetch.py kimi-k2.6 materialize first",
            model.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn generate_compiles_descriptors_before_capability_error() {
        let root = env::temp_dir().join(format!("blok-generate-test-{}", std::process::id()));
        let manifest_dir = root.join("blok");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            manifest_dir.join("manifest.blok"),
            concat!(
                "blok-manifest-v1\n",
                "architecture=hybrid\n",
                "layout=repacked\n",
                "tensor model.embed_tokens.weight resident bf16 1x2048 0 4096 4096\n",
                "tensor layer.0.dense_ffn.neuron.0 dense_ffn_rowcol bf16 1x2048 4096 4096 4096\n",
                "tensor model.layers.0.mlp.experts.0.w1.weight routed_expert bf16 1x2048 8192 4096 4096\n",
                "tensor model.layers.0.mlp.experts.1.w1.weight routed_expert bf16 1x2048 12288 4096 4096\n"
            ),
        )
        .expect("write manifest");
        fs::write(
            root.join("tokenizer.json"),
            r#"{"model":{"type":"BPE","vocab":{"H":0,"i":1},"merges":[]}}"#,
        )
        .expect("write tokenizer");
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"bos_token_id":0,"eos_token_id":1}"#,
        )
        .expect("write tokenizer config");

        let mut out = Vec::new();
        let result = run(
            [
                OsString::from("blok"),
                OsString::from("generate"),
                OsString::from("--model"),
                root.as_os_str().to_owned(),
                OsString::from("--prompt"),
                OsString::from("Hi"),
                OsString::from("--tokens"),
                OsString::from("1000000"),
                OsString::from("--agents"),
                OsString::from("8"),
                OsString::from("--context"),
                OsString::from("1000000"),
            ],
            &mut out,
        );

        assert!(matches!(
            result,
            Err(Error::Capability("generate_requires_cuda_token_emission"))
        ));
        let report = String::from_utf8(out).expect("utf8 report");
        assert!(report.contains(
            "\"decision\":\"direct_io_cuda_and_tokenizer_required_before_payload_reads\""
        ));
        assert!(report.contains("\"scheduled_bytes\""));
        assert!(report.contains("\"arena\""));
        assert!(report.contains("\"io\""));
        assert!(report.contains("\"tokenizer\""));
        assert!(report.contains("\"model_type\":\"BPE\""));
        assert!(report.contains("\"prompt_roundtrip\":\"metadata_only_tokenizer_execution_pending\""));
        assert!(report.contains("\"top_k\":4"));
        assert!(report.contains("\"agents\":8"));
        assert!(report.contains("\"context_tokens\":1000000"));

        let _ = fs::remove_dir_all(root);
    }
}

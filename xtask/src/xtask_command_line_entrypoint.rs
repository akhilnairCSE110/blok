use std::env;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("ci") => run("scripts/ci.sh", args),
        Some("help") | None => {
            eprintln!("usage: cargo run -p xtask -- ci <ci-command>");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown xtask command: {other}");
            ExitCode::from(64)
        }
    }
}

fn run(program: &str, args: impl IntoIterator<Item = String>) -> ExitCode {
    let status = match Command::new(program).args(args).status() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("failed to run {program}: {error}");
            return ExitCode::from(74);
        }
    };
    match status.code() {
        Some(code) => ExitCode::from(code as u8),
        None => ExitCode::from(74),
    }
}

use std::env;
use std::io::stdout;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut out = stdout().lock();

    match blok::run(env::args_os(), &mut out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(error.exit_code() as u8)
        }
    }
}

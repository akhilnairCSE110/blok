use std::fmt::{self, Display, Formatter};
use std::io;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Capability(&'static str),
    Cli(String),
    Config(String),
    Manifest(String),
    Io(io::Error),
}

impl Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Cli(_) => 64,
            Self::Config(_) => 78,
            Self::Capability(_) => 78,
            Self::Manifest(_) => 65,
            Self::Io(_) => 74,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capability(code) => write!(f, "capability error: {code}"),
            Self::Cli(message) => write!(f, "cli error: {message}"),
            Self::Config(message) => write!(f, "config error: {message}"),
            Self::Manifest(message) => write!(f, "manifest error: {message}"),
            Self::Io(error) => write!(f, "io error: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

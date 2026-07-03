use std::ops::Range;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Dense,
    Gqa,
    Moe,
    Mla,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    Source,
    Sidecar,
    Repacked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    I8,
    U8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub role: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub source: Range<u64>,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    pub architecture: Architecture,
    pub layout: Layout,
    pub tensors: Vec<Tensor>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let mut architecture = None;
        let mut layout = None;
        let mut tensors = Vec::new();

        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line == "blok-manifest-v1" {
                continue;
            }
            if let Some(value) = line.strip_prefix("architecture=") {
                architecture = Some(parse_arch(value)?);
            } else if let Some(value) = line.strip_prefix("layout=") {
                layout = Some(parse_layout(value)?);
            } else if let Some(rest) = line.strip_prefix("tensor ") {
                tensors.push(parse_tensor(rest, line_no + 1)?);
            } else {
                return Err(Error::Manifest(format!("line {} is unknown", line_no + 1)));
            }
        }

        let manifest = Self {
            architecture: architecture
                .ok_or_else(|| Error::Manifest("architecture is required".to_owned()))?,
            layout: layout.ok_or_else(|| Error::Manifest("layout is required".to_owned()))?,
            tensors,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn payload_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.source.end - t.source.start)
            .sum()
    }

    pub fn max_alignment(&self) -> u64 {
        self.tensors.iter().map(|t| t.alignment).max().unwrap_or(1)
    }

    fn validate(&self) -> Result<()> {
        if self.tensors.is_empty() {
            return Err(Error::Manifest(
                "at least one tensor is required".to_owned(),
            ));
        }
        for tensor in &self.tensors {
            if tensor.source.start >= tensor.source.end {
                return Err(Error::Manifest(format!(
                    "{} has an empty range",
                    tensor.name
                )));
            }
            if tensor.alignment == 0 || !tensor.alignment.is_power_of_two() {
                return Err(Error::Manifest(format!(
                    "{} alignment must be a power of two",
                    tensor.name
                )));
            }
            if tensor.source.start % tensor.alignment != 0 {
                return Err(Error::Manifest(format!(
                    "{} offset is unaligned",
                    tensor.name
                )));
            }
        }
        Ok(())
    }
}

impl Architecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Gqa => "gqa",
            Self::Moe => "moe",
            Self::Mla => "mla",
            Self::Hybrid => "hybrid",
        }
    }
}

impl Layout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Sidecar => "sidecar",
            Self::Repacked => "repacked",
        }
    }
}

fn parse_tensor(rest: &str, line_no: usize) -> Result<Tensor> {
    let parts: Vec<_> = rest.split_whitespace().collect();
    if parts.len() != 7 {
        return Err(Error::Manifest(format!(
            "line {line_no} tensor requires 7 fields"
        )));
    }
    let offset = u64_field(parts[4], "offset")?;
    let size = u64_field(parts[5], "size")?;
    let end = offset
        .checked_add(size)
        .ok_or_else(|| Error::Manifest(format!("line {line_no} tensor range overflows")))?;
    Ok(Tensor {
        name: parts[0].to_owned(),
        role: parts[1].to_owned(),
        dtype: parse_dtype(parts[2])?,
        shape: parts[3]
            .split('x')
            .map(|v| u64_field(v, "shape"))
            .collect::<Result<_>>()?,
        source: offset..end,
        alignment: u64_field(parts[6], "alignment")?,
    })
}

fn u64_field(value: &str, name: &str) -> Result<u64> {
    value
        .parse()
        .map_err(|_| Error::Manifest(format!("{name} must be an integer")))
}

fn parse_arch(value: &str) -> Result<Architecture> {
    match value {
        "dense" => Ok(Architecture::Dense),
        "gqa" => Ok(Architecture::Gqa),
        "moe" => Ok(Architecture::Moe),
        "mla" => Ok(Architecture::Mla),
        "hybrid" => Ok(Architecture::Hybrid),
        _ => Err(Error::Manifest(format!("unknown architecture: {value}"))),
    }
}

fn parse_layout(value: &str) -> Result<Layout> {
    match value {
        "source" => Ok(Layout::Source),
        "sidecar" => Ok(Layout::Sidecar),
        "repacked" => Ok(Layout::Repacked),
        _ => Err(Error::Manifest(format!("unknown layout: {value}"))),
    }
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value {
        "bf16" => Ok(DType::Bf16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "i8" => Ok(DType::I8),
        "u8" => Ok(DType::U8),
        _ => Err(Error::Manifest(format!("unknown dtype: {value}"))),
    }
}

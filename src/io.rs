use std::process::Command as ProcessCommand;

use crate::{Error, Graph, Manifest, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferWindow {
    pub tensor: String,
    pub file: Option<String>,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub arena_offset: u64,
    pub backend: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferPlan {
    pub windows: Vec<TransferWindow>,
    pub scheduled_bytes: u64,
    pub max_alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectIoProbe {
    pub file: Option<String>,
    pub offset: u64,
    pub bytes: u64,
    pub block: u64,
}

impl TransferPlan {
    pub fn first_token(manifest: &Manifest, graph: &Graph) -> Result<Self> {
        if manifest.tensors.len() != graph.ops.len() {
            return Err(Error::Capability("manifest_graph_descriptor_mismatch"));
        }

        let mut windows = Vec::with_capacity(graph.ops.len());
        let mut arena_offset = 0_u64;
        let mut scheduled_bytes = 0_u64;
        let mut max_alignment = 1_u64;

        for (tensor, op) in manifest.tensors.iter().zip(&graph.ops) {
            if tensor.name != op.tensor {
                return Err(Error::Capability("manifest_graph_order_mismatch"));
            }
            max_alignment = max_alignment.max(tensor.alignment);
            let aligned_arena = align(arena_offset, tensor.alignment)?;
            let bytes = tensor.source.end - tensor.source.start;
            windows.push(TransferWindow {
                tensor: tensor.name.clone(),
                file: tensor.file.clone(),
                offset: tensor.source.start,
                bytes,
                alignment: tensor.alignment,
                arena_offset: aligned_arena,
                backend: "linux_odirect_file",
            });
            scheduled_bytes = scheduled_bytes
                .checked_add(bytes)
                .ok_or(Error::Capability("transfer_descriptor_overflow"))?;
            arena_offset = aligned_arena
                .checked_add(bytes)
                .ok_or(Error::Capability("transfer_arena_offset_overflow"))?;
        }

        Ok(Self {
            windows,
            scheduled_bytes,
            max_alignment,
        })
    }
}

impl DirectIoProbe {
    pub fn first_window(plan: &TransferPlan) -> Self {
        let Some(window) = plan.windows.iter().find(|w| w.file.is_some()) else {
            return Self {
                file: None,
                offset: 0,
                bytes: 0,
                block: 4096,
            };
        };
        let block = window.alignment.max(4096);
        Self {
            file: window.file.clone(),
            offset: window.offset,
            bytes: window.bytes.min(block),
            block,
        }
    }

    pub fn run(&self) -> Result<()> {
        let Some(file) = &self.file else {
            return Ok(());
        };
        if self.bytes == 0
            || !self.offset.is_multiple_of(self.block)
            || !self.bytes.is_multiple_of(self.block)
        {
            return Err(Error::Capability("direct_io_probe_window_unaligned"));
        }
        let status = ProcessCommand::new("dd")
            .args([
                &format!("if={file}"),
                "of=/dev/null",
                "iflag=direct",
                &format!("bs={}", self.block),
                &format!("skip={}", self.offset / self.block),
                &format!("count={}", self.bytes / self.block),
                "status=none",
            ])
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(Error::Capability("direct_io_probe_failed"))
        }
    }
}

fn align(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::Capability("transfer_alignment_invalid"));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(Error::Capability("transfer_alignment_overflow"))
}

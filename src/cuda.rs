use crate::{Error, Result, TransferPlan};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaPlan {
    pub target_sm: &'static str,
    pub entry: &'static str,
    pub source: &'static str,
    pub inline_ptx: &'static [&'static str],
    pub input_bytes: u64,
    pub vector_bytes: u64,
    pub block_threads: u32,
    pub grid_blocks: u64,
    pub source_backend: &'static str,
    pub requires_direct_storage_to_vram: bool,
}

impl CudaPlan {
    pub fn byte_probe(transfers: &TransferPlan) -> Result<Self> {
        let window = transfers
            .windows
            .iter()
            .find(|w| w.bytes > 0)
            .ok_or(Error::Capability("cuda_probe_requires_transfer_window"))?;
        if window.bytes % 16 != 0 || window.alignment < 16 {
            return Err(Error::Capability(
                "cuda_probe_requires_16_byte_aligned_window",
            ));
        }
        let block_threads = 256;
        let lanes = window.bytes / 16;
        Ok(Self {
            target_sm: "sm_120",
            entry: "blok_byte_probe_kernel",
            source: "src/cuda_byte_probe.cu",
            inline_ptx: &["ld.global.v4.u32", "ld.global.f32", "fma.rn.f32"],
            input_bytes: window.bytes,
            vector_bytes: 16,
            block_threads,
            grid_blocks: lanes.div_ceil(u64::from(block_threads)),
            source_backend: window.backend,
            requires_direct_storage_to_vram: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{TransferPlan, TransferWindow};

    #[test]
    fn byte_probe_describes_sm120_vectorized_ptx() {
        let plan = CudaPlan::byte_probe(&TransferPlan {
            windows: vec![TransferWindow {
                tensor: "tensor".to_owned(),
                file: Some("model.safetensors".to_owned()),
                offset: 0,
                bytes: 4096,
                alignment: 4096,
                arena_offset: 0,
                backend: "ugds_nvme_to_vram",
            }],
            scheduled_bytes: 4096,
            max_alignment: 4096,
        })
        .expect("cuda plan");

        assert_eq!(plan.target_sm, "sm_120");
        assert_eq!(plan.entry, "blok_byte_probe_kernel");
        assert_eq!(plan.vector_bytes, 16);
        assert_eq!(plan.grid_blocks, 1);
        assert!(plan.inline_ptx.contains(&"ld.global.v4.u32"));
        assert!(plan.inline_ptx.contains(&"fma.rn.f32"));
    }
}

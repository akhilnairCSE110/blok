use crate::{Error, Graph, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArenaTier {
    Vram,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaView {
    pub name: String,
    pub tier: ArenaTier,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub lifetime: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaPlan {
    pub views: Vec<ArenaView>,
    pub reserved_bytes: u64,
    pub max_alignment: u64,
}

impl ArenaPlan {
    pub fn first_token(graph: &Graph) -> Result<Self> {
        let mut views = Vec::with_capacity(graph.ops.len());
        let mut offset = 0_u64;
        let mut max_alignment = 1_u64;

        for op in &graph.ops {
            max_alignment = max_alignment.max(op.alignment);
            let aligned = align(offset, op.alignment)?;
            let end = aligned
                .checked_add(op.bytes)
                .ok_or(Error::Capability("arena_descriptor_overflow"))?;
            views.push(ArenaView {
                name: op.arena.to_owned(),
                tier: ArenaTier::Vram,
                offset: aligned,
                bytes: op.bytes,
                alignment: op.alignment,
                lifetime: op.dependency,
            });
            offset = end;
        }

        Ok(Self {
            views,
            reserved_bytes: offset,
            max_alignment,
        })
    }
}

fn align(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::Capability("arena_alignment_invalid"));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(Error::Capability("arena_alignment_overflow"))
}

use std::collections::BTreeSet;

use crate::manifest::{Architecture, Manifest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Op {
    pub tensor: String,
    pub layer: Option<u32>,
    pub neuron: Option<u32>,
    pub expert: Option<u32>,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub arena: &'static str,
    pub consumer: &'static str,
    pub dependency: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub ops: Vec<Op>,
    pub resident_bytes: u64,
    pub dense_bytes: u64,
    pub expert_bytes: u64,
    pub dense_neurons: u32,
    pub expert_tensors: u32,
    pub top_k: u32,
    pub expert_layers: u32,
}

impl Graph {
    pub fn first_token(manifest: &Manifest) -> Self {
        let top_k = u32::from(matches!(
            manifest.architecture,
            Architecture::Moe | Architecture::Hybrid
        )) * 4;
        let mut graph = Self {
            ops: Vec::with_capacity(manifest.tensors.len()),
            resident_bytes: 0,
            dense_bytes: 0,
            expert_bytes: 0,
            dense_neurons: 0,
            expert_tensors: 0,
            top_k,
            expert_layers: 0,
        };
        let mut expert_layers = BTreeSet::new();
        for tensor in &manifest.tensors {
            let (arena, consumer, dependency) = route(&tensor.role);
            let bytes = tensor.source.end - tensor.source.start;
            match tensor.role.as_str() {
                "routed_expert" => {
                    graph.expert_bytes += bytes;
                    graph.expert_tensors += 1;
                    if let Some(layer) = layer(&tensor.name) {
                        expert_layers.insert(layer);
                    }
                }
                "dense_ffn_rowcol" => {
                    graph.dense_bytes += bytes;
                    graph.dense_neurons += 1;
                }
                _ => graph.resident_bytes += bytes,
            }
            graph.ops.push(Op {
                tensor: tensor.name.clone(),
                layer: layer(&tensor.name),
                neuron: neuron(&tensor.name),
                expert: expert(&tensor.name),
                offset: tensor.source.start,
                bytes,
                alignment: tensor.alignment,
                arena,
                consumer,
                dependency,
            });
        }
        graph.expert_layers = expert_layers.len() as u32;
        graph
    }

    pub fn payload_bytes(&self) -> u64 {
        self.resident_bytes + self.dense_bytes + self.expert_bytes
    }

    pub fn scheduled_bytes(&self) -> u64 {
        self.resident_bytes + self.dense_bytes + self.expert_bytes.min(self.expert_bytes_per_k())
    }

    pub fn skipped_expert_bytes(&self) -> u64 {
        self.expert_bytes.saturating_sub(self.expert_bytes_per_k())
    }

    fn expert_bytes_per_k(&self) -> u64 {
        if self.top_k == 0 || self.expert_layers == 0 {
            0
        } else {
            (self.expert_bytes / u64::from(self.expert_layers))
                .saturating_mul(u64::from(self.top_k))
        }
    }
}

fn route(role: &str) -> (&'static str, &'static str, &'static str) {
    match role {
        "routed_expert" => (
            "vram.expert_slot",
            "moe.grouped_gemm",
            "router_top_k_before_fetch",
        ),
        "dense_ffn_rowcol" => ("vram.ffn_slot", "ffn.row_column_bundle", "layer_prefetch"),
        _ => ("vram.resident", "resident_weight", "model_open"),
    }
}

fn layer(name: &str) -> Option<u32> {
    for marker in ["model.layers.", "layers.", "layer."] {
        if let Some(i) = name.find(marker) {
            let s = &name[i + marker.len()..];
            let n: String = s.chars().take_while(char::is_ascii_digit).collect();
            return n.parse().ok();
        }
    }
    None
}

fn neuron(name: &str) -> Option<u32> {
    number_after(name, ".neuron.")
}

fn expert(name: &str) -> Option<u32> {
    number_after(name, ".experts.").or_else(|| number_after(name, ".expert."))
}

fn number_after(name: &str, marker: &str) -> Option<u32> {
    let i = name.find(marker)?;
    let s = &name[i + marker.len()..];
    let n: String = s.chars().take_while(char::is_ascii_digit).collect();
    n.parse().ok()
}

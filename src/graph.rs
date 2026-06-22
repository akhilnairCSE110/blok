use crate::manifest::Manifest;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Op {
    pub tensor: String,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub arena: &'static str,
    pub lifetime: &'static str,
    pub consumer: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn first_token(manifest: &Manifest) -> Self {
        Self {
            ops: manifest
                .tensors
                .iter()
                .map(|t| Op {
                    tensor: t.name.clone(),
                    offset: t.source.start,
                    bytes: t.source.end - t.source.start,
                    alignment: t.alignment,
                    arena: "vram.weights",
                    lifetime: "first_token",
                    consumer: "decode.prefill",
                })
                .collect(),
        }
    }

    pub fn payload_bytes(&self) -> u64 {
        self.ops.iter().map(|op| op.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    #[test]
    fn graph_preserves_declared_payload_ranges() {
        let manifest = Manifest::parse(
            "architecture=dense\nlayout=sidecar\ntensor w weight bf16 4 0 8 4096\n",
        )
        .unwrap();
        let graph = Graph::first_token(&manifest);

        assert_eq!(graph.payload_bytes(), 8);
        assert_eq!(graph.ops[0].arena, "vram.weights");
    }
}

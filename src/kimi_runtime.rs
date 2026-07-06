use crate::{Error, Manifest, Result, KIMI_K26_TEXT};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiNativeRequest<'a> {
    pub manifest: &'a Manifest,
    pub prompt: &'a str,
    pub max_tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiNativeResponse {
    pub text: String,
    pub tokens: u64,
    pub predicted_tps: f64,
    pub watts: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KimiExecutionPlan {
    pub resident_router_bytes: u64,
    pub resident_shared_expert_bytes: u64,
    pub resident_attention_bytes: u64,
    pub streamed_expert_bytes_per_token_floor: u64,
    pub top_k: u32,
    pub expert_cache_policy: &'static str,
    pub io_policy: &'static str,
    pub kernel_policy: &'static str,
    pub max_concurrent_instances: u32,
    pub decode_batch_cap: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutedToken {
    pub request: u32,
    pub token: u32,
    pub experts: [u16; 8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertBatch {
    pub expert: u16,
    pub tokens: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerRouteSchedule {
    pub layer: u32,
    pub batches: Vec<ExpertBatch>,
    pub unique_experts: u32,
}

impl KimiExecutionPlan {
    pub fn from_manifest(manifest: &Manifest) -> Self {
        let mut resident_router_bytes = 0_u64;
        let mut resident_shared_expert_bytes = 0_u64;
        let mut resident_attention_bytes = 0_u64;
        let mut routed_expert_bytes = 0_u64;
        let mut routed_expert_tensors = 0_u64;

        for tensor in &manifest.tensors {
            let bytes = tensor.source.end - tensor.source.start;
            match tensor.role.as_str() {
                "router" => resident_router_bytes += bytes,
                "shared_expert_resident" => resident_shared_expert_bytes += bytes,
                "attention_resident" => resident_attention_bytes += bytes,
                "routed_expert" => {
                    routed_expert_bytes += bytes;
                    routed_expert_tensors += 1;
                }
                _ => {}
            }
        }

        let bytes_per_expert_tensor = routed_expert_bytes
            .checked_div(routed_expert_tensors)
            .unwrap_or(0);
        let streamed_expert_bytes_per_token_floor =
            bytes_per_expert_tensor * u64::from(KIMI_K26_TEXT.experts_per_token) * 60;

        Self {
            resident_router_bytes,
            resident_shared_expert_bytes,
            resident_attention_bytes,
            streamed_expert_bytes_per_token_floor,
            top_k: KIMI_K26_TEXT.experts_per_token,
            expert_cache_policy: "layer_partitioned_lfu_lru_hbm_cache",
            io_policy: "host_issued_gds_or_odirect_into_registered_gpu_slabs",
            kernel_policy: "grouped_router_sorted_expert_decode",
            max_concurrent_instances: concurrency_from_cache_budget(),
            decode_batch_cap: 48,
        }
    }
}

impl LayerRouteSchedule {
    pub fn from_tokens(layer: u32, tokens: &[RoutedToken]) -> Self {
        let mut per_expert: Vec<ExpertBatch> = (0..KIMI_K26_TEXT.routed_experts)
            .map(|expert| ExpertBatch {
                expert: expert as u16,
                tokens: Vec::new(),
            })
            .collect();
        for token in tokens {
            for expert in token.experts {
                per_expert[usize::from(expert)]
                    .tokens
                    .push((token.request, token.token));
            }
        }
        let batches: Vec<_> = per_expert
            .into_iter()
            .filter(|batch| !batch.tokens.is_empty())
            .collect();
        Self {
            layer,
            unique_experts: batches.len() as u32,
            batches,
        }
    }

    pub fn cold_expert_pressure(&self, hot_experts_per_layer: u32) -> u32 {
        self.unique_experts.saturating_sub(hot_experts_per_layer)
    }
}

pub fn generate_native(request: KimiNativeRequest<'_>) -> Result<KimiNativeResponse> {
    validate_request(&request)?;
    let _plan = KimiExecutionPlan::from_manifest(request.manifest);
    Err(Error::Capability("native_kimi_executor_incomplete"))
}

fn concurrency_from_cache_budget() -> u32 {
    std::env::var("BLOK_MAX_CONCURRENT_INSTANCES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn validate_request(request: &KimiNativeRequest<'_>) -> Result<()> {
    if request.max_tokens == 0 {
        return Err(Error::Cli("--tokens must be greater than zero".to_owned()));
    }
    if request.prompt.trim().is_empty() {
        return Err(Error::Cli("--prompt must not be empty".to_owned()));
    }
    if request.manifest.tensors.is_empty() {
        return Err(Error::Manifest("manifest has no tensors".to_owned()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    #[test]
    fn execution_plan_keeps_routers_resident_and_streams_routed_experts() {
        let manifest = Manifest::parse(concat!(
            "blok-manifest-v1\n",
            "architecture=hybrid\n",
            "layout=sidecar\n",
            "tensor model.layers.1.mlp.gate.weight router bf16 384x7168 0 4096 4096\n",
            "tensor model.layers.1.self_attn.q_a_proj.weight attention_resident bf16 1x2048 4096 4096 4096\n",
            "tensor model.layers.1.mlp.shared_experts.w1.weight shared_expert_resident bf16 1x2048 8192 4096 4096\n",
            "tensor model.layers.1.mlp.experts.0.w1.weight routed_expert bf16 1x2048 12288 4096 4096\n",
            "tensor model.layers.1.mlp.experts.1.w1.weight routed_expert bf16 1x2048 16384 4096 4096\n",
            "tensor model.layers.1.mlp.experts.2.w1.weight routed_expert bf16 1x2048 20480 4096 4096\n",
            "tensor model.layers.1.mlp.experts.3.w1.weight routed_expert bf16 1x2048 24576 4096 4096\n",
            "tensor model.layers.1.mlp.experts.4.w1.weight routed_expert bf16 1x2048 28672 4096 4096\n",
            "tensor model.layers.1.mlp.experts.5.w1.weight routed_expert bf16 1x2048 32768 4096 4096\n",
            "tensor model.layers.1.mlp.experts.6.w1.weight routed_expert bf16 1x2048 36864 4096 4096\n",
            "tensor model.layers.1.mlp.experts.7.w1.weight routed_expert bf16 1x2048 40960 4096 4096\n"
        ))
        .expect("manifest");

        let plan = KimiExecutionPlan::from_manifest(&manifest);

        assert_eq!(plan.top_k, 8);
        assert_eq!(plan.resident_router_bytes, 4096);
        assert_eq!(plan.resident_attention_bytes, 4096);
        assert_eq!(plan.resident_shared_expert_bytes, 4096);
        assert_eq!(plan.streamed_expert_bytes_per_token_floor, 4096 * 8 * 60);
        assert_eq!(
            plan.io_policy,
            "host_issued_gds_or_odirect_into_registered_gpu_slabs"
        );
        assert_eq!(plan.decode_batch_cap, 48);
    }

    #[test]
    fn route_schedule_groups_tokens_by_expert_for_grouped_gemm() {
        let schedule = LayerRouteSchedule::from_tokens(
            7,
            &[
                RoutedToken {
                    request: 0,
                    token: 10,
                    experts: [1, 2, 3, 4, 5, 6, 7, 8],
                },
                RoutedToken {
                    request: 1,
                    token: 20,
                    experts: [1, 2, 9, 10, 11, 12, 13, 14],
                },
            ],
        );

        assert_eq!(schedule.layer, 7);
        assert_eq!(schedule.unique_experts, 14);
        assert_eq!(schedule.cold_expert_pressure(8), 6);
        let expert_one = schedule
            .batches
            .iter()
            .find(|batch| batch.expert == 1)
            .expect("expert 1 batch");
        assert_eq!(expert_one.tokens, vec![(0, 10), (1, 20)]);
    }
}

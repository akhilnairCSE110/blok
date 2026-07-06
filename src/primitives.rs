#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiK26TextSpec {
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope: u32,
    pub qk_rope: u32,
    pub v_head: u32,
    pub routed_experts: u32,
    pub experts_per_token: u32,
    pub moe_hidden: u32,
    pub vocab: u32,
    pub max_context: u32,
}

pub const KIMI_K26_TEXT: KimiK26TextSpec = KimiK26TextSpec {
    layers: 61,
    hidden: 7168,
    heads: 64,
    q_lora_rank: 1536,
    kv_lora_rank: 512,
    qk_nope: 128,
    qk_rope: 64,
    v_head: 128,
    routed_experts: 384,
    experts_per_token: 8,
    moe_hidden: 2048,
    vocab: 163840,
    max_context: 262144,
};

pub const KIMI_K26_DECODE: &[&str] = &[
    "direct_read",
    "rmsnorm_int4_matvec",
    "mla_qkv",
    "yarn_rope",
    "flash_attention",
    "router_topk",
    "moe_silu_weighted_sum",
    "bf16_matmul",
    "residual",
    "sample_argmax",
];

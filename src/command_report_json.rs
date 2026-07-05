use std::path::PathBuf;

use crate::{
    ArenaPlan, Graph, HardwareReport, Manifest, RuntimeConfig, TokenizerPlan, TransferPlan,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateIntent {
    pub model: PathBuf,
    pub prompt: String,
    pub tokens: u64,
    pub agents: u32,
    pub context: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandReport {
    command: &'static str,
    status: &'static str,
    decision: &'static str,
    generate: Option<GenerateIntent>,
    hardware: HardwareReport,
    manifest: Option<String>,
    graph: Option<String>,
    arena: Option<String>,
    io: Option<String>,
    tokenizer: Option<String>,
    schedule: Option<String>,
    config: RuntimeConfig,
}

impl CommandReport {
    pub fn ok(command: &'static str, config: RuntimeConfig) -> Self {
        Self {
            command,
            status: "ok",
            decision: "diagnostic_only_no_payload_bytes_touched",
            generate: None,
            hardware: HardwareReport::probe(),
            manifest: None,
            graph: None,
            arena: None,
            io: None,
            tokenizer: None,
            schedule: None,
            config,
        }
    }

    pub fn inspect(config: RuntimeConfig, manifest: &Manifest, graph: &Graph) -> Self {
        Self {
            command: "inspect",
            status: "ok",
            decision: "metadata_only_payload_bytes_not_touched",
            generate: None,
            hardware: HardwareReport::probe(),
            manifest: Some(format!(
                concat!(
                    "{{\"architecture\":{},\"layout\":{},\"tensors\":{},",
                    "\"payload_bytes\":{},\"max_alignment\":{},",
                    "\"graph_ops\":{},\"graph_payload_bytes\":{}}}"
                ),
                j(manifest.architecture.as_str()),
                j(manifest.layout.as_str()),
                manifest.tensors.len(),
                manifest.payload_bytes(),
                manifest.max_alignment(),
                graph.ops.len(),
                graph.payload_bytes()
            )),
            graph: Some(graph_json(graph)),
            arena: None,
            io: None,
            tokenizer: None,
            schedule: None,
            config,
        }
    }

    pub fn generate_descriptors(
        generate: GenerateIntent,
        config: RuntimeConfig,
        manifest: &Manifest,
        graph: &Graph,
        arena: &ArenaPlan,
        transfers: &TransferPlan,
        tokenizer: &TokenizerPlan,
    ) -> Self {
        let schedule = schedule_json(&generate, graph);
        Self {
            command: "generate",
            status: "blocked",
            decision: "direct_io_cuda_and_tokenizer_required_before_payload_reads",
            generate: Some(generate),
            hardware: HardwareReport::probe(),
            manifest: Some(format!(
                concat!(
                    "{{\"architecture\":{},\"layout\":{},\"tensors\":{},",
                    "\"payload_bytes\":{},\"max_alignment\":{}}}"
                ),
                j(manifest.architecture.as_str()),
                j(manifest.layout.as_str()),
                manifest.tensors.len(),
                manifest.payload_bytes(),
                manifest.max_alignment(),
            )),
            graph: Some(graph_json(graph)),
            arena: Some(arena_json(arena)),
            io: Some(io_json(transfers)),
            tokenizer: Some(tokenizer_json(tokenizer)),
            schedule: Some(schedule),
            config,
        }
    }

    pub fn to_json(&self) -> String {
        let generate = self.generate.as_ref().map_or(String::new(), |g| {
            format!(
                ",\"generate\":{{\"model\":{},\"prompt\":{},\"tokens\":{},\"agents\":{},\"context\":{}}}",
                j(&g.model.display().to_string()),
                j(&g.prompt),
                g.tokens,
                g.agents,
                g.context
            )
        });
        let manifest = self
            .manifest
            .as_ref()
            .map_or(String::new(), |m| format!(",\"manifest\":{m}"));
        let graph = self
            .graph
            .as_ref()
            .map_or(String::new(), |m| format!(",\"graph\":{m}"));
        let arena = self
            .arena
            .as_ref()
            .map_or(String::new(), |m| format!(",\"arena\":{m}"));
        let io = self
            .io
            .as_ref()
            .map_or(String::new(), |m| format!(",\"io\":{m}"));
        let tokenizer = self
            .tokenizer
            .as_ref()
            .map_or(String::new(), |m| format!(",\"tokenizer\":{m}"));
        let schedule = self
            .schedule
            .as_ref()
            .map_or(String::new(), |m| format!(",\"schedule\":{m}"));
        format!(
            concat!(
                "{{\"schema_version\":1,\"crate\":\"blok\",\"version\":{},",
                "\"command\":{},\"status\":{},\"decision\":{},",
                "\"config\":{{\"blok_home\":{},\"model_root\":{},\"report_root\":{},",
                "\"strict_direct_io\":{}}},\"hardware\":{}{}{}{}{}{}{}{} }}"
            ),
            j(env!("CARGO_PKG_VERSION")),
            j(self.command),
            j(self.status),
            j(self.decision),
            j(&self.config.blok_home.display().to_string()),
            j(&self.config.model_root.display().to_string()),
            j(&self.config.report_root.display().to_string()),
            self.config.strict_direct_io,
            self.hardware.json,
            manifest,
            graph,
            arena,
            io,
            tokenizer,
            schedule,
            generate
        )
    }
}

fn schedule_json(g: &GenerateIntent, graph: &Graph) -> String {
    const KV_PAGE: u64 = 128;
    let live_tokens = g.context.saturating_mul(u64::from(g.agents));
    let kv_pages = live_tokens.div_ceil(KV_PAGE);
    format!(
        concat!(
            "{{\"agents\":{},\"decode_steps\":{},\"context_tokens\":{},",
            "\"kv_page_tokens\":{},\"kv_pages\":{},\"bytes_per_decode_step\":{},",
            "\"tokenizer_required\":true,\"executor\":\"cuda_direct_io_required\"}}"
        ),
        g.agents,
        g.tokens,
        g.context,
        KV_PAGE,
        kv_pages,
        graph.scheduled_bytes() * u64::from(g.agents)
    )
}

fn graph_json(graph: &Graph) -> String {
    format!(
        concat!(
            "{{\"ops\":{},\"payload_bytes\":{},\"scheduled_bytes\":{},",
            "\"resident_bytes\":{},\"dense_bytes\":{},\"expert_bytes\":{},",
            "\"top_k\":{},\"expert_layers\":{}}}"
        ),
        graph.ops.len(),
        graph.payload_bytes(),
        graph.scheduled_bytes(),
        graph.resident_bytes,
        graph.dense_bytes,
        graph.expert_bytes,
        graph.top_k,
        graph.expert_layers
    )
}

fn arena_json(arena: &ArenaPlan) -> String {
    format!(
        "{{\"views\":{},\"reserved_bytes\":{},\"max_alignment\":{},\"tier\":\"vram\"}}",
        arena.views.len(),
        arena.reserved_bytes,
        arena.max_alignment
    )
}

fn io_json(transfers: &TransferPlan) -> String {
    let first = transfers.windows.first().map_or(String::from("null"), |w| {
        format!(
            concat!(
                "{{\"tensor\":{},\"file_present\":{},\"offset\":{},\"bytes\":{},",
                "\"alignment\":{},\"arena_offset\":{},\"backend\":{}}}"
            ),
            j(&w.tensor),
            w.file.is_some(),
            w.offset,
            w.bytes,
            w.alignment,
            w.arena_offset,
            j(w.backend)
        )
    });
    format!(
        concat!(
            "{{\"windows\":{},\"scheduled_bytes\":{},\"max_alignment\":{},",
            "\"first_window\":{},\"probe\":\"dd_iflag_direct_first_aligned_block\"}}"
        ),
        transfers.windows.len(),
        transfers.scheduled_bytes,
        transfers.max_alignment,
        first
    )
}

fn tokenizer_json(tokenizer: &TokenizerPlan) -> String {
    format!(
        concat!(
            "{{\"tokenizer_json\":{},\"tokenizer_config_json\":{},",
            "\"model_type\":{},\"vocab_size\":{},\"merge_count\":{},",
            "\"bos_token_id\":{},\"eos_token_id\":{},\"prompt_bytes\":{},",
            "\"prompt_chars\":{},\"estimated_prompt_tokens\":{},\"prompt_roundtrip\":{}}}"
        ),
        j(&tokenizer.tokenizer_json.display().to_string()),
        tokenizer
            .tokenizer_config_json
            .as_ref()
            .map(|p| j(&p.display().to_string()))
            .unwrap_or_else(|| "null".to_owned()),
        tokenizer
            .model_type
            .as_ref()
            .map(|s| j(s))
            .unwrap_or_else(|| "null".to_owned()),
        opt_u64(tokenizer.vocab_size),
        opt_u64(tokenizer.merge_count),
        opt_u64(tokenizer.bos_token_id),
        opt_u64(tokenizer.eos_token_id),
        tokenizer.prompt_bytes,
        tokenizer.prompt_chars,
        tokenizer.estimated_prompt_tokens,
        j(tokenizer.prompt_roundtrip)
    )
}

fn opt_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_owned(), |v| v.to_string())
}

fn j(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

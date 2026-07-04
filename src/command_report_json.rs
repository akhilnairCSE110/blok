use std::path::PathBuf;

use crate::{Graph, HardwareReport, Manifest, RuntimeConfig};

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
            schedule: None,
            config,
        }
    }

    pub fn generate_descriptors(
        generate: GenerateIntent,
        config: RuntimeConfig,
        manifest: &Manifest,
        graph: &Graph,
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
        let schedule = self
            .schedule
            .as_ref()
            .map_or(String::new(), |m| format!(",\"schedule\":{m}"));
        format!(
            concat!(
                "{{\"schema_version\":1,\"crate\":\"blok\",\"version\":{},",
                "\"command\":{},\"status\":{},\"decision\":{},",
                "\"config\":{{\"blok_home\":{},\"model_root\":{},\"report_root\":{},",
                "\"strict_direct_io\":{}}},\"hardware\":{}{}{}{}{} }}"
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

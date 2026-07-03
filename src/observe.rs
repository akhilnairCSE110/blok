use std::path::PathBuf;

use crate::{Graph, HardwareReport, Manifest, RuntimeConfig};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateIntent {
    pub model: PathBuf,
    pub prompt: String,
    pub tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandReport {
    command: &'static str,
    status: &'static str,
    decision: &'static str,
    generate: Option<GenerateIntent>,
    hardware: HardwareReport,
    manifest: Option<String>,
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
            config,
        }
    }

    pub fn blocked_generate(generate: GenerateIntent, config: RuntimeConfig) -> Self {
        Self {
            command: "generate",
            status: "blocked",
            decision: "manifest_layout_and_direct_io_are_required_before_payload_reads",
            generate: Some(generate),
            hardware: HardwareReport::probe(),
            manifest: None,
            config,
        }
    }

    pub fn to_json(&self) -> String {
        let generate = self.generate.as_ref().map_or(String::new(), |g| {
            format!(
                ",\"generate\":{{\"model\":{},\"prompt\":{},\"tokens\":{}}}",
                j(&g.model.display().to_string()),
                j(&g.prompt),
                g.tokens
            )
        });
        let manifest = self
            .manifest
            .as_ref()
            .map_or(String::new(), |m| format!(",\"manifest\":{m}"));
        format!(
            concat!(
                "{{\"schema_version\":1,\"crate\":\"blok\",\"version\":{},",
                "\"command\":{},\"status\":{},\"decision\":{},",
                "\"config\":{{\"blok_home\":{},\"model_root\":{},\"report_root\":{},",
                "\"strict_direct_io\":{}}},\"hardware\":{}{}{} }}"
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
            generate
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_escapes_prompt_text() {
        let config = RuntimeConfig {
            blok_home: PathBuf::from("/tmp/blok"),
            model_root: PathBuf::from("/tmp/blok/models"),
            report_root: PathBuf::from("/tmp/blok/reports"),
            strict_direct_io: true,
        };
        let report = CommandReport::blocked_generate(
            GenerateIntent {
                model: PathBuf::from("m"),
                prompt: "a \"quoted\"\nline".to_owned(),
                tokens: 1,
            },
            config,
        );

        assert!(report.to_json().contains("a \\\"quoted\\\"\\nline"));
    }
}

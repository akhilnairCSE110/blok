use std::path::PathBuf;

use crate::{FetchPlan, Graph, Manifest, RuntimeConfig};

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
    fetch: Option<String>,
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
            fetch: None,
            manifest: None,
            config,
        }
    }

    pub fn fetch(config: &RuntimeConfig, plan: &FetchPlan) -> Self {
        Self {
            command: "fetch",
            status: "running",
            decision: "download_full_kimi_k2_6_to_nvme_with_hf_xet",
            generate: None,
            fetch: Some(format!(
                concat!(
                    "{{\"repo\":{},\"revision\":{},\"local_dir\":{},\"cache_dir\":{},",
                    "\"expected_bytes\":{},\"files\":{},\"safetensors\":{}}}"
                ),
                j(plan.repo),
                j(plan.revision),
                j(&plan.local_dir.display().to_string()),
                j(&plan.cache_dir.display().to_string()),
                plan.expected_bytes,
                plan.files,
                plan.safetensors
            )),
            manifest: None,
            config: config.clone(),
        }
    }

    pub fn inspect(config: RuntimeConfig, manifest: &Manifest, graph: &Graph) -> Self {
        Self {
            command: "inspect",
            status: "ok",
            decision: "metadata_only_payload_bytes_not_touched",
            generate: None,
            fetch: None,
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
            fetch: None,
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
        let fetch = self
            .fetch
            .as_ref()
            .map_or(String::new(), |f| format!(",\"fetch\":{f}"));

        format!(
            concat!(
                "{{\"schema_version\":1,\"crate\":\"blok\",\"version\":{},",
                "\"command\":{},\"status\":{},\"decision\":{},",
                "\"config\":{{\"blok_home\":{},\"model_root\":{},\"report_root\":{},",
                "\"strict_direct_io\":{}}},\"target_host\":{}{}{}{} }}"
            ),
            j(env!("CARGO_PKG_VERSION")),
            j(self.command),
            j(self.status),
            j(self.decision),
            j(&self.config.blok_home.display().to_string()),
            j(&self.config.model_root.display().to_string()),
            j(&self.config.report_root.display().to_string()),
            self.config.strict_direct_io,
            target_host_json(),
            fetch,
            manifest,
            generate
        )
    }
}

fn target_host_json() -> &'static str {
    concat!(
        "{\"hostname\":\"ubuntudev\",\"os\":\"Ubuntu 26.04 bare metal\",",
        "\"kernel\":\"7.0.0-22-generic\",",
        "\"board\":\"ASUSTeK ROG CROSSHAIR VII HERO\",",
        "\"cpu\":{\"vendor\":\"AuthenticAMD\",\"model\":\"AMD Ryzen 9 5950X 16-Core Processor\",",
        "\"logical_cpus\":32,\"physical_cores\":16,\"threads_per_core\":2,",
        "\"sockets\":1,\"numa_nodes\":1,\"l3_instances\":2,\"l3_kib\":65536,",
        "\"max_mhz\":5086.1812,\"boost\":true,\"microcode\":\"0xa201213\",",
        "\"cpufreq_driver\":\"amd-pstate-epp\",\"cpufreq_governor\":\"powersave\",",
        "\"idle_driver\":\"acpi_idle\",\"idle_governor\":\"menu\",",
        "\"cpu_dma_latency_usec\":2000000000,\"rapl_range_watts\":280,",
        "\"idle_pkg_watts\":23.38,\"tsc_mhz\":3394},",
        "\"gpu\":{\"name\":\"NVIDIA GeForce RTX 5060 Ti\",\"bus_id\":\"00000000:09:00.0\",",
        "\"driver\":\"595.71.05\",\"cuda\":\"13.2\",\"vram_mib\":16311,\"power_w\":180},",
        "\"sensors\":{\"cpu\":\"k10temp\",\"memory\":\"jc42\",",
        "\"super_io\":\"ITE IT8665E present; upstream hwmon driver unavailable\"}}"
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

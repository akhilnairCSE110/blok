use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HardwareReport {
    pub json: String,
}

impl HardwareReport {
    pub fn probe() -> Self {
        let mounts = mountpoints();
        let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
        let modules = read("/proc/modules").unwrap_or_default();
        let gpus = pci("0x10de", "0x03");
        let gpu = gpus.first();
        let smi = nvidia_smi();
        let nvm = nvme(&mounts);
        let sandboxed = Path::new("/sys/bus/pci/devices").exists()
            && gpu.is_some()
            && !Path::new("/dev/nvidia0").exists();
        let failures = failures(gpu, smi.as_ref(), sandboxed, &nvm);

        Self {
            json: format!(
                concat!(
                    "{{\"schema\":\"blok.hardware.v0\",\"host\":{},\"cpu\":{},",
                    "\"gpu\":{},\"nvme\":[{}],\"capabilities\":{},\"failures\":[{}]}}"
                ),
                host(sandboxed),
                cpu(&cpuinfo),
                gpu_json(gpu, smi.as_ref()),
                nvm.join(","),
                caps(&modules),
                failures.join(",")
            ),
        }
    }
}

fn host(sandboxed: bool) -> String {
    format!(
        concat!(
            "{{\"hostname\":{},\"os\":{},\"kernel\":{},\"sandboxed\":{},",
            "\"probe_utc\":{}}}"
        ),
        q(&trimmed("/proc/sys/kernel/hostname")),
        q(&os()),
        q(&trimmed("/proc/sys/kernel/osrelease")),
        sandboxed,
        q(&format!(
            "unix:{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        ))
    )
}

fn cpu(cpuinfo: &str) -> String {
    let logical = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count() as u32;
    let model = cpu_field(cpuinfo, "model name");
    let vendor = cpu_field(cpuinfo, "vendor_id");
    let physical = core_count(cpuinfo).unwrap_or(logical);
    format!(
        concat!(
            "{{\"model\":{},\"vendor\":{},\"sockets\":{},\"physical_cores\":{},",
            "\"logical_cpus\":{},\"numa_nodes\":{},\"frequency_governor\":{},",
            "\"scaling_driver\":{},\"isolated_cpus\":{},\"ccd_plan\":{}}}"
        ),
        q(&model),
        q(&vendor),
        socket_count(cpuinfo),
        physical,
        logical,
        dir_count("/sys/devices/system/node", "node").max(1),
        q(&trimmed(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        )),
        q(&trimmed(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
        )),
        opt_q(Some(&trimmed("/sys/devices/system/cpu/isolated"))),
        q(if vendor == "AuthenticAMD" && physical >= 16 {
            "ccd0_cuda_submission_ccd1_io_polling"
        } else {
            "unknown"
        })
    )
}

fn gpu_json(gpu: Option<&PathBuf>, smi: Option<&SmiGpu>) -> String {
    let status = if gpu.is_none() {
        "absent"
    } else if !Path::new("/dev/nvidiactl").exists() {
        "permission_blocked"
    } else if smi.is_none() {
        "driver_unreachable"
    } else {
        "ready"
    };
    let p = gpu.map(PathBuf::as_path);
    let bdf = p.and_then(file_name).unwrap_or("");
    let device = p.map_or(String::new(), |p| trimmed_path(&p.join("device")));
    let name = smi
        .map(|s| s.name.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| lspci_label(bdf));
    format!(
        concat!(
            "{{\"pci_bdf\":{},\"name\":{},\"vendor_id\":{},\"device_id\":{},",
            "\"subsystem_vendor_id\":{},\"subsystem_device_id\":{},\"numa_node\":{},",
            "\"local_cpulist\":{},\"kernel_driver\":{},",
            "\"nvidia_module_version\":{},\"nvidia_smi_ok\":{},",
            "\"compute_capability\":{},\"vram_bytes\":{},\"pcie_link_gen\":{},",
            "\"pcie_link_width\":{},\"pcie_max_link_gen\":{},\"pcie_max_link_width\":{},",
            "\"status\":{}}}"
        ),
        q(bdf),
        q(&name),
        q(&p.map_or(String::new(), |p| trimmed_path(&p.join("vendor")))),
        q(&device),
        q(&p.map_or(String::new(), |p| trimmed_path(&p.join("subsystem_vendor")))),
        q(&p.map_or(String::new(), |p| trimmed_path(&p.join("subsystem_device")))),
        p.map_or("null".to_owned(), |p| trimmed_path(&p.join("numa_node"))),
        opt_q(p.map(|p| trimmed_path(&p.join("local_cpulist"))).as_deref()),
        q(&p.and_then(driver).unwrap_or_default()),
        q(&module_version()),
        smi.is_some(),
        opt_q(smi.map(|s| s.compute.as_str())),
        smi.map_or("null".to_owned(), |s| (s.vram_mib * 1024 * 1024)
            .to_string()),
        opt_q(
            p.map(|p| trimmed_path(&p.join("current_link_speed")))
                .as_deref()
        ),
        opt_q(
            p.map(|p| trimmed_path(&p.join("current_link_width")))
                .as_deref()
        ),
        opt_q(
            p.map(|p| trimmed_path(&p.join("max_link_speed")))
                .as_deref()
        ),
        opt_q(
            p.map(|p| trimmed_path(&p.join("max_link_width")))
                .as_deref()
        ),
        q(status)
    )
}

fn nvme(mounts: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for block in blocks("nvme") {
        let dev = Path::new("/sys/block").join(&block);
        let pci = fs::canonicalize(dev.join("device")).ok().and_then(|p| {
            p.ancestors()
                .find(|a| a.join("vendor").exists())
                .map(Path::to_path_buf)
        });
        let mps: Vec<_> = mounts
            .iter()
            .filter(|(source, _)| source.contains(&block))
            .map(|(_, target)| q(target))
            .collect();
        let role = if mps.iter().any(|m| m == "\"/\"") {
            "root"
        } else if mps.is_empty() {
            "unknown"
        } else {
            "model"
        };
        out.push(format!(
            concat!(
                "{{\"pci_bdf\":{},\"block\":{},\"model\":{},\"controller\":{},",
                "\"vendor_id\":{},\"device_id\":{},\"numa_node\":{},",
                "\"local_cpulist\":{},\"kernel_driver\":{},",
                "\"mountpoints\":[{}],\"logical_block_bytes\":{},",
                "\"physical_block_bytes\":{},\"read_ahead_kb\":{},\"nr_requests\":{},",
                "\"scheduler\":{},\"write_cache\":{},\"pcie_link_gen\":{},\"pcie_link_width\":{},",
                "\"pcie_max_link_gen\":{},\"pcie_max_link_width\":{},",
                "\"role\":{},\"ugds_rebind_allowed\":{}}}"
            ),
            q(pci.as_ref().and_then(|p| file_name(p)).unwrap_or("")),
            q(&block),
            q(trimmed_path(&dev.join("device/model")).as_str()),
            q(&pci.as_ref().map_or(String::new(), |p| {
                lspci_label(file_name(p).unwrap_or(""))
            })),
            q(&pci
                .as_ref()
                .map_or(String::new(), |p| { trimmed_path(&p.join("vendor")) })),
            q(&pci
                .as_ref()
                .map_or(String::new(), |p| { trimmed_path(&p.join("device")) })),
            pci.as_ref()
                .map_or("null".to_owned(), |p| trimmed_path(&p.join("numa_node"))),
            opt_q(
                pci.as_ref()
                    .map(|p| trimmed_path(&p.join("local_cpulist")))
                    .as_deref()
            ),
            q(&pci.as_deref().and_then(driver).unwrap_or_default()),
            mps.join(","),
            num_or_null(&dev.join("queue/logical_block_size")),
            num_or_null(&dev.join("queue/physical_block_size")),
            num_or_null(&dev.join("queue/read_ahead_kb")),
            num_or_null(&dev.join("queue/nr_requests")),
            q(&trimmed_path(&dev.join("queue/scheduler"))),
            q(&trimmed_path(&dev.join("queue/write_cache"))),
            opt_q(
                pci.as_ref()
                    .map(|p| trimmed_path(&p.join("current_link_speed")))
                    .as_deref()
            ),
            opt_q(
                pci.as_ref()
                    .map(|p| trimmed_path(&p.join("current_link_width")))
                    .as_deref()
            ),
            opt_q(
                pci.as_ref()
                    .map(|p| trimmed_path(&p.join("max_link_speed")))
                    .as_deref()
            ),
            opt_q(
                pci.as_ref()
                    .map(|p| trimmed_path(&p.join("max_link_width")))
                    .as_deref()
            ),
            q(role),
            role != "root"
        ));
    }
    out
}

fn caps(modules: &str) -> String {
    let cuda = if Path::new("/dev/nvidiactl").exists() {
        "present"
    } else if module(modules, "nvidia") == "loaded" {
        "permission_blocked"
    } else {
        "missing"
    };
    format!(
        concat!(
            "{{\"cuda_runtime\":{},\"nvcc\":{},\"fio\":{},\"nvidia_fs\":{},",
            "\"ugds_drv\":{}}}"
        ),
        q(cuda),
        q(cmd("nvcc")),
        q(cmd("fio")),
        q(module(modules, "nvidia_fs")),
        q(module(modules, "ugds_drv"))
    )
}

fn failures(
    gpu: Option<&PathBuf>,
    smi: Option<&SmiGpu>,
    sandboxed: bool,
    nvme: &[String],
) -> Vec<String> {
    let mut v = Vec::new();
    if gpu.is_none() {
        v.push(failure(
            "gpu",
            "no NVIDIA PCI display device observed",
            "check PCI visibility",
        ));
    } else if sandboxed {
        v.push(failure(
            "gpu.device_nodes",
            "NVIDIA PCI device is visible but /dev/nvidia* is hidden",
            "run outside the sandbox or grant device access",
        ));
    } else if smi.is_none() {
        v.push(failure(
            "gpu.nvidia_smi",
            "nvidia-smi failed",
            "load driver and check permissions",
        ));
    }
    if nvme.is_empty() {
        v.push(failure(
            "nvme",
            "no NVMe block device observed",
            "check /sys/block visibility",
        ));
    }
    v
}

fn failure(field: &str, reason: &str, fix: &str) -> String {
    format!(
        "{{\"field\":{},\"reason\":{},\"fix\":{}}}",
        q(field),
        q(reason),
        q(fix)
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmiGpu {
    name: String,
    compute: String,
    vram_mib: u64,
}

fn nvidia_smi() -> Option<SmiGpu> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    parse_smi(text.lines().next()?)
}

fn parse_smi(line: &str) -> Option<SmiGpu> {
    let mut parts = line.split(',').map(str::trim);
    Some(SmiGpu {
        name: parts.next()?.to_owned(),
        compute: parts.next()?.to_owned(),
        vram_mib: parts.next()?.parse().ok()?,
    })
}

fn lspci_label(bdf: &str) -> String {
    let short = bdf.strip_prefix("0000:").unwrap_or(bdf);
    Command::new("lspci")
        .args(["-s", short])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().split_once(": ").map(|(_, label)| label.to_owned()))
        .unwrap_or_default()
}

fn pci(vendor: &str, class_prefix: &str) -> Vec<PathBuf> {
    fs::read_dir("/sys/bus/pci/devices")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| trimmed_path(&p.join("vendor")) == vendor)
        .filter(|p| trimmed_path(&p.join("class")).starts_with(class_prefix))
        .collect()
}

fn mountpoints() -> Vec<(String, String)> {
    read("/proc/self/mountinfo")
        .unwrap_or_default()
        .lines()
        .filter_map(parse_mount)
        .collect()
}

fn parse_mount(line: &str) -> Option<(String, String)> {
    let mut split = line.split(" - ");
    let left = split.next()?;
    let right = split.next()?;
    let target = left.split_whitespace().nth(4)?.replace("\\040", " ");
    let source = right.split_whitespace().nth(1)?.to_owned();
    Some((source, target))
}

fn core_count(cpuinfo: &str) -> Option<u32> {
    let mut cores = HashSet::new();
    let mut physical = None;
    let mut core = None;
    for line in cpuinfo.lines().chain([""].iter().copied()) {
        if line.is_empty() {
            if let (Some(p), Some(c)) = (physical.take(), core.take()) {
                cores.insert((p, c));
            }
        } else if let Some(v) = line.strip_prefix("physical id") {
            physical = v.split(':').nth(1).map(str::trim).map(str::to_owned);
        } else if let Some(v) = line.strip_prefix("core id") {
            core = v.split(':').nth(1).map(str::trim).map(str::to_owned);
        }
    }
    (!cores.is_empty()).then_some(cores.len() as u32)
}

fn socket_count(cpuinfo: &str) -> u32 {
    let sockets: HashSet<_> = cpuinfo
        .lines()
        .filter_map(|l| l.strip_prefix("physical id")?.split(':').nth(1))
        .map(str::trim)
        .collect();
    sockets.len().max(1) as u32
}

fn blocks(prefix: &str) -> Vec<String> {
    fs::read_dir("/sys/block")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix))
        .collect()
}

fn cpu_field(cpuinfo: &str, key: &str) -> String {
    cpuinfo
        .lines()
        .find_map(|l| l.strip_prefix(key)?.split(':').nth(1).map(str::trim))
        .unwrap_or("")
        .to_owned()
}

fn module(modules: &str, name: &str) -> &'static str {
    if modules.lines().any(|l| l.starts_with(name)) {
        "loaded"
    } else {
        "missing"
    }
}

fn module_version() -> String {
    trimmed("/proc/driver/nvidia/version")
        .split_whitespace()
        .find(|s| s.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .unwrap_or("")
        .to_owned()
}

fn cmd(name: &str) -> &'static str {
    env::var_os("PATH")
        .and_then(|p| env::split_paths(&p).find(|d| d.join(name).is_file()))
        .map_or("missing", |_| "present")
}

fn driver(path: &Path) -> Option<String> {
    fs::read_link(path.join("driver"))
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
}

fn file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|s| s.to_str())
}

fn dir_count(path: &str, prefix: &str) -> u32 {
    fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
        .count() as u32
}

fn num_or_null(path: &Path) -> String {
    trimmed_path(path)
        .parse::<u64>()
        .map_or("null".to_owned(), |n| n.to_string())
}

fn trimmed(path: &str) -> String {
    read(path).map(|s| s.trim().to_owned()).unwrap_or_default()
}

fn trimmed_path(path: &Path) -> String {
    fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

fn read(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn os() -> String {
    read("/etc/os-release")
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|v| v.trim_matches('"').to_owned())
        })
        .unwrap_or_default()
}

fn opt_q(value: Option<&str>) -> String {
    value.filter(|v| !v.is_empty()).map_or("null".to_owned(), q)
}

fn q(value: &str) -> String {
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
    fn parses_nvidia_smi_csv() {
        let smi = parse_smi("NVIDIA GeForce RTX 5060 Ti, 12.0, 16311").unwrap();
        assert_eq!(smi.compute, "12.0");
        assert_eq!(smi.vram_mib, 16311);
    }

    #[test]
    fn parses_mountinfo_source_and_target() {
        let mount = parse_mount("42 31 259:1 / / rw,relatime - ext4 /dev/nvme0n1p2 rw").unwrap();
        assert_eq!(mount, ("/dev/nvme0n1p2".to_owned(), "/".to_owned()));
    }
}

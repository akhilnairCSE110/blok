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
        let cpuinfo = read("/proc/cpuinfo");
        let modules = read("/proc/modules");
        let gpu = pci("0x10de", "0x03").into_iter().next();
        let smi = nvidia_smi();
        let nvme = nvme(&mounts());
        let sandbox = Path::new("/sys/bus/pci/devices").exists()
            && gpu.is_some()
            && !Path::new("/dev/nvidia0").exists();
        Self {
            json: format!(
                "{{\"schema\":\"blok.hardware.v0\",\"host\":{},\"cpu\":{},\"gpu\":{},\"nvme\":[{}],\"capabilities\":{},\"failures\":[{}]}}",
                host(sandbox), cpu(&cpuinfo), gpu_json(gpu.as_deref(), smi.as_ref()),
                nvme.join(","), caps(&modules), failures(gpu.as_deref(), smi.as_ref(), sandbox, &nvme).join(",")
            ),
        }
    }
}

fn host(sandbox: bool) -> String {
    format!(
        "{{\"hostname\":{},\"os\":{},\"kernel\":{},\"sandboxed\":{},\"probe_utc\":{}}}",
        q(&read_trim("/proc/sys/kernel/hostname")),
        q(&os()),
        q(&read_trim("/proc/sys/kernel/osrelease")),
        sandbox,
        q(&format!(
            "unix:{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        ))
    )
}

fn cpu(s: &str) -> String {
    let logical = s.lines().filter(|l| l.starts_with("processor")).count() as u32;
    let vendor = cpu_field(s, "vendor_id");
    let physical = core_count(s).unwrap_or(logical);
    format!(
        concat!(
            "{{\"model\":{},\"vendor\":{},\"sockets\":{},\"physical_cores\":{},",
            "\"logical_cpus\":{},\"numa_nodes\":{},\"frequency_governor\":{},",
            "\"scaling_driver\":{},\"isolated_cpus\":{},\"ccd_plan\":{}}}"
        ),
        q(&cpu_field(s, "model name")),
        q(&vendor),
        socket_count(s),
        physical,
        logical,
        dirs("/sys/devices/system/node", "node").max(1),
        q(&read_trim(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        )),
        q(&read_trim(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"
        )),
        oq(read_trim("/sys/devices/system/cpu/isolated")),
        q(if vendor == "AuthenticAMD" && physical >= 16 {
            "ccd0_cuda_submission_ccd1_io_polling"
        } else {
            "unknown"
        })
    )
}

fn gpu_json(gpu: Option<&Path>, smi: Option<&SmiGpu>) -> String {
    let bdf = gpu.and_then(name).unwrap_or("");
    let status = match (
        gpu.is_some(),
        Path::new("/dev/nvidiactl").exists(),
        smi.is_some(),
    ) {
        (false, _, _) => "absent",
        (true, false, _) => "permission_blocked",
        (true, true, false) => "driver_unreachable",
        _ => "ready",
    };
    let gpu_name = smi
        .map(|s| s.name.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| lspci(bdf));
    format!(
        concat!(
            "{{\"pci_bdf\":{},\"name\":{},\"vendor_id\":{},\"device_id\":{},",
            "\"subsystem_vendor_id\":{},\"subsystem_device_id\":{},\"numa_node\":{},",
            "\"local_cpulist\":{},\"kernel_driver\":{},\"nvidia_module_version\":{},",
            "\"nvidia_smi_ok\":{},\"compute_capability\":{},\"vram_bytes\":{},",
            "\"pcie_link_gen\":{},\"pcie_link_width\":{},\"pcie_max_link_gen\":{},",
            "\"pcie_max_link_width\":{},\"status\":{}}}"
        ),
        q(bdf),
        q(&gpu_name),
        sysq(gpu, "vendor"),
        sysq(gpu, "device"),
        sysq(gpu, "subsystem_vendor"),
        sysq(gpu, "subsystem_device"),
        sysn(gpu, "numa_node"),
        syso(gpu, "local_cpulist"),
        q(&gpu.map(driver).unwrap_or_default()),
        q(&module_version()),
        smi.is_some(),
        smi.map_or("null".to_owned(), |s| q(&s.compute)),
        smi.map_or("null".to_owned(), |s| (s.vram_mib * 1024 * 1024)
            .to_string()),
        syso(gpu, "current_link_speed"),
        syso(gpu, "current_link_width"),
        syso(gpu, "max_link_speed"),
        syso(gpu, "max_link_width"),
        q(status)
    )
}

fn nvme(mounts: &[(String, String)]) -> Vec<String> {
    blocks("nvme")
        .into_iter()
        .map(|block| {
            let dev = Path::new("/sys/block").join(&block);
            let pci = fs::canonicalize(dev.join("device")).ok().and_then(|p| {
                p.ancestors()
                    .find(|a| a.join("vendor").exists())
                    .map(Path::to_path_buf)
            });
            let mps: Vec<_> = mounts
                .iter()
                .filter(|(src, _)| src.contains(&block))
                .map(|(_, dst)| q(dst))
                .collect();
            let role = if mps.iter().any(|m| m == "\"/\"") {
                "root"
            } else if mps.is_empty() {
                "unknown"
            } else {
                "model"
            };
            format!(
                concat!(
                    "{{\"pci_bdf\":{},\"block\":{},\"model\":{},\"controller\":{},",
                    "\"vendor_id\":{},\"device_id\":{},\"numa_node\":{},\"local_cpulist\":{},",
                    "\"kernel_driver\":{},\"mountpoints\":[{}],\"logical_block_bytes\":{},",
                    "\"physical_block_bytes\":{},\"read_ahead_kb\":{},\"nr_requests\":{},",
                    "\"scheduler\":{},\"write_cache\":{},\"pcie_link_gen\":{},\"pcie_link_width\":{},",
                    "\"pcie_max_link_gen\":{},\"pcie_max_link_width\":{},\"role\":{},",
                    "\"ugds_rebind_allowed\":{}}}"
                ),
                q(pci.as_deref().and_then(name).unwrap_or("")),
                q(&block),
                q(&read_path(&dev.join("device/model"))),
                q(&pci.as_deref().and_then(name).map(lspci).unwrap_or_default()),
                sysq(pci.as_deref(), "vendor"),
                sysq(pci.as_deref(), "device"),
                sysn(pci.as_deref(), "numa_node"),
                syso(pci.as_deref(), "local_cpulist"),
                q(&pci.as_deref().map(driver).unwrap_or_default()),
                mps.join(","),
                num(&dev.join("queue/logical_block_size")),
                num(&dev.join("queue/physical_block_size")),
                num(&dev.join("queue/read_ahead_kb")),
                num(&dev.join("queue/nr_requests")),
                q(&read_path(&dev.join("queue/scheduler"))),
                q(&read_path(&dev.join("queue/write_cache"))),
                syso(pci.as_deref(), "current_link_speed"),
                syso(pci.as_deref(), "current_link_width"),
                syso(pci.as_deref(), "max_link_speed"),
                syso(pci.as_deref(), "max_link_width"),
                q(role),
                role != "root"
            )
        })
        .collect()
}

fn caps(mods: &str) -> String {
    let cuda = if Path::new("/dev/nvidiactl").exists() {
        "present"
    } else if module(mods, "nvidia") == "loaded" {
        "permission_blocked"
    } else {
        "missing"
    };
    format!(
        "{{\"cuda_runtime\":{},\"nvcc\":{},\"fio\":{},\"nvidia_fs\":{},\"ugds_drv\":{}}}",
        q(cuda),
        q(cmd("nvcc")),
        q(cmd("fio")),
        q(module(mods, "nvidia_fs")),
        q(module(mods, "ugds_drv"))
    )
}

fn failures(
    gpu: Option<&Path>,
    smi: Option<&SmiGpu>,
    sandbox: bool,
    nvme: &[String],
) -> Vec<String> {
    let mut v = Vec::new();
    if gpu.is_none() {
        v.push(fail(
            "gpu",
            "no NVIDIA PCI display device observed",
            "check PCI visibility",
        ));
    } else if sandbox {
        v.push(fail(
            "gpu.device_nodes",
            "NVIDIA PCI device is visible but /dev/nvidia* is hidden",
            "run outside the sandbox or grant device access",
        ));
    } else if smi.is_none() {
        v.push(fail(
            "gpu.nvidia_smi",
            "nvidia-smi failed",
            "load driver and check permissions",
        ));
    }
    if nvme.is_empty() {
        v.push(fail(
            "nvme",
            "no NVMe block device observed",
            "check /sys/block visibility",
        ));
    }
    v
}

fn fail(field: &str, reason: &str, fix: &str) -> String {
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
    let o = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    o.status
        .success()
        .then(|| String::from_utf8(o.stdout).ok())
        .flatten()
        .and_then(|s| parse_smi(s.lines().next()?))
}

fn parse_smi(line: &str) -> Option<SmiGpu> {
    let mut p = line.split(',').map(str::trim);
    Some(SmiGpu {
        name: p.next()?.to_owned(),
        compute: p.next()?.to_owned(),
        vram_mib: p.next()?.parse().ok()?,
    })
}

fn lspci(bdf: &str) -> String {
    let short = bdf.strip_prefix("0000:").unwrap_or(bdf);
    Command::new("lspci")
        .args(["-s", short])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().split_once(": ").map(|(_, v)| v.to_owned()))
        .unwrap_or_default()
}

fn pci(vendor: &str, class: &str) -> Vec<PathBuf> {
    fs::read_dir("/sys/bus/pci/devices")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            read_path(&p.join("vendor")) == vendor && read_path(&p.join("class")).starts_with(class)
        })
        .collect()
}

fn mounts() -> Vec<(String, String)> {
    read("/proc/self/mountinfo")
        .lines()
        .filter_map(parse_mount)
        .collect()
}

fn parse_mount(line: &str) -> Option<(String, String)> {
    let (left, right) = line.split_once(" - ")?;
    Some((
        right.split_whitespace().nth(1)?.to_owned(),
        left.split_whitespace().nth(4)?.replace("\\040", " "),
    ))
}

fn core_count(cpuinfo: &str) -> Option<u32> {
    let mut cores = HashSet::new();
    let (mut socket, mut core) = (None, None);
    for line in cpuinfo.lines().chain([""]) {
        if line.is_empty() {
            if let (Some(s), Some(c)) = (socket.take(), core.take()) {
                cores.insert((s, c));
            }
        } else if let Some(v) = field(line, "physical id") {
            socket = Some(v.to_owned());
        } else if let Some(v) = field(line, "core id") {
            core = Some(v.to_owned());
        }
    }
    (!cores.is_empty()).then_some(cores.len() as u32)
}

fn socket_count(cpuinfo: &str) -> u32 {
    cpuinfo
        .lines()
        .filter_map(|l| field(l, "physical id"))
        .collect::<HashSet<_>>()
        .len()
        .max(1) as u32
}

fn cpu_field(cpuinfo: &str, key: &str) -> String {
    cpuinfo
        .lines()
        .find_map(|l| field(l, key))
        .unwrap_or("")
        .to_owned()
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.strip_prefix(key)?.split(':').nth(1).map(str::trim)
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

fn module(mods: &str, name: &str) -> &'static str {
    if mods.lines().any(|l| l.starts_with(name)) {
        "loaded"
    } else {
        "missing"
    }
}

fn module_version() -> String {
    read_trim("/proc/driver/nvidia/version")
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

fn dirs(path: &str, prefix: &str) -> u32 {
    fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
        .count() as u32
}

fn driver(p: &Path) -> String {
    fs::read_link(p.join("driver"))
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_default()
}

fn name(p: &Path) -> Option<&str> {
    p.file_name().and_then(|s| s.to_str())
}

fn sysq(p: Option<&Path>, f: &str) -> String {
    q(&p.map(|p| read_path(&p.join(f))).unwrap_or_default())
}

fn syso(p: Option<&Path>, f: &str) -> String {
    oq(p.map(|p| read_path(&p.join(f))).unwrap_or_default())
}

fn sysn(p: Option<&Path>, f: &str) -> String {
    p.map(|p| read_path(&p.join(f)))
        .filter(|s| !s.is_empty())
        .unwrap_or("null".to_owned())
}

fn num(p: &Path) -> String {
    read_path(p)
        .parse::<u64>()
        .map_or("null".to_owned(), |n| n.to_string())
}

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

fn read_trim(path: &str) -> String {
    read(path).trim().to_owned()
}

fn read_path(path: &Path) -> String {
    fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

fn os() -> String {
    read("/etc/os-release")
        .lines()
        .find_map(|l| {
            l.strip_prefix("PRETTY_NAME=")
                .map(|v| v.trim_matches('"').to_owned())
        })
        .unwrap_or_default()
}

fn oq(s: String) -> String {
    if s.is_empty() {
        "null".to_owned()
    } else {
        q(&s)
    }
}

fn q(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
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

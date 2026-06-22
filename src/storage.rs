use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{Error, Result, RuntimeConfig};

pub const KIMI_REPO: &str = "moonshotai/Kimi-K2.6";
pub const KIMI_REVISION: &str = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb";
pub const KIMI_FILES: u32 = 96;
pub const KIMI_SAFETENSORS: u32 = 64;
pub const KIMI_BYTES: u64 = 595_421_860_056;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchPlan {
    pub repo: &'static str,
    pub revision: &'static str,
    pub root_dir: PathBuf,
    pub local_dir: PathBuf,
    pub blok_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub expected_bytes: u64,
    pub files: u32,
    pub safetensors: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchStatus {
    pub downloaded_bytes: u64,
    pub safetensors: u32,
    pub complete: bool,
    pub pid: Option<u32>,
}

impl FetchPlan {
    pub fn kimi(config: &RuntimeConfig) -> Self {
        let root_dir = config.model_root.join("moonshotai").join("Kimi-K2.6");
        let local_dir = root_dir.join("source").join("hf").join(KIMI_REVISION);
        let blok_dir = root_dir.join("blok");
        let cache_dir = config.blok_home.join("hf-cache");
        Self {
            repo: KIMI_REPO,
            revision: KIMI_REVISION,
            root_dir,
            local_dir,
            blok_dir,
            cache_dir,
            expected_bytes: KIMI_BYTES,
            files: KIMI_FILES,
            safetensors: KIMI_SAFETENSORS,
        }
    }

    pub fn receipt(&self) -> String {
        format!(
            concat!(
                "repo={}\nrevision={}\nroot_dir={}\nlocal_dir={}\nblok_dir={}\ncache_dir={}\n",
                "expected_bytes={}\nfiles={}\nsafetensors={}\n"
            ),
            self.repo,
            self.revision,
            self.root_dir.display(),
            self.local_dir.display(),
            self.blok_dir.display(),
            self.cache_dir.display(),
            self.expected_bytes,
            self.files,
            self.safetensors
        )
    }
}

pub fn fetch_kimi(config: &RuntimeConfig, hf_bin: &Path) -> Result<()> {
    let plan = FetchPlan::kimi(config);
    prepare(&plan)?;
    let status = hf_command(hf_bin, &plan).status()?;

    let downloaded = local_bytes(&plan.local_dir)?;
    let safetensors = file_count(&plan.local_dir, ".safetensors")?;
    fs::write(
        plan.blok_dir.join("fetch-status.txt"),
        format!(
            "downloaded_bytes={downloaded}\nsafetensors={safetensors}\ncomplete={}\n",
            status.success()
                && downloaded >= plan.expected_bytes
                && safetensors == plan.safetensors
        ),
    )?;
    if status.success() && downloaded >= plan.expected_bytes && safetensors == plan.safetensors {
        Ok(())
    } else {
        Err(Error::Storage(format!(
            "kimi-k2.6 download incomplete: {downloaded}/{} bytes, {safetensors}/{} safetensors",
            plan.expected_bytes, plan.safetensors
        )))
    }
}

pub fn fetch_kimi_detached(config: &RuntimeConfig, hf_bin: &Path) -> Result<u32> {
    let plan = FetchPlan::kimi(config);
    prepare(&plan)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(plan.blok_dir.join("fetch.log"))?;
    let err = log.try_clone()?;
    let child = hf_command(hf_bin, &plan)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err))
        .spawn()?;
    let pid = child.id();
    fs::write(plan.blok_dir.join("fetch.pid"), format!("{pid}\n"))?;
    Ok(pid)
}

pub fn kimi_status(config: &RuntimeConfig) -> Result<FetchStatus> {
    let plan = FetchPlan::kimi(config);
    let downloaded_bytes = local_bytes(&plan.local_dir).unwrap_or(0);
    let safetensors = file_count(&plan.local_dir, ".safetensors").unwrap_or(0);
    let pid = fs::read_to_string(plan.blok_dir.join("fetch.pid"))
        .ok()
        .and_then(|s| s.trim().parse().ok());
    Ok(FetchStatus {
        downloaded_bytes,
        safetensors,
        complete: downloaded_bytes >= plan.expected_bytes && safetensors == plan.safetensors,
        pid,
    })
}

fn prepare(plan: &FetchPlan) -> Result<()> {
    fs::create_dir_all(&plan.local_dir)?;
    fs::create_dir_all(&plan.blok_dir)?;
    fs::create_dir_all(&plan.cache_dir)?;
    fs::write(plan.blok_dir.join("fetch-plan.txt"), plan.receipt())?;
    Ok(())
}

fn hf_command(hf_bin: &Path, plan: &FetchPlan) -> Command {
    let mut command = Command::new("setsid");
    command
        .arg(hf_bin)
        .args([
            "download",
            plan.repo,
            "--revision",
            plan.revision,
            "--local-dir",
        ])
        .arg(&plan.local_dir)
        .env("HF_HOME", &plan.cache_dir)
        .env("HF_HUB_CACHE", plan.cache_dir.join("hub"))
        .env("HF_XET_CACHE", plan.cache_dir.join("xet"))
        .env("HF_XET_HIGH_PERFORMANCE", "1")
        .env("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
        .env("HF_HUB_DOWNLOAD_TIMEOUT", "60");
    command
}

pub fn local_bytes(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total = total
                .checked_add(meta.len())
                .ok_or_else(|| Error::Storage("local byte count overflowed".to_owned()))?;
        } else if meta.is_dir() && entry.file_name() != ".cache" {
            total = total
                .checked_add(local_bytes(&entry.path())?)
                .ok_or_else(|| Error::Storage("local byte count overflowed".to_owned()))?;
        }
    }
    Ok(total)
}

pub fn file_count(path: &Path, suffix: &str) -> Result<u32> {
    let mut total = 0_u32;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            if entry.file_name().to_string_lossy().ends_with(suffix) {
                total += 1;
            }
        } else if meta.is_dir() && entry.file_name() != ".cache" {
            total += file_count(&entry.path(), suffix)?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_plan_targets_blok_model_root() {
        let config = RuntimeConfig {
            blok_home: PathBuf::from("/nvme/.blok"),
            model_root: PathBuf::from("/nvme/.blok/models"),
            report_root: PathBuf::from("/nvme/.blok/reports"),
            strict_direct_io: true,
        };
        let plan = FetchPlan::kimi(&config);

        assert_eq!(
            plan.local_dir,
            PathBuf::from(
                "/nvme/.blok/models/moonshotai/Kimi-K2.6/source/hf/7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
            )
        );
        assert_eq!(
            plan.blok_dir,
            PathBuf::from("/nvme/.blok/models/moonshotai/Kimi-K2.6/blok")
        );
        assert_eq!(plan.safetensors, 64);
        assert!(plan.expected_bytes > 500 * 1024_u64.pow(3));
    }

    #[test]
    fn local_bytes_counts_nested_files() {
        let root = std::env::temp_dir().join(format!("blok-bytes-{}", std::process::id()));
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("a"), [1, 2, 3]).unwrap();
        fs::write(nested.join("b"), [4, 5]).unwrap();

        assert_eq!(local_bytes(&root).unwrap(), 5);
        assert_eq!(file_count(&root, "b").unwrap(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}

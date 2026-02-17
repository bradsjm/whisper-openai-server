//! Model path resolution and optional Hugging Face download support.
//!
//! This module guarantees that `cfg.whisper_model` points to a readable local
//! file before backend initialization.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::StatusCode;

use crate::config::AppConfig;
use crate::error::AppError;

const LOCK_TIMEOUT: Duration = Duration::from_secs(120);
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Ensures a local Whisper model file exists, downloading from Hugging Face if needed.
pub fn ensure_model_ready(cfg: &mut AppConfig) -> Result<(), AppError> {
    if model_file_exists(&cfg.whisper_model) {
        return Ok(());
    }

    if !cfg.whisper_auto_download {
        return Err(AppError::internal(format!(
            "model file not found at {:?}; set WHISPER_MODEL to an existing file or enable WHISPER_AUTO_DOWNLOAD",
            cfg.whisper_model
        )));
    }

    let target_path = model_target_path(cfg);
    if model_file_exists(&target_path.to_string_lossy()) {
        cfg.whisper_model = target_path.to_string_lossy().to_string();
        return Ok(());
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::internal(format!(
                "failed to create model cache directory {:?}: {err}",
                parent
            ))
        })?;
    }

    let lock_path = lock_path_for(&target_path);
    let _guard = acquire_lock(&lock_path)?;

    if model_file_exists(&target_path.to_string_lossy()) {
        cfg.whisper_model = target_path.to_string_lossy().to_string();
        return Ok(());
    }

    download_model_to_path(cfg, &target_path)?;
    cfg.whisper_model = target_path.to_string_lossy().to_string();
    Ok(())
}

fn model_file_exists(path: &str) -> bool {
    fs::metadata(path)
        .map(|meta| meta.is_file() && meta.len() > 0)
        .unwrap_or(false)
}

fn model_target_path(cfg: &AppConfig) -> PathBuf {
    if cfg.whisper_model_explicit {
        return PathBuf::from(&cfg.whisper_model);
    }
    Path::new(&cfg.whisper_cache_dir).join(&cfg.whisper_hf_filename)
}

fn lock_path_for(target_path: &Path) -> PathBuf {
    let lock_name = format!(
        "{}.lock",
        target_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    );
    target_path.with_file_name(lock_name)
}

fn acquire_lock(path: &Path) -> Result<LockGuard, AppError> {
    let start = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                return Ok(LockGuard {
                    path: path.to_path_buf(),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if start.elapsed() >= LOCK_TIMEOUT {
                    return Err(AppError::internal(format!(
                        "timed out waiting for model download lock at {:?}",
                        path
                    )));
                }
                thread::sleep(LOCK_POLL_INTERVAL);
            }
            Err(err) => {
                return Err(AppError::internal(format!(
                    "failed to acquire model download lock at {:?}: {err}",
                    path
                )));
            }
        }
    }
}

fn download_model_to_path(cfg: &AppConfig, target_path: &Path) -> Result<(), AppError> {
    let url = hf_resolve_url(&cfg.whisper_hf_repo, &cfg.whisper_hf_filename);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|err| AppError::internal(format!("failed to create HTTP client: {err}")))?;

    let mut request = client.get(&url);
    if let Some(token) = cfg.hf_token.as_deref() {
        request = request.bearer_auth(token);
    }

    let mut response = request.send().map_err(|err| {
        AppError::internal(format!(
            "failed to download model from {url}: {err}; check network connectivity"
        ))
    })?;

    if !response.status().is_success() {
        return match response.status() {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(AppError::internal(format!(
                "Hugging Face rejected model download from {url} with {}; set HF_TOKEN for authenticated access",
                response.status()
            ))),
            StatusCode::NOT_FOUND => Err(AppError::internal(format!(
                "model not found at {url}; verify WHISPER_HF_REPO and WHISPER_HF_FILENAME"
            ))),
            status => Err(AppError::internal(format!(
                "model download failed from {url} with HTTP status {status}"
            ))),
        };
    }

    let tmp_path = target_path.with_extension("part");
    let mut out = File::create(&tmp_path).map_err(|err| {
        AppError::internal(format!(
            "failed to create temporary model file {:?}: {err}",
            tmp_path
        ))
    })?;
    std::io::copy(&mut response, &mut out).map_err(|err| {
        AppError::internal(format!(
            "failed writing downloaded model to {:?}: {err}",
            tmp_path
        ))
    })?;
    out.flush().map_err(|err| {
        AppError::internal(format!(
            "failed to flush downloaded model file {:?}: {err}",
            tmp_path
        ))
    })?;

    let size = out.metadata().map(|m| m.len()).unwrap_or_default();
    if size == 0 {
        let _ = fs::remove_file(&tmp_path);
        return Err(AppError::internal(format!(
            "downloaded empty model file from {url}; refusing to continue"
        )));
    }

    fs::rename(&tmp_path, target_path).map_err(|err| {
        AppError::internal(format!(
            "failed to move model from {:?} to {:?}: {err}",
            tmp_path, target_path
        ))
    })?;

    Ok(())
}

fn hf_resolve_url(repo: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo.trim_matches('/'),
        filename.trim_matches('/')
    )
}

struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::{hf_resolve_url, lock_path_for};
    use std::path::Path;

    #[test]
    fn resolve_url_normalizes_edges() {
        assert_eq!(
            hf_resolve_url("/ggerganov/whisper.cpp/", "/ggml-small.bin/"),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
        );
    }

    #[test]
    fn lock_path_uses_sibling_file() {
        let path = Path::new("/tmp/ggml-small.bin");
        assert_eq!(
            lock_path_for(path).to_string_lossy(),
            "/tmp/ggml-small.bin.lock"
        );
    }
}

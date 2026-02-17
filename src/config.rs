//! Configuration loading from environment variables.
//!
//! Values are intentionally validated early so startup fails fast with
//! actionable errors.

use crate::error::AppError;
use std::env;

pub const DEFAULT_WHISPER_PARALLELISM: usize = 1;
pub const MAX_WHISPER_PARALLELISM: usize = 8;

/// Supported inference backend implementations.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BackendKind {
    /// Uses `whisper-rs` (`whisper.cpp`) for local inference.
    WhisperRs,
}

/// Runtime configuration for the HTTP server and inference backend.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Host interface to bind, for example `127.0.0.1`.
    pub host: String,
    /// TCP port to bind.
    pub port: u16,
    /// Optional bearer token required by all endpoints.
    pub api_key: Option<String>,
    /// Path to a Whisper model file on disk.
    pub whisper_model: String,
    /// Whether `whisper_model` came from explicit `WHISPER_MODEL`.
    pub whisper_model_explicit: bool,
    /// Enables startup download when the model file is missing.
    pub whisper_auto_download: bool,
    /// Hugging Face repository used for model download.
    pub whisper_hf_repo: String,
    /// Whisper model filename in the Hugging Face repository.
    pub whisper_hf_filename: String,
    /// Local cache directory for downloaded models.
    pub whisper_cache_dir: String,
    /// Optional Hugging Face token for authenticated model downloads.
    pub hf_token: Option<String>,
    /// Additional accepted model identifier exposed by the API.
    pub api_model_alias: String,
    /// Selected backend implementation.
    pub backend_kind: BackendKind,
    /// Number of parallel whisper-rs inference workers.
    pub whisper_parallelism: usize,
}

impl AppConfig {
    /// Builds configuration from environment variables.
    ///
    /// Variables:
    /// - `HOST` (default `127.0.0.1`)
    /// - `PORT` (default `8000`)
    /// - `WHISPER_MODEL` (optional explicit local model path)
    /// - `WHISPER_AUTO_DOWNLOAD` (default `true`)
    /// - `WHISPER_HF_REPO` (default `ggerganov/whisper.cpp`)
    /// - `WHISPER_HF_FILENAME` (default `ggml-small.bin`)
    /// - `WHISPER_CACHE_DIR` (default `$HOME/.cache/whispercpp/models`)
    /// - `HF_TOKEN` (optional Hugging Face token)
    /// - `WHISPER_MODEL_ALIAS` (default `whisper-mlx`)
    /// - `WHISPER_BACKEND` (only `whisper-rs` is currently supported)
    /// - `WHISPER_PARALLELISM` (default `1`, min `1`, max `8`)
    /// - `API_KEY` (optional)
    pub fn from_env() -> Result<Self, AppError> {
        let host = env_str("HOST", "127.0.0.1");
        let port = env_u16("PORT", 8000)?;
        let whisper_auto_download = env_bool("WHISPER_AUTO_DOWNLOAD", true)?;
        let whisper_hf_repo = env_str("WHISPER_HF_REPO", "ggerganov/whisper.cpp");
        let whisper_hf_filename = env_str("WHISPER_HF_FILENAME", "ggml-small.bin");
        let whisper_cache_dir = env_str("WHISPER_CACHE_DIR", &default_whisper_cache_dir());
        let whisper_model_explicit = env_opt("WHISPER_MODEL").is_some();
        let whisper_model = env_opt("WHISPER_MODEL")
            .unwrap_or_else(|| format!("{}/{}", whisper_cache_dir, whisper_hf_filename));
        let api_model_alias = env_str("WHISPER_MODEL_ALIAS", "whisper-mlx");

        let backend_kind = match env_str("WHISPER_BACKEND", "whisper-rs").as_str() {
            "whisper-rs" => BackendKind::WhisperRs,
            other => {
                return Err(AppError::internal(format!(
                    "invalid WHISPER_BACKEND={other:?}; expected whisper-rs"
                )));
            }
        };
        let whisper_parallelism = env_usize_bounded(
            "WHISPER_PARALLELISM",
            DEFAULT_WHISPER_PARALLELISM,
            1,
            MAX_WHISPER_PARALLELISM,
        )?;

        Ok(Self {
            host,
            port,
            api_key: env_opt("API_KEY"),
            whisper_model,
            whisper_model_explicit,
            whisper_auto_download,
            whisper_hf_repo,
            whisper_hf_filename,
            whisper_cache_dir,
            hf_token: env_opt("HF_TOKEN"),
            api_model_alias,
            backend_kind,
            whisper_parallelism,
        })
    }

    /// Returns all accepted model identifiers for request validation.
    ///
    /// This always includes `whisper-1` for OpenAI compatibility and may include
    /// `api_model_alias` when it is different.
    pub fn accepted_model_ids(&self) -> Vec<String> {
        let mut ids = vec!["whisper-1".to_string()];
        if self.api_model_alias != "whisper-1" {
            ids.push(self.api_model_alias.clone());
        }
        ids
    }
}

fn default_whisper_cache_dir() -> String {
    format!(
        "{}/.cache/whispercpp/models",
        std::env::var("HOME").unwrap_or_else(|_| "/Users/user".to_string())
    )
}

fn env_str(name: &str, default: &str) -> String {
    match env::var(name) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                default.to_string()
            } else {
                trimmed.to_string()
            }
        }
        Err(_) => default.to_string(),
    }
}

fn env_opt(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Err(_) => None,
    }
}

fn env_u16(name: &str, default: u16) -> Result<u16, AppError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    let parsed = raw.trim().parse::<u16>().map_err(|_| {
        AppError::internal(format!("invalid {name}={raw:?}; expected integer 1-65535"))
    })?;
    if parsed == 0 {
        return Err(AppError::internal(format!(
            "invalid {name}={raw:?}; expected > 0"
        )));
    }
    Ok(parsed)
}

fn env_bool(name: &str, default: bool) -> Result<bool, AppError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AppError::internal(format!(
            "invalid {name}={raw:?}; expected true/false"
        ))),
    }
}

fn env_usize_bounded(
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, AppError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    parse_usize_bounded(name, &raw, min, max)
}

fn parse_usize_bounded(name: &str, raw: &str, min: usize, max: usize) -> Result<usize, AppError> {
    let trimmed = raw.trim();
    let parsed = trimmed.parse::<usize>().map_err(|_| {
        AppError::internal(format!(
            "invalid {name}={raw:?}; expected integer in range [{min}, {max}]"
        ))
    })?;
    if parsed < min || parsed > max {
        return Err(AppError::internal(format!(
            "invalid {name}={raw:?}; expected integer in range [{min}, {max}]"
        )));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::parse_usize_bounded;

    #[test]
    fn parse_usize_bounded_accepts_in_range_values() {
        assert_eq!(
            parse_usize_bounded("WHISPER_PARALLELISM", "1", 1, 8).unwrap(),
            1
        );
        assert_eq!(
            parse_usize_bounded("WHISPER_PARALLELISM", "8", 1, 8).unwrap(),
            8
        );
    }

    #[test]
    fn parse_usize_bounded_rejects_non_numeric_value() {
        assert!(parse_usize_bounded("WHISPER_PARALLELISM", "abc", 1, 8).is_err());
    }

    #[test]
    fn parse_usize_bounded_rejects_out_of_range_values() {
        assert!(parse_usize_bounded("WHISPER_PARALLELISM", "0", 1, 8).is_err());
        assert!(parse_usize_bounded("WHISPER_PARALLELISM", "9", 1, 8).is_err());
    }
}

//! Configuration loading from environment variables.
//!
//! Values are intentionally validated early so startup fails fast with
//! actionable errors.

use crate::error::AppError;
use std::env;

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
    /// Additional accepted model identifier exposed by the API.
    pub api_model_alias: String,
    /// Selected backend implementation.
    pub backend_kind: BackendKind,
}

impl AppConfig {
    /// Builds configuration from environment variables.
    ///
    /// Variables:
    /// - `HOST` (default `127.0.0.1`)
    /// - `PORT` (default `8000`)
    /// - `WHISPER_MODEL` (default `$HOME/.cache/whispercpp/models/ggml-small.bin`)
    /// - `WHISPER_MODEL_ALIAS` (default `whisper-mlx`)
    /// - `WHISPER_BACKEND` (only `whisper-rs` is currently supported)
    /// - `API_KEY` (optional)
    pub fn from_env() -> Result<Self, AppError> {
        let host = env_str("HOST", "127.0.0.1");
        let port = env_u16("PORT", 8000)?;
        let whisper_model = env_str(
            "WHISPER_MODEL",
            &format!(
                "{}/.cache/whispercpp/models/ggml-small.bin",
                std::env::var("HOME").unwrap_or_else(|_| "/Users/user".to_string())
            ),
        );
        let api_model_alias = env_str("WHISPER_MODEL_ALIAS", "whisper-mlx");

        let backend_kind = match env_str("WHISPER_BACKEND", "whisper-rs").as_str() {
            "whisper-rs" => BackendKind::WhisperRs,
            other => {
                return Err(AppError::internal(format!(
                    "invalid WHISPER_BACKEND={other:?}; expected whisper-rs"
                )));
            }
        };

        Ok(Self {
            host,
            port,
            api_key: env_opt("API_KEY"),
            whisper_model,
            api_model_alias,
            backend_kind,
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

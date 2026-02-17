use crate::error::AppError;
use std::env;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BackendKind {
    WhisperRs,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub whisper_model: String,
    pub api_model_alias: String,
    pub backend_kind: BackendKind,
}

impl AppConfig {
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

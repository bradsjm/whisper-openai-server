//! Configuration loading from environment variables and CLI arguments.
//!
//! Values are intentionally validated early so startup fails fast with
//! actionable errors.

use crate::error::AppError;
use clap::{Parser, ValueEnum};

pub const MAX_WHISPER_PARALLELISM: usize = 8;

/// Supported acceleration modes for whisper-rs context initialization.
#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum AccelerationKind {
    /// Prefer Metal/GPU acceleration.
    Metal,
    /// Disable GPU acceleration and run on CPU.
    None,
}

impl AccelerationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::None => "none",
        }
    }
}

/// Supported whisper.cpp model sizes.
#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum WhisperModelSize {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    #[value(name = "large-v1")]
    LargeV1,
    #[value(name = "large-v2")]
    LargeV2,
    #[value(name = "large-v3", alias = "large")]
    LargeV3,
    #[value(name = "large-v3-turbo", alias = "turbo")]
    Turbo,
}

impl Default for WhisperModelSize {
    fn default() -> Self {
        Self::Small
    }
}

/// Supported inference backend implementations.
#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum BackendKind {
    /// Uses `whisper-rs` (`whisper.cpp`) for local inference.
    #[value(name = "whisper-rs")]
    WhisperRs,
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::WhisperRs
    }
}

/// Command-line arguments for whisper-openai-server.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "whisper-openai-server",
    about = "OpenAI-compatible Whisper transcription/translation API server",
    version
)]
pub struct CliArgs {
    /// Host address to bind to
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, env = "PORT", default_value = "8000")]
    pub port: u16,

    /// API key for authentication (optional)
    #[arg(long, env = "API_KEY")]
    pub api_key: Option<String>,

    /// Local model path
    #[arg(long, env = "WHISPER_MODEL")]
    pub model: Option<String>,

    /// Model size
    #[arg(long, env = "WHISPER_MODEL_SIZE", value_enum, default_value = "small")]
    pub model_size: WhisperModelSize,

    /// Download missing model
    #[arg(long, env = "WHISPER_AUTO_DOWNLOAD", default_value = "true")]
    pub auto_download: bool,

    /// Hugging Face repository for model download
    #[arg(long, env = "WHISPER_HF_REPO", default_value = "ggerganov/whisper.cpp")]
    pub hf_repo: String,

    /// Hugging Face model filename
    #[arg(long, env = "WHISPER_HF_FILENAME")]
    pub hf_filename: Option<String>,

    /// Local cache directory for downloaded models
    #[arg(long, env = "WHISPER_CACHE_DIR")]
    pub cache_dir: Option<String>,

    /// Hugging Face auth token
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Extra accepted model id for API requests
    #[arg(long, env = "WHISPER_MODEL_ALIAS", default_value = "whisper-1")]
    pub model_alias: String,

    /// Inference backend
    #[arg(
        long,
        env = "WHISPER_BACKEND",
        value_enum,
        default_value = "whisper-rs"
    )]
    pub backend: BackendKind,

    /// Acceleration mode (metal or none)
    #[arg(
        long,
        env = "WHISPER_ACCELERATION",
        value_enum,
        default_value = "metal"
    )]
    pub acceleration: AccelerationKind,

    /// Number of inference workers (1-8)
    #[arg(long, env = "WHISPER_PARALLELISM", default_value = "1", value_parser = parse_parallelism)]
    pub parallelism: usize,
}

fn parse_parallelism(s: &str) -> Result<usize, String> {
    let value: usize = s
        .parse()
        .map_err(|_| format!("expected integer in range [1, {MAX_WHISPER_PARALLELISM}]"))?;
    if value < 1 || value > MAX_WHISPER_PARALLELISM {
        return Err(format!(
            "expected integer in range [1, {MAX_WHISPER_PARALLELISM}]"
        ));
    }
    Ok(value)
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
    /// Requested acceleration mode used when initializing whisper contexts.
    pub acceleration_kind: AccelerationKind,
    /// Whether acceleration mode was explicitly provided via env/CLI.
    pub acceleration_explicit: bool,
    /// Number of parallel whisper-rs inference workers.
    pub whisper_parallelism: usize,
    /// Requested model size used to resolve default model filename.
    pub whisper_model_size: WhisperModelSize,
}

impl AppConfig {
    /// Builds configuration from CLI arguments (which also read environment variables).
    pub fn from_args() -> Result<Self, AppError> {
        let args = CliArgs::parse();
        Self::from_cli_args(args)
    }

    /// Builds configuration from parsed CLI arguments.
    pub fn from_cli_args(args: CliArgs) -> Result<Self, AppError> {
        let cache_dir = args
            .cache_dir
            .unwrap_or_else(|| default_whisper_cache_dir());
        let model_explicit = args.model.is_some();
        let model_size = args.model_size;
        let hf_filename = args
            .hf_filename
            .unwrap_or_else(|| whisper_model_filename(model_size).to_string());
        let model = args
            .model
            .unwrap_or_else(|| format!("{}/ {}", cache_dir, hf_filename));

        Ok(Self {
            host: args.host,
            port: args.port,
            api_key: args.api_key,
            whisper_model: model,
            whisper_model_explicit: model_explicit,
            whisper_auto_download: args.auto_download,
            whisper_hf_repo: args.hf_repo,
            whisper_hf_filename: hf_filename,
            whisper_cache_dir: cache_dir,
            hf_token: args.hf_token,
            api_model_alias: args.model_alias,
            backend_kind: args.backend,
            acceleration_kind: args.acceleration,
            acceleration_explicit: true,
            whisper_parallelism: args.parallelism,
            whisper_model_size: model_size,
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

fn whisper_model_filename(size: WhisperModelSize) -> &'static str {
    match size {
        WhisperModelSize::Tiny => "ggml-tiny.bin",
        WhisperModelSize::TinyEn => "ggml-tiny.en.bin",
        WhisperModelSize::Base => "ggml-base.bin",
        WhisperModelSize::BaseEn => "ggml-base.en.bin",
        WhisperModelSize::Small => "ggml-small.bin",
        WhisperModelSize::SmallEn => "ggml-small.en.bin",
        WhisperModelSize::Medium => "ggml-medium.bin",
        WhisperModelSize::MediumEn => "ggml-medium.en.bin",
        WhisperModelSize::LargeV1 => "ggml-large-v1.bin",
        WhisperModelSize::LargeV2 => "ggml-large-v2.bin",
        WhisperModelSize::LargeV3 => "ggml-large-v3.bin",
        WhisperModelSize::Turbo => "ggml-large-v3-turbo.bin",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_parallelism, whisper_model_filename, CliArgs, WhisperModelSize,
        MAX_WHISPER_PARALLELISM,
    };
    use clap::Parser;

    #[test]
    fn parse_parallelism_accepts_in_range_values() {
        assert_eq!(parse_parallelism("1").unwrap(), 1);
        assert_eq!(parse_parallelism("8").unwrap(), 8);
    }

    #[test]
    fn parse_parallelism_rejects_non_numeric_value() {
        assert!(parse_parallelism("abc").is_err());
    }

    #[test]
    fn parse_parallelism_rejects_out_of_range_values() {
        assert!(parse_parallelism("0").is_err());
        assert!(parse_parallelism("9").is_err());
    }

    #[test]
    fn cli_parsing_supports_model_size() {
        let args = CliArgs::parse_from(["whisper-openai-server", "--model-size=medium"]);
        assert_eq!(args.model_size, WhisperModelSize::Medium);
    }

    #[test]
    fn cli_parsing_supports_acceleration() {
        let args = CliArgs::parse_from(["whisper-openai-server", "--acceleration=none"]);
        assert_eq!(args.acceleration, super::AccelerationKind::None);
    }

    #[test]
    fn whisper_model_filename_uses_expected_small_name() {
        assert_eq!(
            whisper_model_filename(WhisperModelSize::Small),
            "ggml-small.bin"
        );
    }

    #[test]
    fn whisper_model_filename_uses_expected_en_name() {
        assert_eq!(
            whisper_model_filename(WhisperModelSize::SmallEn),
            "ggml-small.en.bin"
        );
    }
}

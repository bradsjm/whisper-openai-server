//! Configuration loading from environment variables.
//!
//! Values are intentionally validated early so startup fails fast with
//! actionable errors.

use crate::error::AppError;
use std::env;

pub const DEFAULT_WHISPER_PARALLELISM: usize = 1;
pub const MAX_WHISPER_PARALLELISM: usize = 8;

/// Supported whisper.cpp model sizes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WhisperModelSize {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV1,
    LargeV2,
    LargeV3,
    Turbo,
}

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
    /// Requested model size used to resolve default model filename.
    pub whisper_model_size: WhisperModelSize,
}

/// Command-line overrides for runtime configuration.
#[derive(Debug, Clone, Default)]
pub struct CliOptions {
    pub help_requested: bool,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub api_key: Option<String>,
    pub whisper_model: Option<String>,
    pub whisper_model_size: Option<WhisperModelSize>,
    pub whisper_auto_download: Option<bool>,
    pub whisper_hf_repo: Option<String>,
    pub whisper_hf_filename: Option<String>,
    pub whisper_cache_dir: Option<String>,
    pub hf_token: Option<String>,
    pub api_model_alias: Option<String>,
    pub backend_kind: Option<BackendKind>,
    pub whisper_parallelism: Option<usize>,
}

impl AppConfig {
    /// Builds configuration from environment variables.
    ///
    /// Variables:
    /// - `HOST` (default `127.0.0.1`)
    /// - `PORT` (default `8000`)
    /// - `WHISPER_MODEL` (optional explicit local model path)
    /// - `WHISPER_MODEL_SIZE` (default `small`)
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
        let whisper_model_size = env_model_size("WHISPER_MODEL_SIZE", WhisperModelSize::Small)?;
        let whisper_auto_download = env_bool("WHISPER_AUTO_DOWNLOAD", true)?;
        let whisper_hf_repo = env_str("WHISPER_HF_REPO", "ggerganov/whisper.cpp");
        let whisper_hf_filename = env_str(
            "WHISPER_HF_FILENAME",
            whisper_model_filename(whisper_model_size),
        );
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
            whisper_model_size,
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

    /// Applies command-line overrides on top of environment-derived values.
    pub fn apply_cli_overrides(&mut self, options: CliOptions) {
        if let Some(host) = options.host {
            self.host = host;
        }
        if let Some(port) = options.port {
            self.port = port;
        }
        if let Some(api_key) = options.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(whisper_model) = options.whisper_model {
            self.whisper_model = whisper_model;
            self.whisper_model_explicit = true;
        }
        if let Some(whisper_model_size) = options.whisper_model_size {
            self.whisper_model_size = whisper_model_size;
            self.whisper_hf_filename = whisper_model_filename(whisper_model_size).to_string();
        }
        if let Some(whisper_auto_download) = options.whisper_auto_download {
            self.whisper_auto_download = whisper_auto_download;
        }
        if let Some(whisper_hf_repo) = options.whisper_hf_repo {
            self.whisper_hf_repo = whisper_hf_repo;
        }
        if let Some(whisper_hf_filename) = options.whisper_hf_filename {
            self.whisper_hf_filename = whisper_hf_filename;
        }
        if let Some(whisper_cache_dir) = options.whisper_cache_dir {
            self.whisper_cache_dir = whisper_cache_dir;
        }
        if let Some(hf_token) = options.hf_token {
            self.hf_token = Some(hf_token);
        }
        if let Some(api_model_alias) = options.api_model_alias {
            self.api_model_alias = api_model_alias;
        }
        if let Some(backend_kind) = options.backend_kind {
            self.backend_kind = backend_kind;
        }
        if let Some(whisper_parallelism) = options.whisper_parallelism {
            self.whisper_parallelism = whisper_parallelism;
        }

        if !self.whisper_model_explicit {
            self.whisper_model = format!("{}/{}", self.whisper_cache_dir, self.whisper_hf_filename);
        }
    }
}

impl CliOptions {
    pub fn from_args() -> Result<Self, AppError> {
        let program = env::args()
            .next()
            .unwrap_or_else(|| "whisper-openai-rust".to_string());
        Self::from_tokens(program, env::args().skip(1))
    }

    pub fn print_help(program: &str) {
        println!(
            "Usage: {program} [OPTIONS]\n\n\
Options:\n\
  -h, --help                          Show this help and exit\n\
      --host <HOST>                       Bind host (env: HOST)\n\
      --port <PORT>                       Bind port (env: PORT)\n\
      --api-key <API_KEY>                 Require bearer token (env: API_KEY)\n\
      --whisper-model <PATH>              Local model path (env: WHISPER_MODEL)\n\
      --whisper-model-size <SIZE>         Model size tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v1|large-v2|large-v3|large-v3-turbo|turbo (env: WHISPER_MODEL_SIZE)\n\
      --whisper-auto-download <BOOL>      Download missing model (env: WHISPER_AUTO_DOWNLOAD)\n\
      --whisper-hf-repo <REPO>            HF repo for model download (env: WHISPER_HF_REPO)\n\
      --whisper-hf-filename <FILE>        HF model filename (env: WHISPER_HF_FILENAME)\n\
      --whisper-cache-dir <DIR>           Local model cache dir (env: WHISPER_CACHE_DIR)\n\
      --hf-token <TOKEN>                  HF auth token (env: HF_TOKEN)\n\
      --whisper-model-alias <ALIAS>       Extra accepted model id (env: WHISPER_MODEL_ALIAS)\n\
      --whisper-backend <BACKEND>         Inference backend (env: WHISPER_BACKEND)\n\
      --whisper-parallelism <N>           Inference workers in range [1, 8] (env: WHISPER_PARALLELISM)\n\n\
Notes:\n\
  - Command-line options override environment variable values.\n\
  - Option values accept both --option value and --option=value forms."
        );
    }

    fn from_tokens<I>(program: String, tokens: I) -> Result<Self, AppError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut options = Self::default();
        let mut iter = tokens.into_iter().peekable();

        while let Some(token) = iter.next() {
            if token == "-h" || token == "--help" {
                options.help_requested = true;
                continue;
            }

            if !token.starts_with('-') {
                return Err(AppError::internal(format!(
                    "unexpected positional argument {token:?}; run {program} --help"
                )));
            }

            let (name, inline_value) = split_long_option(&token).ok_or_else(|| {
                AppError::internal(format!("unknown argument {token:?}; run {program} --help"))
            })?;

            match name {
                "--host" => {
                    options.host = Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--port" => {
                    let raw = required_option_value(name, inline_value, &mut iter)?;
                    options.port = Some(parse_u16_option("PORT", &raw)?);
                }
                "--api-key" => {
                    options.api_key = Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-model" => {
                    options.whisper_model =
                        Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-model-size" => {
                    let raw = required_option_value(name, inline_value, &mut iter)?;
                    options.whisper_model_size =
                        Some(parse_model_size("WHISPER_MODEL_SIZE", &raw)?);
                }
                "--whisper-auto-download" => {
                    let raw = required_option_value(name, inline_value, &mut iter)?;
                    options.whisper_auto_download =
                        Some(parse_bool_option("WHISPER_AUTO_DOWNLOAD", &raw)?);
                }
                "--whisper-hf-repo" => {
                    options.whisper_hf_repo =
                        Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-hf-filename" => {
                    options.whisper_hf_filename =
                        Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-cache-dir" => {
                    options.whisper_cache_dir =
                        Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--hf-token" => {
                    options.hf_token = Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-model-alias" => {
                    options.api_model_alias =
                        Some(required_option_value(name, inline_value, &mut iter)?);
                }
                "--whisper-backend" => {
                    let raw = required_option_value(name, inline_value, &mut iter)?;
                    options.backend_kind = Some(parse_backend_kind(&raw)?);
                }
                "--whisper-parallelism" => {
                    let raw = required_option_value(name, inline_value, &mut iter)?;
                    options.whisper_parallelism = Some(parse_usize_bounded(
                        "WHISPER_PARALLELISM",
                        &raw,
                        1,
                        MAX_WHISPER_PARALLELISM,
                    )?);
                }
                _ => {
                    return Err(AppError::internal(format!(
                        "unknown argument {token:?}; run {program} --help"
                    )));
                }
            }
        }

        Ok(options)
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

fn env_model_size(name: &str, default: WhisperModelSize) -> Result<WhisperModelSize, AppError> {
    match env::var(name) {
        Ok(raw) => parse_model_size(name, &raw),
        Err(_) => Ok(default),
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

fn split_long_option(token: &str) -> Option<(&str, Option<&str>)> {
    if !token.starts_with("--") {
        return None;
    }
    if let Some((name, value)) = token.split_once('=') {
        Some((name, Some(value)))
    } else {
        Some((token, None))
    }
}

fn required_option_value<I>(
    option_name: &str,
    inline_value: Option<&str>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<String, AppError>
where
    I: Iterator<Item = String>,
{
    if let Some(value) = inline_value {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(AppError::internal(format!(
                "missing value for {option_name}"
            )));
        }
        return Ok(trimmed.to_string());
    }

    let value = iter
        .next()
        .ok_or_else(|| AppError::internal(format!("missing value for {option_name}")))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(AppError::internal(format!(
            "missing value for {option_name}"
        )));
    }
    Ok(trimmed.to_string())
}

fn parse_bool_option(name: &str, raw: &str) -> Result<bool, AppError> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AppError::internal(format!(
            "invalid {name}={raw:?}; expected true/false"
        ))),
    }
}

fn parse_u16_option(name: &str, raw: &str) -> Result<u16, AppError> {
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

fn parse_backend_kind(raw: &str) -> Result<BackendKind, AppError> {
    match raw.trim() {
        "whisper-rs" => Ok(BackendKind::WhisperRs),
        other => Err(AppError::internal(format!(
            "invalid WHISPER_BACKEND={other:?}; expected whisper-rs"
        ))),
    }
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

fn parse_model_size(name: &str, raw: &str) -> Result<WhisperModelSize, AppError> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "tiny" => Ok(WhisperModelSize::Tiny),
        "tiny.en" => Ok(WhisperModelSize::TinyEn),
        "base" => Ok(WhisperModelSize::Base),
        "base.en" => Ok(WhisperModelSize::BaseEn),
        "small" => Ok(WhisperModelSize::Small),
        "small.en" => Ok(WhisperModelSize::SmallEn),
        "medium" => Ok(WhisperModelSize::Medium),
        "medium.en" => Ok(WhisperModelSize::MediumEn),
        "large-v1" => Ok(WhisperModelSize::LargeV1),
        "large-v2" => Ok(WhisperModelSize::LargeV2),
        "large" | "large-v3" => Ok(WhisperModelSize::LargeV3),
        "large-v3-turbo" | "turbo" => Ok(WhisperModelSize::Turbo),
        _ => Err(AppError::internal(format!(
            "invalid {name}={raw:?}; expected one of tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v1|large-v2|large-v3|large-v3-turbo|turbo"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_model_size, parse_usize_bounded, whisper_model_filename, CliOptions, WhisperModelSize,
    };

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

    #[test]
    fn cli_parsing_supports_help_flag() {
        let options = CliOptions::from_tokens(
            "whisper-openai-rust".to_string(),
            vec!["--help".to_string()],
        )
        .unwrap();
        assert!(options.help_requested);
    }

    #[test]
    fn cli_parsing_reads_value_from_equals_form() {
        let options = CliOptions::from_tokens(
            "whisper-openai-rust".to_string(),
            vec!["--port=9001".to_string()],
        )
        .unwrap();
        assert_eq!(options.port, Some(9001));
    }

    #[test]
    fn cli_parsing_rejects_unknown_flags() {
        let err = CliOptions::from_tokens(
            "whisper-openai-rust".to_string(),
            vec!["--unknown".to_string()],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown argument"));
    }

    #[test]
    fn cli_parsing_supports_model_size() {
        let options = CliOptions::from_tokens(
            "whisper-openai-rust".to_string(),
            vec!["--whisper-model-size=medium".to_string()],
        )
        .unwrap();
        assert_eq!(options.whisper_model_size, Some(WhisperModelSize::Medium));
    }

    #[test]
    fn parse_model_size_accepts_large_alias() {
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "large").unwrap(),
            WhisperModelSize::LargeV3
        );
    }

    #[test]
    fn parse_model_size_accepts_english_variants() {
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "tiny.en").unwrap(),
            WhisperModelSize::TinyEn
        );
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "medium.en").unwrap(),
            WhisperModelSize::MediumEn
        );
    }

    #[test]
    fn parse_model_size_accepts_large_model_versions() {
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "large-v1").unwrap(),
            WhisperModelSize::LargeV1
        );
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "large-v2").unwrap(),
            WhisperModelSize::LargeV2
        );
        assert_eq!(
            parse_model_size("WHISPER_MODEL_SIZE", "large-v3-turbo").unwrap(),
            WhisperModelSize::Turbo
        );
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

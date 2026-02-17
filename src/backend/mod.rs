//! Backend abstraction for speech-to-text engines.
//!
//! The HTTP layer depends on the [`Transcriber`] trait instead of a concrete
//! implementation, which keeps request handling decoupled from inference code.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{AppConfig, BackendKind};
use crate::error::AppError;

pub mod whisper_rs;

/// Type of inference task requested by the client.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TaskKind {
    /// Convert speech to text in the same language as the input audio.
    Transcribe,
    /// Convert speech to English text.
    Translate,
}

impl TaskKind {
    /// Returns the wire-format task value used in API responses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

/// Input payload consumed by a transcription backend.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    /// Requested inference task.
    pub task: TaskKind,
    /// Audio samples as 16 kHz mono PCM in `f32` range `[-1.0, 1.0]`.
    pub audio_16khz_mono_f32: Vec<f32>,
    /// Optional language hint such as `"en"`.
    pub language: Option<String>,
    /// Optional initial prompt to bias decoding.
    pub prompt: Option<String>,
    /// Optional sampling temperature in range `[0.0, 1.0]`.
    pub temperature: Option<f32>,
}

/// Timestamped transcript chunk.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    /// Segment start time in seconds.
    pub start_secs: f64,
    /// Segment end time in seconds.
    pub end_secs: f64,
    /// Text content for this segment.
    pub text: String,
}

/// Full inference result returned by a backend.
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    /// Concatenated normalized transcript text.
    pub text: String,
    /// Detected language if available.
    pub language: Option<String>,
    /// Segment-level timing and text details.
    pub segments: Vec<TranscriptSegment>,
}

/// Backend contract implemented by speech-to-text engines.
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Runs inference and returns a transcript result.
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptResult, AppError>;
}

/// Builds the configured backend implementation.
pub fn build_backend(cfg: &AppConfig) -> Result<Arc<dyn Transcriber>, AppError> {
    match cfg.backend_kind {
        BackendKind::WhisperRs => Ok(Arc::new(whisper_rs::WhisperRsBackend::new(cfg.clone())?)),
    }
}

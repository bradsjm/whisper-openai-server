use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{AppConfig, BackendKind};
use crate::error::AppError;

pub mod whisper_rs;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TaskKind {
    Transcribe,
    Translate,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub task: TaskKind,
    pub audio_16khz_mono_f32: Vec<f32>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<TranscriptSegment>,
}

#[async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptResult, AppError>;
}

pub fn build_backend(cfg: &AppConfig) -> Result<Arc<dyn Transcriber>, AppError> {
    match cfg.backend_kind {
        BackendKind::WhisperRs => Ok(Arc::new(whisper_rs::WhisperRsBackend::new(cfg.clone())?)),
    }
}

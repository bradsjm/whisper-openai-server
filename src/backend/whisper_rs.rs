//! `whisper-rs` backend implementation.
//!
//! This backend keeps a pool of Whisper contexts in memory and runs inference
//! on blocking worker threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::task;
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

use crate::backend::{TranscribeRequest, Transcriber, TranscriptResult, TranscriptSegment};
use crate::config::AppConfig;
use crate::error::AppError;
use crate::formats::normalize_text;

/// Local inference backend powered by `whisper-rs`.
pub struct WhisperRsBackend {
    model_path: String,
    contexts: Vec<Arc<Mutex<WhisperContext>>>,
    next_context_idx: AtomicUsize,
}

impl WhisperRsBackend {
    /// Loads the configured Whisper model and prepares reusable contexts.
    pub fn new(cfg: AppConfig) -> Result<Self, AppError> {
        let model_path = cfg.whisper_model.clone();
        let mut contexts = Vec::with_capacity(cfg.whisper_parallelism);
        for worker_idx in 0..cfg.whisper_parallelism {
            let context =
                WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
                    .map_err(|err| {
                        AppError::backend(format!(
                            "failed to load model at {model_path:?} for worker {}: {err}",
                            worker_idx + 1
                        ))
                    })?;
            contexts.push(Arc::new(Mutex::new(context)));
        }

        Ok(Self {
            model_path,
            contexts,
            next_context_idx: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl Transcriber for WhisperRsBackend {
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptResult, AppError> {
        let model_path = self.model_path.clone();
        let context_idx = self.next_context_idx.fetch_add(1, Ordering::Relaxed) % self.contexts.len();
        let context = Arc::clone(&self.contexts[context_idx]);
        task::spawn_blocking(move || run_whisper_rs(req, &model_path, context))
            .await
            .map_err(|err| AppError::backend(format!("whisper-rs worker task failed: {err}")))?
    }
}

fn run_whisper_rs(
    req: TranscribeRequest,
    model_path: &str,
    context: Arc<Mutex<WhisperContext>>,
) -> Result<TranscriptResult, AppError> {
    let context_guard = context
        .lock()
        .map_err(|_| AppError::backend("failed to lock whisper model context"))?;

    let mut state = context_guard
        .create_state()
        .map_err(|err| AppError::backend(format!("failed to create whisper state: {err}")))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    if let Some(language) = req.language.as_deref() {
        let trimmed = language.trim();
        if !trimmed.is_empty() {
            params.set_language(Some(trimmed));
        }
    } else {
        params.set_detect_language(true);
    }
    if let Some(prompt) = req.prompt.as_deref() {
        let trimmed = prompt.trim();
        if !trimmed.is_empty() {
            params.set_initial_prompt(trimmed);
        }
    }
    if let Some(temp) = req.temperature {
        params.set_temperature(temp);
    }
    params.set_translate(matches!(req.task, crate::backend::TaskKind::Translate));

    state
        .full(params, &req.audio_16khz_mono_f32)
        .map_err(|err| {
            AppError::backend(format!(
                "whisper inference failed using {model_path:?}: {err}"
            ))
        })?;

    let count = state.full_n_segments();
    let mut segments = Vec::with_capacity(count as usize);
    for i in 0..count {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let text = seg
            .to_str_lossy()
            .map_err(|err| AppError::backend(format!("failed to read segment text: {err}")))?
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }

        segments.push(TranscriptSegment {
            start_secs: (seg.start_timestamp() as f64) * 0.01,
            end_secs: (seg.end_timestamp() as f64) * 0.01,
            text,
        });
    }

    let text = normalize_text(
        &segments
            .iter()
            .map(|seg| seg.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );

    let detected_language = if let Some(lang) = req.language {
        Some(lang)
    } else {
        get_lang_str(state.full_lang_id_from_state()).map(ToOwned::to_owned)
    };

    Ok(TranscriptResult {
        text,
        language: detected_language,
        segments,
    })
}

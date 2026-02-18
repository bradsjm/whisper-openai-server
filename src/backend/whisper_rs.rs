//! `whisper-rs` backend implementation.
//!
//! This backend keeps a pool of Whisper contexts in memory and runs inference
//! on blocking worker threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::task;
use tracing::{info, warn};
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

use crate::backend::{TranscribeRequest, Transcriber, TranscriptResult, TranscriptSegment};
use crate::config::{AccelerationKind, AppConfig};
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
        let (contexts, effective_acceleration) = match cfg.acceleration_kind {
            AccelerationKind::None => (
                build_contexts(&model_path, cfg.whisper_parallelism, AccelerationKind::None)?,
                AccelerationKind::None,
            ),
            AccelerationKind::Metal => {
                match build_contexts(
                    &model_path,
                    cfg.whisper_parallelism,
                    AccelerationKind::Metal,
                ) {
                    Ok(contexts) => (contexts, AccelerationKind::Metal),
                    Err(err) if !cfg.acceleration_explicit => {
                        warn!(
                            error = %err,
                            requested_acceleration = "metal",
                            fallback_acceleration = "none",
                            "metal initialization failed; falling back to cpu"
                        );
                        (
                            build_contexts(&model_path, cfg.whisper_parallelism, AccelerationKind::None).map_err(
                                |cpu_err| {
                                    AppError::backend(format!(
                                        "failed to initialize metal acceleration ({err}); cpu fallback also failed: {cpu_err}"
                                    ))
                                },
                            )?,
                            AccelerationKind::None,
                        )
                    }
                    Err(err) => {
                        return Err(AppError::backend(format!(
                            "failed to initialize whisper with metal acceleration: {err}"
                        )));
                    }
                }
            }
            AccelerationKind::Cuda => {
                match build_contexts(&model_path, cfg.whisper_parallelism, AccelerationKind::Cuda) {
                    Ok(contexts) => (contexts, AccelerationKind::Cuda),
                    Err(err) if !cfg.acceleration_explicit => {
                        warn!(
                            error = %err,
                            requested_acceleration = "cuda",
                            fallback_acceleration = "none",
                            "cuda initialization failed; falling back to cpu"
                        );
                        (
                            build_contexts(&model_path, cfg.whisper_parallelism, AccelerationKind::None).map_err(
                                |cpu_err| {
                                    AppError::backend(format!(
                                        "failed to initialize cuda acceleration ({err}); cpu fallback also failed: {cpu_err}"
                                    ))
                                },
                            )?,
                            AccelerationKind::None,
                        )
                    }
                    Err(err) => {
                        return Err(AppError::backend(format!(
                            "failed to initialize whisper with cuda acceleration: {err}"
                        )));
                    }
                }
            }
        };

        info!(
            requested_acceleration = %cfg.acceleration_kind.as_str(),
            effective_acceleration = %effective_acceleration.as_str(),
            whisper_parallelism = cfg.whisper_parallelism,
            "initialized whisper acceleration"
        );

        Ok(Self {
            model_path,
            contexts,
            next_context_idx: AtomicUsize::new(0),
        })
    }
}

fn build_contexts(
    model_path: &str,
    whisper_parallelism: usize,
    acceleration: AccelerationKind,
) -> Result<Vec<Arc<Mutex<WhisperContext>>>, AppError> {
    let mut contexts = Vec::with_capacity(whisper_parallelism);
    let use_gpu = acceleration != AccelerationKind::None;
    let acceleration_name = acceleration.as_str();

    for worker_idx in 0..whisper_parallelism {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(use_gpu);

        let context = WhisperContext::new_with_params(model_path, params).map_err(|err| {
            AppError::backend(format!(
                "failed to load model at {model_path:?} for worker {} using acceleration={acceleration_name}: {err}",
                worker_idx + 1,
            ))
        })?;

        contexts.push(Arc::new(Mutex::new(context)));
    }

    Ok(contexts)
}

#[async_trait]
impl Transcriber for WhisperRsBackend {
    async fn transcribe(&self, req: TranscribeRequest) -> Result<TranscriptResult, AppError> {
        let model_path = self.model_path.clone();
        let context_idx =
            self.next_context_idx.fetch_add(1, Ordering::Relaxed) % self.contexts.len();
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
    params.set_no_timestamps(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_max_initial_ts(5.0);
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

    let (mut count, mut segments) = extract_segments(&state)?;

    if count == 0 && req.language.is_none() {
        let mut fallback = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        fallback.set_no_timestamps(false);
        fallback.set_print_special(false);
        fallback.set_print_progress(false);
        fallback.set_print_realtime(false);
        fallback.set_print_timestamps(false);
        fallback.set_max_initial_ts(5.0);
        fallback.set_language(Some("en"));
        if let Some(prompt) = req.prompt.as_deref() {
            let trimmed = prompt.trim();
            if !trimmed.is_empty() {
                fallback.set_initial_prompt(trimmed);
            }
        }
        if let Some(temp) = req.temperature {
            fallback.set_temperature(temp);
        }
        fallback.set_translate(matches!(req.task, crate::backend::TaskKind::Translate));

        state
            .full(fallback, &req.audio_16khz_mono_f32)
            .map_err(|err| {
                AppError::backend(format!(
                    "whisper fallback inference failed using {model_path:?}: {err}"
                ))
            })?;
        let (fallback_count, fallback_segments) = extract_segments(&state)?;
        if fallback_count > 0 {
            warn!(
                audio_samples = req.audio_16khz_mono_f32.len(),
                segment_count = fallback_count,
                "whisper fallback used fixed language after empty auto-detect output"
            );
            count = fallback_count;
            segments = fallback_segments;
        }
    }

    if looks_like_non_speech_only(&segments) {
        let mut aggressive = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        aggressive.set_no_timestamps(false);
        aggressive.set_print_special(false);
        aggressive.set_print_progress(false);
        aggressive.set_print_realtime(false);
        aggressive.set_print_timestamps(false);
        aggressive.set_max_initial_ts(5.0);
        aggressive.set_no_speech_thold(1.0);
        aggressive.set_suppress_blank(false);

        if let Some(language) = req.language.as_deref() {
            let trimmed = language.trim();
            if !trimmed.is_empty() {
                aggressive.set_language(Some(trimmed));
            }
        } else {
            aggressive.set_detect_language(true);
        }
        if let Some(prompt) = req.prompt.as_deref() {
            let trimmed = prompt.trim();
            if !trimmed.is_empty() {
                aggressive.set_initial_prompt(trimmed);
            }
        }
        if let Some(temp) = req.temperature {
            aggressive.set_temperature(temp);
        }
        aggressive.set_translate(matches!(req.task, crate::backend::TaskKind::Translate));

        state
            .full(aggressive, &req.audio_16khz_mono_f32)
            .map_err(|err| {
                AppError::backend(format!(
                    "whisper aggressive fallback failed using {model_path:?}: {err}"
                ))
            })?;

        let (aggressive_count, aggressive_segments) = extract_segments(&state)?;
        if transcript_score(&aggressive_segments) > transcript_score(&segments) {
            warn!(
                audio_samples = req.audio_16khz_mono_f32.len(),
                old_segment_count = count,
                new_segment_count = aggressive_count,
                "whisper aggressive fallback replaced non-speech-only transcript"
            );
            count = aggressive_count;
            segments = aggressive_segments;
        }
    }

    let text = normalize_text(
        &segments
            .iter()
            .map(|seg| seg.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );

    if text.is_empty() {
        warn!(
            audio_samples = req.audio_16khz_mono_f32.len(),
            segment_count = count,
            "whisper inference completed with empty transcript"
        );
    }

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

fn extract_segments(
    state: &whisper_rs::WhisperState,
) -> Result<(i32, Vec<TranscriptSegment>), AppError> {
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

    Ok((count, segments))
}

fn looks_like_non_speech_only(segments: &[TranscriptSegment]) -> bool {
    !segments.is_empty()
        && segments
            .iter()
            .all(|seg| is_parenthesized_event(seg.text.as_str()))
}

fn is_parenthesized_event(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with('(') && trimmed.ends_with(')') && !trimmed.contains(' ')
}

fn transcript_score(segments: &[TranscriptSegment]) -> usize {
    normalize_text(
        &segments
            .iter()
            .map(|seg| seg.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    )
    .len()
}

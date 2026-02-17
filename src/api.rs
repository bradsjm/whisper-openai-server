//! HTTP API surface compatible with key OpenAI Whisper endpoints.
//!
//! This module owns request parsing, authentication, input validation, and
//! response formatting while delegating inference to a backend implementation.

use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::audio::{decode_to_mono_16khz_f32, validate_extension};
use crate::backend::{TaskKind, TranscribeRequest, Transcriber};
use crate::config::AppConfig;
use crate::error::AppError;
use crate::formats::{segments_to_srt, segments_to_vtt, ResponseFormat};

/// Human-readable service name returned by health endpoints.
pub const APP_NAME: &str = "whisper-openai-rust";
/// Service version string returned by health endpoints.
pub const APP_VERSION: &str = "0.1.0";

/// Shared state injected into all route handlers.
pub struct AppState {
    /// Runtime configuration loaded at startup.
    pub cfg: AppConfig,
    /// Active inference backend implementation.
    pub backend: Arc<dyn Transcriber>,
}

impl AppState {
    /// Constructs shared handler state.
    pub fn new(cfg: AppConfig, backend: Arc<dyn Transcriber>) -> Self {
        Self { cfg, backend }
    }
}

/// Builds the Axum router for all public endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1", get(v1))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/translations", post(audio_translations))
        .with_state(state)
}

/// Root status endpoint (`GET /`).
pub async fn root(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    require_auth(&state.cfg, &headers)?;
    Ok(Json(json!({
        "status": "ok",
        "name": APP_NAME,
        "version": APP_VERSION,
        "model": state.cfg.api_model_alias,
    })))
}

/// Alias status endpoint (`GET /health`).
pub async fn health(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    root(State(state), headers).await
}

/// API root status endpoint (`GET /v1`).
pub async fn v1(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    root(State(state), headers).await
}

/// Lists accepted model identifiers (`GET /v1/models`).
pub async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    require_auth(&state.cfg, &headers)?;
    let data = state
        .cfg
        .accepted_model_ids()
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "owned_by": "local", "permission": []}))
        .collect::<Vec<_>>();

    Ok(Json(json!({"object": "list", "data": data})))
}

/// Handles speech-to-text transcription requests (`POST /v1/audio/transcriptions`).
pub async fn audio_transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, AppError> {
    handle_audio_request(state, headers, multipart, TaskKind::Transcribe).await
}

/// Handles speech-to-English translation requests (`POST /v1/audio/translations`).
pub async fn audio_translations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, AppError> {
    handle_audio_request(state, headers, multipart, TaskKind::Translate).await
}

struct AudioForm {
    extension: String,
    bytes: Vec<u8>,
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    response_format: ResponseFormat,
    temperature: Option<f32>,
}

async fn handle_audio_request(
    state: Arc<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
    task: TaskKind,
) -> Result<Response, AppError> {
    require_auth(&state.cfg, &headers)?;

    let form = parse_audio_form(&mut multipart).await?;
    validate_requested_model(&state.cfg, &form.model)?;

    let decode_bytes = form.bytes;
    let extension_hint = form.extension;
    let audio_16khz_mono_f32 = tokio::task::spawn_blocking(move || {
        decode_to_mono_16khz_f32(&decode_bytes, &extension_hint)
    })
    .await
    .map_err(|err| AppError::internal(format!("audio decode task failed: {err}")))??;

    let request = TranscribeRequest {
        task,
        audio_16khz_mono_f32,
        language: form.language,
        prompt: form.prompt,
        temperature: form.temperature,
    };

    let result = state.backend.transcribe(request).await?;

    match form.response_format {
        ResponseFormat::Json => Ok(Json(json!({"text": result.text})).into_response()),
        ResponseFormat::Text => Ok((
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response()),
        ResponseFormat::Srt => Ok((
            [(header::CONTENT_TYPE, "application/x-subrip; charset=utf-8")],
            segments_to_srt(&result.segments),
        )
            .into_response()),
        ResponseFormat::Vtt => Ok((
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            segments_to_vtt(&result.segments),
        )
            .into_response()),
        ResponseFormat::VerboseJson => {
            let language = result.language.unwrap_or_else(|| "unknown".to_string());
            let segments = result
                .segments
                .into_iter()
                .enumerate()
                .map(|(idx, seg)| {
                    json!({
                        "id": idx,
                        "start": seg.start_secs,
                        "end": seg.end_secs,
                        "text": seg.text,
                    })
                })
                .collect::<Vec<_>>();

            Ok(Json(json!({
                "task": task.as_str(),
                "language": language,
                "text": result.text,
                "segments": segments,
            }))
            .into_response())
        }
    }
}

/// Parses and validates multipart form fields for audio endpoints.
async fn parse_audio_form(multipart: &mut Multipart) -> Result<AudioForm, AppError> {
    let mut file_name: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut model = "whisper-1".to_string();
    let mut language: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut response_format = ResponseFormat::Json;
    let mut temperature: Option<f32> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| AppError::bad_multipart(format!("invalid multipart body: {err}")))?
    {
        let Some(name) = field.name().map(ToOwned::to_owned) else {
            continue;
        };

        match name.as_str() {
            "file" => {
                let filename = field
                    .file_name()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| AppError::bad_multipart("file field is missing filename"))?;
                let bytes = field.bytes().await.map_err(|err| {
                    AppError::bad_multipart(format!("failed to read file bytes: {err}"))
                })?;
                file_name = Some(filename);
                file_bytes = Some(bytes.to_vec());
            }
            "model" => {
                model = field
                    .text()
                    .await
                    .map_err(|err| AppError::bad_multipart(format!("invalid model field: {err}")))?
                    .trim()
                    .to_string();
            }
            "language" => {
                language = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| {
                            AppError::bad_multipart(format!("invalid language field: {err}"))
                        })?
                        .trim()
                        .to_string(),
                )
                .filter(|v| !v.is_empty());
            }
            "prompt" => {
                prompt = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| {
                            AppError::bad_multipart(format!("invalid prompt field: {err}"))
                        })?
                        .trim()
                        .to_string(),
                )
                .filter(|v| !v.is_empty());
            }
            "response_format" => {
                let raw = field
                    .text()
                    .await
                    .map_err(|err| {
                        AppError::bad_multipart(format!("invalid response_format field: {err}"))
                    })?
                    .trim()
                    .to_string();
                response_format = ResponseFormat::parse(&raw)?;
            }
            "temperature" => {
                let raw = field
                    .text()
                    .await
                    .map_err(|err| {
                        AppError::bad_multipart(format!("invalid temperature field: {err}"))
                    })?
                    .trim()
                    .to_string();

                if !raw.is_empty() {
                    let value = raw.parse::<f32>().map_err(|_| {
                        AppError::invalid_request(
                            format!("invalid temperature={raw:?}; expected float"),
                            Some("temperature"),
                            Some("invalid_temperature"),
                        )
                    })?;
                    if !value.is_finite() {
                        return Err(AppError::invalid_request(
                            format!("invalid temperature={raw:?}; expected a finite float"),
                            Some("temperature"),
                            Some("invalid_temperature"),
                        ));
                    }
                    if !(0.0..=1.0).contains(&value) {
                        return Err(AppError::invalid_request(
                            format!(
                                "invalid temperature={raw:?}; expected a value in range [0.0, 1.0]"
                            ),
                            Some("temperature"),
                            Some("invalid_temperature"),
                        ));
                    }
                    temperature = Some(value);
                }
            }
            _ => {}
        }
    }

    let filename = file_name.ok_or_else(|| {
        AppError::invalid_request("missing required multipart field: file", Some("file"), None)
    })?;
    let extension = validate_extension(&filename)?;
    let bytes = file_bytes
        .ok_or_else(|| AppError::invalid_request("missing file content", Some("file"), None))?;
    if bytes.is_empty() {
        return Err(AppError::invalid_request(
            "uploaded file is empty",
            Some("file"),
            Some("empty_file"),
        ));
    }

    if model.is_empty() {
        return Err(AppError::invalid_request(
            "model must not be empty",
            Some("model"),
            Some("invalid_model"),
        ));
    }

    Ok(AudioForm {
        extension,
        bytes,
        model,
        language,
        prompt,
        response_format,
        temperature,
    })
}

/// Verifies that the requested model id is supported by current configuration.
fn validate_requested_model(cfg: &AppConfig, requested_model: &str) -> Result<(), AppError> {
    if cfg
        .accepted_model_ids()
        .iter()
        .any(|id| id == requested_model)
    {
        return Ok(());
    }

    Err(AppError::invalid_request(
        format!(
            "unsupported model={requested_model:?}; accepted models: {}",
            cfg.accepted_model_ids().join(",")
        ),
        Some("model"),
        Some("invalid_model"),
    ))
}

/// Enforces optional bearer-token authentication.
fn require_auth(cfg: &AppConfig, headers: &HeaderMap) -> Result<(), AppError> {
    let Some(expected_api_key) = cfg.api_key.as_deref() else {
        return Ok(());
    };

    let Some(raw) = headers.get(header::AUTHORIZATION) else {
        return Err(AppError::unauthorized("missing bearer token"));
    };

    let value = raw
        .to_str()
        .map_err(|_| AppError::unauthorized("invalid authorization header"))?;

    let mut parts = value.split_whitespace();
    let scheme = parts
        .next()
        .ok_or_else(|| AppError::unauthorized("missing bearer token"))?;
    let token = parts
        .next()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| AppError::unauthorized("missing bearer token"))?;
    if parts.next().is_some() || !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AppError::unauthorized("missing bearer token"));
    }

    if token != expected_api_key {
        return Err(AppError::unauthorized("invalid token"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tower::ServiceExt;

    use crate::backend::{TranscribeRequest, Transcriber, TranscriptResult, TranscriptSegment};
    use crate::config::{AppConfig, BackendKind};
    use crate::error::AppError;

    use super::{build_router, AppState};

    #[derive(Clone)]
    struct MockBackend;

    #[async_trait]
    impl Transcriber for MockBackend {
        async fn transcribe(&self, _req: TranscribeRequest) -> Result<TranscriptResult, AppError> {
            Ok(TranscriptResult {
                text: "hello world".to_string(),
                language: Some("en".to_string()),
                segments: vec![TranscriptSegment {
                    start_secs: 0.0,
                    end_secs: 1.2,
                    text: "hello world".to_string(),
                }],
            })
        }
    }

    fn test_cfg(api_key: Option<&str>) -> AppConfig {
        AppConfig {
            host: "127.0.0.1".to_string(),
            port: 8000,
            api_key: api_key.map(ToOwned::to_owned),
            whisper_model: "dummy".to_string(),
            whisper_model_explicit: true,
            whisper_auto_download: false,
            whisper_hf_repo: "ggerganov/whisper.cpp".to_string(),
            whisper_hf_filename: "ggml-small.bin".to_string(),
            whisper_cache_dir: "/tmp".to_string(),
            hf_token: None,
            api_model_alias: "whisper-mlx".to_string(),
            backend_kind: BackendKind::WhisperRs,
            whisper_parallelism: 1,
        }
    }

    fn app(api_key: Option<&str>) -> axum::Router {
        let state = Arc::new(AppState::new(test_cfg(api_key), Arc::new(MockBackend)));
        build_router(state)
    }

    async fn parse_json_response(res: axum::response::Response) -> Value {
        let bytes = to_bytes(res.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json body")
    }

    #[tokio::test]
    async fn models_requires_auth_when_api_key_set() {
        let app = app(Some("secret"));

        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .body(Body::empty())
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        let payload = parse_json_response(res).await;
        assert_eq!(payload["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn models_lists_alias_and_whisper_1() {
        let app = app(Some("secret"));

        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .header("Authorization", "Bearer secret")
            .body(Body::empty())
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::OK);

        let payload = parse_json_response(res).await;
        let ids = payload["data"]
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect::<Vec<_>>();

        assert!(ids.contains(&"whisper-1"));
        assert!(ids.contains(&"whisper-mlx"));
    }

    #[tokio::test]
    async fn models_accept_lowercase_bearer_scheme() {
        let app = app(Some("secret"));

        let req = Request::builder()
            .uri("/v1/models")
            .method("GET")
            .header("Authorization", "bearer secret")
            .body(Body::empty())
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn transcriptions_reject_mp4() {
        let app = app(None);
        let boundary = "X-BOUNDARY";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"bad.mp4\"\r\nContent-Type: video/mp4\r\n\r\nnot-a-real-media\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{b}--\r\n",
            b = boundary
        );

        let req = Request::builder()
            .uri("/v1/audio/transcriptions")
            .method("POST")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

        let payload = parse_json_response(res).await;
        assert_eq!(payload["error"]["code"], "unsupported_media_type");
    }

    #[tokio::test]
    async fn transcriptions_validate_model_field() {
        let app = app(None);
        let boundary = "X-BOUNDARY";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ok.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF____WAVE\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nunknown-model\r\n--{b}--\r\n",
            b = boundary
        );

        let req = Request::builder()
            .uri("/v1/audio/transcriptions")
            .method("POST")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        let payload = parse_json_response(res).await;
        assert_eq!(payload["error"]["code"], "invalid_model");
    }

    #[tokio::test]
    async fn transcriptions_reject_non_finite_temperature() {
        let app = app(None);
        let boundary = "X-BOUNDARY";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ok.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF____WAVE\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{b}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\nNaN\r\n--{b}--\r\n",
            b = boundary
        );

        let req = Request::builder()
            .uri("/v1/audio/transcriptions")
            .method("POST")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        let payload = parse_json_response(res).await;
        assert_eq!(payload["error"]["code"], "invalid_temperature");
    }

    #[tokio::test]
    async fn transcriptions_reject_out_of_range_temperature() {
        let app = app(None);
        let boundary = "X-BOUNDARY";
        let body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ok.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF____WAVE\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{b}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n1.5\r\n--{b}--\r\n",
            b = boundary
        );

        let req = Request::builder()
            .uri("/v1/audio/transcriptions")
            .method("POST")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .expect("request");

        let res = app.oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        let payload = parse_json_response(res).await;
        assert_eq!(payload["error"]["code"], "invalid_temperature");
    }
}

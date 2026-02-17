//! Application error types and HTTP-to-OpenAI error mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Error model used throughout request parsing, validation, and inference.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    Unauthorized(String),
    #[error("{message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
        code: Option<String>,
        status: StatusCode,
    },
    #[error("{0}")]
    UnsupportedMediaType(String),
    #[error("{0}")]
    BadMultipart(String),
    #[error("{0}")]
    Backend(String),
    #[error("{0}")]
    Internal(String),
}

impl AppError {
    /// Creates a `401 Unauthorized` error.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::Unauthorized(message.into())
    }

    /// Creates an `invalid_request_error` payload with status `400`.
    pub fn invalid_request(
        message: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: param.map(ToOwned::to_owned),
            code: code.map(ToOwned::to_owned),
            status: StatusCode::BAD_REQUEST,
        }
    }

    /// Creates a `415 Unsupported Media Type` style error.
    pub fn unsupported_media_type(message: impl Into<String>) -> Self {
        Self::UnsupportedMediaType(message.into())
    }

    /// Creates a multipart parsing/shape validation error.
    pub fn bad_multipart(message: impl Into<String>) -> Self {
        Self::BadMultipart(message.into())
    }

    /// Creates an internal inference/backend error.
    pub fn backend(message: impl Into<String>) -> Self {
        Self::Backend(message.into())
    }

    /// Creates a generic internal server error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

#[derive(Debug, Serialize)]
struct OpenAiErrorPayload {
    error: OpenAiError,
}

#[derive(Debug, Serialize)]
struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, payload) = match self {
            AppError::Unauthorized(message) => (
                StatusCode::UNAUTHORIZED,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "authentication_error".to_string(),
                        param: None,
                        code: Some("invalid_api_key".to_string()),
                    },
                },
            ),
            AppError::InvalidRequest {
                message,
                param,
                code,
                status,
            } => (
                status,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "invalid_request_error".to_string(),
                        param,
                        code,
                    },
                },
            ),
            AppError::UnsupportedMediaType(message) => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "invalid_request_error".to_string(),
                        param: Some("file".to_string()),
                        code: Some("unsupported_media_type".to_string()),
                    },
                },
            ),
            AppError::BadMultipart(message) => (
                StatusCode::BAD_REQUEST,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "invalid_request_error".to_string(),
                        param: Some("file".to_string()),
                        code: Some("invalid_multipart".to_string()),
                    },
                },
            ),
            AppError::Backend(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "server_error".to_string(),
                        param: None,
                        code: Some("inference_failed".to_string()),
                    },
                },
            ),
            AppError::Internal(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                OpenAiErrorPayload {
                    error: OpenAiError {
                        message,
                        error_type: "server_error".to_string(),
                        param: None,
                        code: Some("internal_error".to_string()),
                    },
                },
            ),
        };

        (status, Json(payload)).into_response()
    }
}

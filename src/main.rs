//! Application entry point for the local Whisper-compatible HTTP server.
//!
//! This crate is a binary (not a library), so this file wires modules together,
//! starts the Axum server, and handles graceful shutdown signals.

mod api;
mod audio;
mod backend;
mod config;
mod error;
mod formats;
mod model_store;

use std::sync::Arc;

use tracing::info;

use crate::api::{build_router, AppState};
use crate::backend::build_backend;
use crate::config::{AppConfig, CliOptions, MAX_WHISPER_PARALLELISM};
use crate::model_store::ensure_model_ready;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "whisper_openai_server=info,axum=info".into()),
        )
        .compact()
        .init();

    let cli_options = CliOptions::from_args()?;
    if cli_options.help_requested {
        let program = std::env::args()
            .next()
            .unwrap_or_else(|| "whisper-openai-server".to_string());
        CliOptions::print_help(&program);
        return Ok(());
    }

    let mut cfg = AppConfig::from_env()?;
    cfg.apply_cli_overrides(cli_options);
    ensure_model_ready(&mut cfg)?;
    let backend = build_backend(&cfg)?;
    let state = Arc::new(AppState::new(cfg.clone(), backend));

    let app = build_router(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!(
        host = %cfg.host,
        port = cfg.port,
        model = %cfg.whisper_model,
        backend = ?cfg.backend_kind,
        acceleration = %cfg.acceleration_kind.as_str(),
        whisper_parallelism = cfg.whisper_parallelism,
        max_whisper_parallelism = MAX_WHISPER_PARALLELISM,
        "starting whisper-openai-server"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Waits for a shutdown signal and then returns.
///
/// On Unix systems this listens for both Ctrl+C and SIGTERM.
/// On non-Unix systems this listens for Ctrl+C only.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
            let _ = sigterm.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

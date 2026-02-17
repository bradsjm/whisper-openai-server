mod api;
mod audio;
mod backend;
mod config;
mod error;
mod formats;

use std::sync::Arc;

use tracing::info;

use crate::api::{build_router, AppState};
use crate::backend::build_backend;
use crate::config::AppConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "whisper_openai_rust=info,axum=info".into()),
        )
        .compact()
        .init();

    let cfg = AppConfig::from_env()?;
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
        "starting whisper-openai-rust"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

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

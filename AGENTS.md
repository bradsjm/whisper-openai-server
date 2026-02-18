# AGENTS.md

Guidance for agentic coding assistants working in `whisper-openai-server`.

## Project Overview

- Language: Rust (`edition = 2021`)
- Crate type: binary application (entry point: `src/main.rs`)
- Framework/runtime: Axum + Tokio
- Inference backend: `whisper-rs`
- Current layout is source-only under `src/` (no `tests/` directory today)

## Build, Run, Lint, and Test Commands

Run commands from repository root: `/Users/user/whisper-openai-server`.

### Build

- Debug build: `cargo build`
- Release build: `cargo build --release`
- Build without running: `cargo check`

### Run

- Run release server directly: `cargo run --release`
- Run with helper script: `./run.sh`
- Show CLI options: `cargo run --release -- --help`

### Format

- Format code: `cargo fmt`
- Check formatting in CI mode: `cargo fmt -- --check`

### Lint

- Standard clippy pass: `cargo clippy --all-targets --all-features`
- Strict clippy (recommended before merging):
  - `cargo clippy --all-targets --all-features -- -D warnings`

### Test

- Run all tests: `cargo test`
- Run tests in release mode: `cargo test --release`
- Run one test by name substring:
  - `cargo test parse_model_size_accepts_large_alias`
- Run one exact test name:
  - `cargo test parse_model_size_accepts_large_alias -- --exact`
- Run one module-scoped unit test:
  - `cargo test config::tests::cli_parsing_supports_help_flag`
- Run one async API test:
  - `cargo test api::tests::models_requires_auth_when_api_key_set`
- Show test output (for debugging):
  - `cargo test <test_name> -- --nocapture`

### Suggested Validation Sequence

1. `cargo fmt -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test`

## Code Style and Conventions

This codebase is already consistent; follow existing local patterns over generic defaults.

### Formatting and Layout

- Use rustfmt default style (no custom `rustfmt.toml` found).
- Use 4-space indentation, trailing commas in multiline literals, and short early returns.
- Keep modules focused by concern:
  - `api.rs` for HTTP/validation/auth
  - `audio.rs` for decode/extension checks
  - `backend/*` for transcription engine abstraction and implementation
  - `config.rs` for env/CLI parsing and validation
  - `error.rs` for app/OpenAI error mapping

### Imports

- Group imports in this order:
  1) `std` imports
  2) third-party crate imports
  3) local `crate::...` imports
- Prefer explicit imports over wildcard imports in non-test code.
- In tests, `use super::*;` is acceptable where already used.

### Types and Data Modeling

- Prefer strong enums/structs for domain states (`TaskKind`, `ResponseFormat`, `BackendKind`).
- Derive common traits explicitly (`Debug`, `Clone`, `Copy`, `Eq`, `PartialEq`) where useful.
- Keep public structs documented with concise rustdoc comments.
- Use `Option<T>` for optional request/config fields; avoid sentinel values.
- Keep API wire-compatible fields as strings when required by OpenAI-like contracts.

### Naming

- `snake_case` for functions, variables, modules, and test names.
- `PascalCase` for structs/enums/traits.
- `SCREAMING_SNAKE_CASE` for constants.
- Favor descriptive names that encode constraints (e.g., `audio_16khz_mono_f32`).

### Error Handling

- Use `AppError` variants and constructor helpers (`invalid_request`, `unsupported_media_type`, etc.).
- Include actionable, context-rich error messages (parameter name, expected range, source failure).
- Convert external errors with `map_err` and preserve enough detail to debug quickly.
- Return errors; do not panic in runtime code.
- `unwrap`/`expect` are acceptable in tests, but avoid them in production paths.

### Async, Blocking Work, and Concurrency

- Keep handlers async and non-blocking.
- Move CPU/blocking operations to `tokio::task::spawn_blocking` (as done for decode/inference).
- Share long-lived state via `Arc`.
- If using `Mutex`, ensure lock scope is minimal and never held across `.await` points.
- For worker pools, preserve deterministic bounds from config (`WHISPER_PARALLELISM` range).

### Validation and API Behavior

- Validate request fields early and fail fast with explicit `AppError` values.
- Preserve existing API compatibility:
  - accepted model IDs include `whisper-1` and optional alias
  - error payload shape mirrors OpenAI style
  - `response_format` supports `json|text|verbose_json|srt|vtt`
- Keep extension allowlist behavior strict (`wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`); reject `.mp4`.

### Logging and Observability

- Use `tracing` for startup and operational events.
- Prefer structured fields in logs (`host`, `port`, `model`, etc.) rather than free-form strings.
- Avoid logging secrets/tokens/API keys.

### Testing Conventions

- Keep unit tests near the implementation using `#[cfg(test)] mod tests`.
- Name tests as behavior statements (`transcriptions_reject_mp4`).
- Assert HTTP status and JSON error codes for API behavior.
- Include edge-case tests for parsing, bounds, and invalid input.

## Agent Implementation Notes

- Make minimal, targeted changes; do not refactor unrelated code.
- Preserve public API behavior unless explicitly asked to change it.
- When adding config/env vars, update help text and validation together.
- When touching request parsing, keep auth and model validation semantics intact.
- When adding dependencies, prefer small and well-maintained crates and justify the need.

## Quick File Map

- Entry: `src/main.rs`
- API routes and handlers: `src/api.rs`
- Audio decode + extension checks: `src/audio.rs`
- Backend contract: `src/backend/mod.rs`
- Whisper implementation: `src/backend/whisper_rs.rs`
- Config/env/CLI parsing: `src/config.rs`
- Error model + HTTP mapping: `src/error.rs`
- Output formatting helpers: `src/formats.rs`
- Model download/cache logic: `src/model_store.rs`

## Definition of Done for Agent Changes

- Code is formatted (`cargo fmt -- --check` passes).
- Lint is clean (`cargo clippy --all-targets --all-features -- -D warnings` passes or known exceptions documented).
- Relevant tests pass (`cargo test` or targeted tests for changed modules).
- Behavior changes are reflected in `README.md` when user-facing.

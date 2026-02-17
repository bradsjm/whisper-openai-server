# whisper-openai-rust

Rust OpenAI-compatible Whisper server with strict upload validation and no runtime Python or `ffmpeg` binary dependency.

Endpoints:

- `GET /`
- `GET /health`
- `GET /v1`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

## Why this exists

- Ports the HTTP layer from the Python `whisper-openai` project into Rust.
- Fixes semantics during port (model validation, stricter request validation).
- Avoids shelling out to `ffmpeg` by decoding audio in-process with pure Rust (`symphonia`).

## Current backend status

- `whisper-rs` backend (default): fully Rust inference via `whisper.cpp` bindings.

## Requirements

- Rust toolchain (`cargo`)
- A Whisper GGML model file, for example:
  - `~/.cache/whispercpp/models/ggml-small.bin`

You can download a model from the `whisper.cpp` model scripts or any compatible GGML model source.

## Run

```bash
export WHISPER_BACKEND=whisper-rs
export WHISPER_MODEL="$HOME/.cache/whispercpp/models/ggml-small.bin"
export WHISPER_MODEL_ALIAS=whisper-mlx
export HOST=127.0.0.1
export PORT=8000
export API_KEY=local-dev

./run.sh
```

Or directly:

```bash
cargo run --release
```

## Behavior notes

- `model` is validated. Accepted IDs are `whisper-1` and `WHISPER_MODEL_ALIAS`.
- Strict extension allowlist:
  - allowed: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`
  - rejected: `.mp4` (always)
- Extension is authoritative if present; MIME type is not required to match.
- `temperature` must be a finite float in `[0.0, 1.0]`.
- If `API_KEY` is set, all endpoints require `Authorization: Bearer <API_KEY>`.
- `.mp4` is always rejected by design.

## Example

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local-dev" \
  -F file=@/path/to/audio.m4a \
  -F model=whisper-1 \
  -F response_format=verbose_json
```

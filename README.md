# whisper-openai-rust

Rust OpenAI-compatible Whisper server.

Endpoints:

- `GET /`
- `GET /health`
- `GET /v1`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

## Current backend status

- `whisper-rs` backend (default): fully Rust inference via `whisper.cpp` bindings.

## Requirements

- Rust toolchain (`cargo`)
- A Whisper GGML model file, downloaded automatically on first run by default

The server uses `whisper-rs`, which requires `whisper.cpp`-compatible model files (for example `ggml-small.bin`).

## Run

```bash
export WHISPER_BACKEND=whisper-rs
export WHISPER_AUTO_DOWNLOAD=true
export WHISPER_HF_REPO=ggerganov/whisper.cpp
export WHISPER_HF_FILENAME=ggml-small.bin
export WHISPER_CACHE_DIR="$HOME/.cache/whispercpp/models"
# Optional: export HF_TOKEN=hf_xxx
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
- Startup model resolution order:
  - if `WHISPER_MODEL` points to an existing file, use it
  - else if `WHISPER_AUTO_DOWNLOAD=true`, download from Hugging Face to `WHISPER_CACHE_DIR/WHISPER_HF_FILENAME`
  - else fail startup with an actionable error
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

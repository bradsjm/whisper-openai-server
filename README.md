# whisper-openai-server

[![Crates.io](https://img.shields.io/crates/v/whisper-openai-server)](https://crates.io/crates/whisper-openai-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

A high-performance, OpenAI-compatible Whisper API server written in Rust. This server provides a drop-in replacement for OpenAI's audio transcription and translation endpoints, running entirely on your own hardware with no external API calls or dependencies.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints
- **Fully Local**: Complete privacy - no data leaves your infrastructure
- **High Performance**: Built with Rust for maximum efficiency and minimal resource usage
- **Configurable Parallelism**: Support for concurrent inference requests with configurable worker pools
- **Automatic Model Download**: Seamlessly downloads Whisper models from Hugging Face on first run
- **Multiple Audio Formats**: Supports WAV, MP3, M4A, FLAC, OGG, and WebM
- **API Key Authentication**: Optional Bearer token authentication for secure deployment
- **Flexible Configuration**: Configure via environment variables or command-line arguments
- **Health Monitoring**: Built-in health check endpoints for monitoring and orchestration

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### Quick Install (Binary Release)

**macOS (Apple Silicon/arm64)**

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/bradsjm/whisper-openai-server/releases/latest/download/whisper-openai-server-installer.sh | sh
```

Or download the binary directly from the [releases page](https://github.com/bradsjm/whisper-openai-server/releases).

### Build from Source

**Prerequisites:**
- **Rust toolchain**: Install via [rustup](https://rustup.rs/) (version 1.70 or later)
- **C Compiler**: Required for building whisper-rs bindings
  - macOS: `xcode-select --install`
  - Linux: `sudo apt install build-essential` (Debian/Ubuntu) or equivalent
- **Git**: For cloning the repository

```bash
git clone https://github.com/bradsjm/whisper-openai-server.git
cd whisper-openai-server
cargo build --release
```

The compiled binary will be available at `target/release/whisper-openai-server`.

### Using Cargo Install

```bash
cargo install whisper-openai-server
```

## Quick Start

### 1. Start the Server

The easiest way to start the server is using the provided script:

```bash
./run.sh
```

Or directly with cargo:

```bash
cargo run --release
```

### 2. Make Your First Request

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret-key-here" \
  -F file=@/path/to/audio.m4a \
  -F model=whisper-1
```

That's it! The server will automatically download the Whisper model on first run and begin processing your audio files.

## Configuration

The server can be configured using environment variables or command-line arguments. Command-line arguments take precedence over environment variables.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BACKEND` | `whisper-rs` | Inference backend (currently only `whisper-rs` supported) |
| `WHISPER_ACCELERATION` | `metal` | Acceleration mode: `metal` or `none` |
| `WHISPER_AUTO_DOWNLOAD` | `true` | Automatically download model if not found |
| `WHISPER_HF_REPO` | `ggerganov/whisper.cpp` | Hugging Face repository for model downloads |
| `WHISPER_MODEL_SIZE` | `small` | Model preset: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo` (`large` -> `large-v3`, `turbo` -> `large-v3-turbo`) |
| `WHISPER_CACHE_DIR` | `$HOME/.cache/whispercpp/models` | Directory for cached model files |
| `WHISPER_MODEL` | - | Path to specific model file (overrides `WHISPER_MODEL_SIZE`) |
| `WHISPER_MODEL_ALIAS` | `whisper-mlx` | Alternative model ID accepted by the API |
| `WHISPER_PARALLELISM` | `1` | Number of concurrent inference workers (1-8) |
| `HF_TOKEN` | - | Hugging Face authentication token (optional) |
| `HOST` | `127.0.0.1` | Server host address |
| `PORT` | `8000` | Server port |
| `API_KEY` | - | Optional API key for authentication (if unset, no auth required) |

### Command-Line Arguments

```bash
cargo run --release -- --help
```

| Argument | Description |
|----------|-------------|
| `--host <HOST>` | Server host address |
| `--port <PORT>` | Server port |
| `--whisper-backend <BACKEND>` | Inference backend |
| `--acceleration <MODE>` | Acceleration mode: `metal` or `none` |
| `--whisper-model-size <SIZE>` | Model size |
| `--whisper-model <PATH>` | Path to specific model file |
| `--whisper-parallelism <N>` | Number of workers (1-8) |
| `--api-key <KEY>` | API key for authentication |

### Model Sizes

| Model preset | Notes |
|--------------|-------|
| `tiny`, `base`, `small`, `medium` | Multilingual baseline models |
| `tiny.en`, `base.en`, `small.en`, `medium.en` | English-only variants from the same HF repo |
| `large-v1`, `large-v2`, `large-v3` | Explicit large model versions |
| `large-v3-turbo` (`turbo`) | Fast large-v3-turbo preset |

For custom and quantized files (`q5`, `q8`, etc.), use `WHISPER_HF_FILENAME` or provide an explicit local path with `WHISPER_MODEL`.

Acceleration behavior:
- `WHISPER_ACCELERATION=none` (or `--acceleration=none`) forces CPU mode.
- `WHISPER_ACCELERATION=metal` (or `--acceleration=metal`) requires Metal and fails startup if unavailable.
- Default behavior (`metal` not explicitly set) tries Metal first and falls back to CPU if Metal initialization fails.

Example startup logs:

```text
INFO initialized whisper acceleration requested_acceleration=metal effective_acceleration=metal whisper_parallelism=2
INFO starting whisper-openai-server host=127.0.0.1 port=8000 model=/.../ggml-small.bin backend=WhisperRs acceleration=metal whisper_parallelism=2 max_whisper_parallelism=8
```

If fallback is used (default `metal` not explicitly set), `effective_acceleration=none` is logged.

## API Documentation

The server implements OpenAI-compatible endpoints for audio processing.

### Endpoints Overview

- `GET /` - Server information
- `GET /health` - Health check endpoint
- `GET /v1` - API information
- `GET /v1/models` - List available models
- `POST /v1/audio/transcriptions` - Transcribe audio to text
- `POST /v1/audio/translations` - Translate audio to English text

### POST /v1/audio/transcriptions

Transcribes audio files to text in the original language.

**Request:**

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.wav \
  -F model=whisper-1
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | The audio file to transcribe |
| model | String | Yes | Model ID (`whisper-1` or `WHISPER_MODEL_ALIAS`) |
| language | String | No | Language code (e.g., `en`, `es`, `fr`) |
| prompt | String | No | Text to guide the model's style |
| response_format | String | No | Format: `json`, `text`, `srt`, `verbose_json`, `vtt` |
| temperature | Float | No | Sampling temperature (0.0-1.0) |
| timestamp_granularities | Array | No | Granularities: `word` |

Maximum multipart upload size is 25 MiB per request.

**Response (JSON):**

```json
{
  "text": "Hello, this is a transcription.",
  "task": "transcribe",
  "language": "en",
  "duration": 2.5,
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a transcription.",
      "tokens": [50364, 2425, 11, 428, 367, 257, 31495, 13],
      "temperature": 0.0,
      "avg_logprob": -0.2,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.1
    }
  ]
}
```

### POST /v1/audio/translations

Translates audio files to English text.

**Request:**

```bash
curl http://127.0.0.1:8000/v1/audio/translations \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio_spanish.wav \
  -F model=whisper-1
```

**Parameters:** Same as `/transcriptions` endpoint.

**Response:** Same format as `/transcriptions`.

## Examples

### Basic Transcription

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@podcast_episode.mp3 \
  -F model=whisper-1
```

### Transcription with Language Specified

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@french_speech.wav \
  -F model=whisper-1 \
  -F language=fr
```

### Verbose JSON with Timestamps

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@interview.ogg \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F timestamp_granularities[]=word
```

### Translation to English

```bash
curl http://127.0.0.1:8000/v1/audio/translations \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@german_speech.m4a \
  -F model=whisper-1
```

### SRT Subtitle Format

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@video_audio.webm \
  -F model=whisper-1 \
  -F response_format=srt
```

### Using a Custom Model

```bash
# First, set the custom model path
export WHISPER_MODEL=/path/to/custom/ggml-model.bin

# Then use it in your request
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.flac \
  -F model=whisper-1
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response: `{"status":"ok"}`

### List Available Models

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1234567890,
      "owned_by": "whisper-openai-server"
    },
    {
      "id": "whisper-mlx",
      "object": "model",
      "created": 1234567890,
      "owned_by": "whisper-openai-server"
    }
  ]
}
```

## Development

### Building

```bash
cargo build
cargo build --release  # Optimized build
```

### Running Tests

```bash
cargo test
cargo test --release
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy
```

### Project Structure

```
whisper-openai-server/
├── src/
│   ├── main.rs           # Server entry point
│   ├── config.rs         # Configuration management
│   ├── server.rs         # HTTP server setup
│   ├── handlers.rs       # Request handlers
│   └── inference.rs      # Whisper inference logic
├── tests/                # Integration tests
├── run.sh               # Convenience script
└── README.md            # This file
```

## Troubleshooting

### Model Download Issues

**Problem:** Model fails to download automatically.

**Solutions:**
- Check your internet connection
- Verify `WHISPER_CACHE_DIR` is writable
- Try manually downloading the model:
  ```bash
  mkdir -p $HOME/.cache/whispercpp/models
  wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
    -O $HOME/.cache/whispercpp/models/ggml-small.bin
  ```

### Memory Issues

**Problem:** Server runs out of memory with large models.

**Solutions:**
- Reduce `WHISPER_PARALLELISM` to 1
- Use a smaller model size (`base` or `tiny`)
- Increase system swap space
- Reduce `WHISPER_MODEL_SIZE`

### Slow Performance

**Problem:** Transcription takes too long.

**Solutions:**
- Use a smaller model if accuracy requirements allow
- Increase `WHISPER_PARALLELISM` (up to 8) for concurrent requests
- Ensure you're running the release build (`cargo run --release`)
- Check system resource usage (CPU, memory)

### Authentication Errors

**Problem:** Receiving 401 Unauthorized errors.

**Solutions:**
- Ensure `API_KEY` is set on the server
- Include the `Authorization: Bearer <API_KEY>` header in requests
- Check for typos in the API key

### Unsupported File Format

**Problem:** "Unsupported file format" error.

**Solutions:**
- Ensure file extension is one of: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`
- Note that `.mp4` files are explicitly rejected
- Convert your file using FFmpeg:
  ```bash
  ffmpeg -i input.mp4 -acodec libmp3lame output.mp3
  ```

### Port Already in Use

**Problem:** Server fails to start with "address already in use" error.

**Solutions:**
- Change the port: `export PORT=8001`
- Find and kill the process using the port:
  ```bash
  lsof -i :8000
  kill -9 <PID>
  ```

## Behavior Notes

### Model Resolution

The server resolves models in the following order at startup:

1. If `WHISPER_MODEL` points to an existing file, use it directly
2. Otherwise, resolve a default filename from `WHISPER_MODEL_SIZE` (default: `small`)
3. If `WHISPER_AUTO_DOWNLOAD=true`, download from Hugging Face to `WHISPER_CACHE_DIR`
4. If none of the above succeed, fail startup with an actionable error

### Audio File Validation

- **Strict extension allowlist**: Only `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm` are accepted
- **Extension is authoritative**: The file extension determines processing, not the MIME type
- **MP4 rejection**: `.mp4` files are always rejected by design (use container extraction or conversion)
- **Validation happens early**: Invalid files are rejected before processing begins

### Request Validation

- **Model ID validation**: Only `whisper-1` and `WHISPER_MODEL_ALIAS` are accepted
- **Temperature range**: Must be a finite float between 0.0 and 1.0
- **Required parameters**: Both `file` and `model` parameters are mandatory
- **Multipart body limit**: Requests over 25 MiB are rejected before parsing

### Concurrency and Memory

- **Worker isolation**: Each parallelism worker loads its own model context
- **Memory scaling**: Memory usage scales linearly with `WHISPER_PARALLELISM`
- **Request queuing**: Requests exceeding parallelism limit are queued until a worker is free
- **Parallelism limits**: Minimum 1, maximum 8 workers

### Authentication

- **Optional auth**: If `API_KEY` is not set, no authentication is required
- **Bearer token**: When enabled, all endpoints require `Authorization: Bearer <API_KEY>`
- **Consistent validation**: The same API key must be used for all authenticated requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for the Whisper model and API specification
- [ggerganov](https://github.com/ggerganov) for [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- The Rust community for excellent tooling and libraries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/bradsjm/whisper-openai-server).

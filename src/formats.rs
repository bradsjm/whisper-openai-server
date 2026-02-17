//! Helpers for OpenAI-compatible response formatting.

use std::fmt;

use crate::backend::TranscriptSegment;
use crate::error::AppError;

/// Output format accepted by `response_format` in audio endpoints.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResponseFormat {
    /// JSON object with a single `text` field.
    Json,
    /// Raw plain-text transcript body.
    Text,
    /// JSON object with transcript text plus segment timings.
    VerboseJson,
    /// SubRip subtitle format.
    Srt,
    /// WebVTT subtitle format.
    Vtt,
}

impl ResponseFormat {
    /// Parses a `response_format` string used by the HTTP API.
    pub fn parse(raw: &str) -> Result<Self, AppError> {
        match raw.trim() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(AppError::invalid_request(
                format!("invalid response_format={other:?}; expected one of json,text,verbose_json,srt,vtt"),
                Some("response_format"),
                Some("invalid_response_format"),
            )),
        }
    }
}

impl fmt::Display for ResponseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Text => write!(f, "text"),
            Self::VerboseJson => write!(f, "verbose_json"),
            Self::Srt => write!(f, "srt"),
            Self::Vtt => write!(f, "vtt"),
        }
    }
}

/// Normalizes transcript text by collapsing all whitespace runs to one space.
pub fn normalize_text(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Converts transcript segments to SRT subtitle text.
pub fn segments_to_srt(segments: &[TranscriptSegment]) -> String {
    let mut lines = Vec::new();
    for (idx, seg) in segments.iter().enumerate() {
        if seg.text.trim().is_empty() {
            continue;
        }
        lines.push((idx + 1).to_string());
        lines.push(format!(
            "{} --> {}",
            srt_timestamp(seg.start_secs),
            srt_timestamp(seg.end_secs)
        ));
        lines.push(seg.text.trim().to_string());
        lines.push(String::new());
    }

    let out = lines.join("\n");
    if out.is_empty() {
        "\n".to_string()
    } else {
        format!("{}\n", out.trim_end())
    }
}

/// Converts transcript segments to WebVTT subtitle text.
pub fn segments_to_vtt(segments: &[TranscriptSegment]) -> String {
    let mut lines = vec!["WEBVTT".to_string(), String::new()];
    for seg in segments {
        if seg.text.trim().is_empty() {
            continue;
        }
        lines.push(format!(
            "{} --> {}",
            vtt_timestamp(seg.start_secs),
            vtt_timestamp(seg.end_secs)
        ));
        lines.push(seg.text.trim().to_string());
        lines.push(String::new());
    }

    format!("{}\n", lines.join("\n").trim_end())
}

fn srt_timestamp(seconds: f64) -> String {
    let ms = seconds_to_millis(seconds);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let frac = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02},{frac:03}")
}

fn vtt_timestamp(seconds: f64) -> String {
    let ms = seconds_to_millis(seconds);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let frac = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{frac:03}")
}

fn seconds_to_millis(seconds: f64) -> u64 {
    if seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_format_parse() {
        assert!(matches!(
            ResponseFormat::parse("json"),
            Ok(ResponseFormat::Json)
        ));
        assert!(ResponseFormat::parse("nope").is_err());
    }

    #[test]
    fn normalize_collapses_spaces() {
        assert_eq!(
            normalize_text("  hello   world\nagain"),
            "hello world again"
        );
    }
}

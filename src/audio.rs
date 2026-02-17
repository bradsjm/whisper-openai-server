use std::io::{Cursor, ErrorKind};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use crate::error::AppError;

const TARGET_SAMPLE_RATE: u32 = 16_000;

pub const SUPPORTED_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "flac", "ogg", "webm"];

pub fn validate_extension(filename: &str) -> Result<String, AppError> {
    let extension = filename
        .rsplit_once('.')
        .map(|(_, ext)| ext.trim().to_ascii_lowercase())
        .ok_or_else(|| {
            AppError::unsupported_media_type(
                "file must include an extension; accepted extensions: .wav,.mp3,.m4a,.flac,.ogg,.webm",
            )
        })?;

    if extension == "mp4" {
        return Err(AppError::unsupported_media_type(
            "unsupported file extension .mp4; accepted extensions: .wav,.mp3,.m4a,.flac,.ogg,.webm",
        ));
    }

    if !SUPPORTED_EXTENSIONS.iter().any(|ext| *ext == extension) {
        return Err(AppError::unsupported_media_type(format!(
            "unsupported file extension .{extension}; accepted extensions: .wav,.mp3,.m4a,.flac,.ogg,.webm"
        )));
    }

    Ok(extension)
}

pub fn decode_to_mono_16khz_f32(bytes: &[u8], extension_hint: &str) -> Result<Vec<f32>, AppError> {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension_hint);

    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| {
            AppError::unsupported_media_type(format!("failed to open media file: {err}"))
        })?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| AppError::unsupported_media_type("no audio track found in uploaded file"))?;

    if track.codec_params.codec == CODEC_TYPE_NULL {
        return Err(AppError::unsupported_media_type(
            "unsupported codec: missing codec information",
        ));
    }

    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| AppError::unsupported_media_type(format!("unsupported codec: {err}")))?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);
    let track_id = track.id;
    let mut mono = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err)) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err(AppError::unsupported_media_type(
                    "decoder reset required for this media stream",
                ));
            }
            Err(err) => {
                return Err(AppError::unsupported_media_type(format!(
                    "failed while reading media stream: {err}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => {
                return Err(AppError::unsupported_media_type(format!(
                    "failed to decode audio packet: {err}"
                )));
            }
        };

        sample_rate = decoded.spec().rate;
        let channels = decoded.spec().channels.count();

        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);
        let samples = sample_buffer.samples();

        if channels <= 1 {
            mono.extend_from_slice(samples);
            continue;
        }

        for frame in samples.chunks(channels) {
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / channels as f32);
        }
    }

    if mono.is_empty() {
        return Err(AppError::unsupported_media_type(
            "decoded audio is empty after processing",
        ));
    }

    let normalized = mono
        .into_iter()
        .map(|s| s.clamp(-1.0, 1.0))
        .collect::<Vec<_>>();

    Ok(if sample_rate == TARGET_SAMPLE_RATE {
        normalized
    } else {
        resample_linear(&normalized, sample_rate, TARGET_SAMPLE_RATE)
    })
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.len() < 2 {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) * (dst_rate as f64) / (src_rate as f64)).round() as usize;
    let out_len = out_len.max(1);

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;

        let a = input[idx.min(input.len() - 1)];
        let b = input[(idx + 1).min(input.len() - 1)];
        out.push(a + (b - a) * frac);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mp4() {
        assert!(validate_extension("clip.mp4").is_err());
    }

    #[test]
    fn accepts_m4a() {
        assert!(matches!(
            validate_extension("clip.m4a").as_deref(),
            Ok("m4a")
        ));
    }
}

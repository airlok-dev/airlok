//! Preparing an image for a provider: decode, downscale, re-encode, and
//! report what was done. Nothing here touches the terminal or the
//! clipboard, so every limit is testable without either.
//!
//! Images cannot be scanned for secrets, so they never pass through the
//! redactor. What protects the user is the confirmation before sending
//! and the fact that a session file keeps no bytes.

use std::io::Cursor;
use std::path::Path;

use base64::Engine as _;
use image::{DynamicImage, ImageFormat};

use crate::session::fnv1a;

/// Longest edge a provider takes without downscaling for you. Anything
/// larger is shrunk before sending, which costs nothing in quality that
/// the model would have kept.
pub const MAX_EDGE: u32 = 1568;

/// Past this, one image is re-encoded to JPEG rather than sent as is.
pub const REENCODE_OVER: usize = 5 * 1024 * 1024;

/// Most one message's images may total once prepared.
pub const MESSAGE_BUDGET: usize = 20 * 1024 * 1024;

/// An image ready to send, and what was done to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub media_type: String,
    /// Base64, which is the only form a provider takes.
    pub data: String,
    pub width: u32,
    pub height: u32,
    /// Of the encoded bytes, so a session can say which image it was
    /// without keeping it.
    pub hash: String,
    /// Bytes after processing, for the per-message budget.
    pub bytes: usize,
    /// One line per thing done, for the user to see.
    pub notes: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ImageError {
    #[error("{0} is not an image format airlok can send (png, jpeg, gif and webp are)")]
    Unsupported(String),
    #[error("cannot read {path}: {problem}")]
    Read { path: String, problem: String },
    #[error("cannot decode the image: {0}")]
    Decode(String),
    #[error("cannot re-encode the image: {0}")]
    Encode(String),
    #[error("this message's images come to {total}, over the {limit} a message may carry")]
    OverBudget { total: String, limit: String },
}

/// The media type for a path's extension, or `None` when it is not an
/// image airlok sends.
pub fn media_type_for_path(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    })
}

fn format_for(media_type: &str) -> Option<ImageFormat> {
    Some(match media_type {
        "image/png" => ImageFormat::Png,
        "image/jpeg" => ImageFormat::Jpeg,
        "image/gif" => ImageFormat::Gif,
        "image/webp" => ImageFormat::WebP,
        _ => return None,
    })
}

fn media_type_of(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Gif => "image/gif",
        ImageFormat::WebP => "image/webp",
        _ => "image/png",
    }
}

/// Human bytes, for a message a person reads.
pub fn human(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}

fn encode(image: &DynamicImage, format: ImageFormat) -> Result<Vec<u8>, ImageError> {
    let mut out = Cursor::new(Vec::new());
    image
        .write_to(&mut out, format)
        .map_err(|e| ImageError::Encode(e.to_string()))?;
    Ok(out.into_inner())
}

/// Reads an image from disk and prepares it.
pub fn prepare_path(path: &Path) -> Result<Prepared, ImageError> {
    let media_type = media_type_for_path(path).ok_or_else(|| {
        ImageError::Unsupported(
            path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("that file")
                .to_string(),
        )
    })?;
    let bytes = std::fs::read(path).map_err(|e| ImageError::Read {
        path: path.display().to_string(),
        problem: e.to_string(),
    })?;
    prepare(&bytes, media_type)
}

/// Decodes `bytes`, downscales anything past [`MAX_EDGE`], re-encodes
/// anything past [`REENCODE_OVER`], and says what it did.
pub fn prepare(bytes: &[u8], media_type: &str) -> Result<Prepared, ImageError> {
    let format =
        format_for(media_type).ok_or_else(|| ImageError::Unsupported(media_type.to_string()))?;
    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| ImageError::Decode(e.to_string()))?;

    let mut notes = Vec::new();
    let (was_wide, was_high) = (decoded.width(), decoded.height());
    let mut prepared = decoded;
    if was_wide.max(was_high) > MAX_EDGE {
        prepared = prepared.resize(MAX_EDGE, MAX_EDGE, image::imageops::FilterType::Lanczos3);
        notes.push(format!(
            "downscaled {was_wide}x{was_high} to {}x{}",
            prepared.width(),
            prepared.height()
        ));
    }

    // This crate decodes webp but cannot write it, so a webp becomes a png.
    let mut out_format = match format {
        ImageFormat::WebP => ImageFormat::Png,
        other => other,
    };
    if out_format != format {
        notes.push("converted webp to png, which is what can be written".to_string());
    }

    let mut encoded = encode(&prepared, out_format)?;
    if encoded.len() > REENCODE_OVER {
        let before = encoded.len();
        // JPEG carries no alpha, so the conversion has to be explicit.
        let flattened = DynamicImage::ImageRgb8(prepared.to_rgb8());
        let smaller = encode(&flattened, ImageFormat::Jpeg)?;
        notes.push(format!(
            "re-encoded {} to jpeg at {}",
            human(before),
            human(smaller.len())
        ));
        out_format = ImageFormat::Jpeg;
        encoded = smaller;
    }

    Ok(Prepared {
        media_type: media_type_of(out_format).to_string(),
        data: base64::engine::general_purpose::STANDARD.encode(&encoded),
        width: prepared.width(),
        height: prepared.height(),
        hash: format!("{:016x}", fnv1a(&encoded)),
        bytes: encoded.len(),
        notes,
    })
}

/// Refuses a message whose images come to more than [`MESSAGE_BUDGET`].
pub fn within_budget(prepared: &[Prepared]) -> Result<(), ImageError> {
    let total: usize = prepared.iter().map(|p| p.bytes).sum();
    if total > MESSAGE_BUDGET {
        return Err(ImageError::OverBudget {
            total: human(total),
            limit: human(MESSAGE_BUDGET),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(width, height));
        encode(&image, ImageFormat::Png).unwrap()
    }

    #[test]
    fn a_large_image_is_downscaled_and_says_so() {
        let prepared = prepare(&png(3000, 1500), "image/png").unwrap();
        assert_eq!(prepared.width, MAX_EDGE);
        assert_eq!(prepared.height, 784, "the aspect ratio is kept");
        assert!(
            prepared
                .notes
                .iter()
                .any(|n| n.contains("downscaled 3000x1500")),
            "{:?}",
            prepared.notes
        );
    }

    #[test]
    fn a_small_image_is_left_alone() {
        let prepared = prepare(&png(320, 200), "image/png").unwrap();
        assert_eq!((prepared.width, prepared.height), (320, 200));
        assert!(prepared.notes.is_empty(), "{:?}", prepared.notes);
        assert_eq!(prepared.media_type, "image/png");
    }

    #[test]
    fn an_unsupported_format_is_named() {
        let error = prepare(b"not an image", "image/tiff").unwrap_err();
        assert_eq!(error, ImageError::Unsupported("image/tiff".into()));
        assert!(error.to_string().contains("png, jpeg, gif and webp"));
    }

    #[test]
    fn bytes_that_are_not_an_image_report_a_decode_failure() {
        let error = prepare(b"not an image at all", "image/png").unwrap_err();
        assert!(matches!(error, ImageError::Decode(_)), "{error:?}");
    }

    #[test]
    fn an_extension_decides_the_media_type() {
        assert_eq!(
            media_type_for_path(Path::new("/tmp/a.JPG")),
            Some("image/jpeg")
        );
        assert_eq!(
            media_type_for_path(Path::new("/tmp/a.webp")),
            Some("image/webp")
        );
        assert_eq!(media_type_for_path(Path::new("/tmp/notes.txt")), None);
        assert_eq!(media_type_for_path(Path::new("/tmp/noext")), None);
    }

    #[test]
    fn a_message_over_the_budget_is_refused() {
        let one = Prepared {
            media_type: "image/png".into(),
            data: String::new(),
            width: 10,
            height: 10,
            hash: "abc".into(),
            bytes: MESSAGE_BUDGET / 2 + 1,
            notes: Vec::new(),
        };
        assert!(within_budget(std::slice::from_ref(&one)).is_ok());
        let error = within_budget(&[one.clone(), one]).unwrap_err();
        assert!(matches!(error, ImageError::OverBudget { .. }), "{error:?}");
        assert!(error.to_string().contains("20.0 MB"), "{error}");
    }

    #[test]
    fn the_hash_is_of_the_bytes_that_are_sent() {
        let a = prepare(&png(64, 64), "image/png").unwrap();
        let b = prepare(&png(64, 64), "image/png").unwrap();
        let c = prepare(&png(65, 64), "image/png").unwrap();
        assert_eq!(a.hash, b.hash);
        assert_ne!(a.hash, c.hash);
        assert!(!a.data.is_empty());
    }
}

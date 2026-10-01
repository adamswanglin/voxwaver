//! `media://` custom URI scheme for webview `<audio>` playback.
//!
//! WKWebView's media stack (AVFoundation) refuses to play media served over
//! Tauri's `asset://` protocol (the element fails with
//! MEDIA_ERR_SRC_NOT_SUPPORTED) even though `fetch`/`<img>` work fine there.
//! Serving the same bytes from a custom scheme that sets an explicit audio
//! Content-Type and honors Range requests makes both playback and seeking
//! work; Range support is also what WKWebView requires for seeking.
//!
//! Frontend usage: `convertFileSrc(absPath, 'media')`. Only files inside the
//! app data's audio/voices/samples directories are served.

use std::borrow::Cow;
use std::path::PathBuf;

use tauri::http::{header, Request, Response, StatusCode};
use tauri::{Manager, UriSchemeContext};

use crate::state::AppCtx;

pub const SCHEME: &str = "media";

pub fn handle<R: tauri::Runtime>(
    ctx: UriSchemeContext<'_, R>,
    request: Request<Vec<u8>>,
) -> Response<Cow<'static, [u8]>> {
    match serve(&ctx, &request) {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("[media] {e}");
            text_response(StatusCode::FORBIDDEN, e)
        }
    }
}

fn serve<R: tauri::Runtime>(
    ctx: &UriSchemeContext<'_, R>,
    request: &Request<Vec<u8>>,
) -> Result<Response<Cow<'static, [u8]>>, String> {
    let app = ctx.app_handle();
    let root = app.state::<AppCtx>().dirs().root.clone();

    // media://localhost/<percent-encoded absolute path>
    // (http://media.localhost/... on Windows — same path extraction)
    let encoded = request.uri().path().trim_start_matches('/');
    let path = percent_decode(encoded);
    let path = PathBuf::from(&path);
    let canon = path
        .canonicalize()
        .map_err(|e| format!("cannot access '{}': {e}", path.display()))?;
    let allowed = ["audio", "voices", "samples"]
        .iter()
        .any(|sub| canon.starts_with(root.join(sub)));
    if !allowed {
        return Err(format!("path outside media scope: {}", canon.display()));
    }

    let data = std::fs::read(&canon).map_err(|e| format!("read '{}': {e}", canon.display()))?;
    let mime = match canon.extension().and_then(|e| e.to_str()) {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        _ => "application/octet-stream",
    };
    let len = data.len() as u64;

    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, mime)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");

    // Honor a single "bytes=start-end" / "bytes=start-" / "bytes=-suffix" range;
    // WKWebView's media element issues these while playing and seeking.
    let range = request
        .headers()
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("bytes="))
        .and_then(|v| parse_range(v, len));
    let (status, body): (StatusCode, Cow<'static, [u8]>) = match range {
        Some((start, end)) => {
            builder = builder.header(
                header::CONTENT_RANGE,
                format!("bytes {start}-{end}/{len}"),
            );
            (
                StatusCode::PARTIAL_CONTENT,
                Cow::Owned(data[start as usize..=end as usize].to_vec()),
            )
        }
        None => (StatusCode::OK, Cow::Owned(data)),
    };
    builder = builder.header(header::CONTENT_LENGTH, body.len());

    builder
        .status(status)
        .body(body)
        .map_err(|e| format!("build response: {e}"))
}

/// Parse a single-range spec into an inclusive (start, end) against `len`.
fn parse_range(spec: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let (start, end) = match spec.split_once('-') {
        Some(("", suffix)) => (len.saturating_sub(suffix.parse().ok()?), len - 1), // last N bytes
        Some((start, "")) => (start.parse().ok()?, len - 1),
        Some((start, end)) => {
            let end: u64 = end.parse().ok()?;
            (start.parse().ok()?, end.min(len - 1))
        }
        None => return None,
    };
    (start <= end && start < len).then_some((start, end))
}

fn text_response(status: StatusCode, msg: String) -> Response<Cow<'static, [u8]>> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Cow::Owned(msg.into_bytes()))
        .expect("static response parts")
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            let hex = &s[i + 1..i + 3];
            if let Ok(b) = u8::from_str_radix(hex, 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("0-99", 1000), Some((0, 99)));
        assert_eq!(parse_range("500-", 1000), Some((500, 999)));
        assert_eq!(parse_range("-100", 1000), Some((900, 999)));
        assert_eq!(parse_range("0-5000", 1000), Some((0, 999)));
        assert_eq!(parse_range("900-100", 1000), None);
        assert_eq!(parse_range("0-", 0), None);
    }
}

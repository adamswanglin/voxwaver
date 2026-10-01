//! History: list / delete / export / reveal.

use mp3lame_encoder::{Builder, DualPcm, FlushNoGap};
use tauri::{AppHandle, Manager};
use tauri_plugin_opener::OpenerExt;
use voxwaver_core::wavio;

use crate::state::AppCtx;
use crate::store::{self, HistoryEntry};

#[tauri::command]
pub fn list_history(app: AppHandle) -> Vec<HistoryEntry> {
    let ctx = app.state::<AppCtx>();
    store::load_history_in(&ctx.dirs().history_dir(), &ctx.dirs().root)
}

#[tauri::command]
pub fn delete_history(app: AppHandle, ids: Vec<String>) -> Result<(), String> {
    let ctx = app.state::<AppCtx>();
    for id in ids {
        let _ = std::fs::remove_file(ctx.dirs().history_path(&id));
        let _ = std::fs::remove_file(ctx.dirs().audio_path(&id));
    }
    Ok(())
}

/// Export the selected entries into `dest_dir` (chosen by the frontend via
/// the dialog plugin), transcoding the stored WAVs to MP3. Returns the
/// number of exported files.
#[tauri::command]
pub fn export_audio(app: AppHandle, ids: Vec<String>, dest_dir: String) -> Result<usize, String> {
    let ctx = app.state::<AppCtx>();
    let dest = std::path::PathBuf::from(&dest_dir);
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
    let mut n = 0;
    for id in ids {
        let src = ctx.dirs().audio_path(&id);
        if src.is_file() {
            let to = dest.join(format!("{id}.mp3"));
            wav_to_mp3(&src, &to).map_err(|e| format!("{id}: {e:#}"))?;
            n += 1;
        }
    }
    Ok(n)
}

/// Transcode a mono WAV file to MP3 (128 kbps CBR).
fn wav_to_mp3(src: &std::path::Path, dest: &std::path::Path) -> anyhow::Result<()> {
    let (samples, sample_rate) = wavio::read_wav_mono(src)?;
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect();
    let mut encoder = Builder::new()
        .ok_or_else(|| anyhow::anyhow!("init lame failed"))?
        .with_num_channels(1)
        .map_err(|e| anyhow::anyhow!("set mono: {e:?}"))?
        .with_sample_rate(sample_rate)
        .map_err(|e| anyhow::anyhow!("set sample rate: {e:?}"))?
        .with_brate(mp3lame_encoder::Bitrate::Kbps128)
        .map_err(|e| anyhow::anyhow!("set bitrate: {e:?}"))?
        .build()
        .map_err(|e| anyhow::anyhow!("build encoder: {e:?}"))?;
    // the crate encodes into the vec's spare capacity, so keep it large:
    // LAME writes at most 1.25 * samples + 7200 bytes per call
    let mut mp3 = Vec::with_capacity(pcm.len() / 8 + 16384);
    for chunk in pcm.chunks(11520) {
        // DualPcm with left == right: LAME mono mode reads only the left
        // channel; MonoPcm passes a NULL right buffer, which segfaults
        encoder
            .encode_to_vec(DualPcm { left: chunk, right: chunk }, &mut mp3)
            .map_err(|e| anyhow::anyhow!("encode: {e:?}"))?;
    }
    encoder
        .flush_to_vec::<FlushNoGap>(&mut mp3)
        .map_err(|e| anyhow::anyhow!("flush: {e:?}"))?;
    std::fs::write(dest, &mp3)?;
    Ok(())
}

/// Reveal one entry's wav in Finder/Explorer.
#[tauri::command]
pub fn reveal_audio(app: AppHandle, id: String) -> Result<(), String> {
    let ctx = app.state::<AppCtx>();
    let path = ctx.dirs().audio_path(&id);
    if !path.is_file() {
        return Err("音频文件不存在".into());
    }
    app.opener()
        .reveal_item_in_dir(path.to_string_lossy().as_ref())
        .map_err(|e| e.to_string())
}

/// Storage stats for the sidebar pill.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageStats {
    pub audio_bytes: u64,
    pub voices_bytes: u64,
    pub models_bytes: u64,
}

#[tauri::command]
pub fn storage_stats(app: AppHandle) -> StorageStats {
    let ctx = app.state::<AppCtx>();
    fn dir_size(p: &std::path::Path) -> u64 {
        let mut s = 0;
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                s += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
        s
    }
    StorageStats {
        audio_bytes: dir_size(&ctx.dirs().audio_dir()),
        voices_bytes: dir_size(&ctx.dirs().root.join("voices")),
        models_bytes: dir_size(&ctx.dirs().root.join("models")),
    }
}

#[cfg(test)]
mod tests {
    use super::wav_to_mp3;

    #[test]
    fn mp3_transcode_produces_valid_output() {
        let dir = std::env::temp_dir().join("voxwaver-app-test");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("t.wav");
        let mp3 = dir.join("t.mp3");
        // 1s of 440 Hz sine at 24 kHz, the codec's native rate
        let x: Vec<f32> = (0..24000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 24000.0).sin() * 0.5)
            .collect();
        voxwaver_core::wavio::write_wav(&wav, &x, 24000, true).unwrap();
        wav_to_mp3(&wav, &mp3).unwrap();
        let data = std::fs::read(&mp3).unwrap();
        // ~1s at 128 kbps CBR ≈ 16 KB; sanity-check size and a frame sync word
        assert!(data.len() > 8000, "mp3 too small: {} bytes", data.len());
        assert!(
            data.windows(2).any(|w| w[0] == 0xFF && w[1] & 0xE0 == 0xE0),
            "no MPEG frame sync found"
        );
    }
}

//! VoxWeaver Tauri app library.

pub mod cmd;
pub mod engine;
pub mod media_proto;
pub mod models;
pub mod state;
pub mod store;

use state::{AppCtx, AppState, Dirs};
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .register_uri_scheme_protocol(media_proto::SCHEME, media_proto::handle)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_data = app.path().app_data_dir()?;
            let dirs = Arc::new(Dirs::init(&app_data)?);
            let settings = store::load_settings(&dirs.settings_path());
            app.manage(AppCtx { dirs });
            app.manage(AppState::new(settings));

            // Main window: macOS overlay title bar so the web content extends
            // under the traffic lights (the design's 48px custom titlebar).
            let builder = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .inner_size(1240.0, 820.0)
            .min_inner_size(1024.0, 700.0)
            .center()
            .title("VoxWeaver")
            .resizable(true)
            .shadow(true);
            #[cfg(target_os = "macos")]
            let builder = builder
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .traffic_light_position(tauri::LogicalPosition::new(20.0, 24.0));
            builder.build()?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            cmd::tts::generate,
            cmd::tts::cancel_generate,
            cmd::tts::get_engine_status,
            cmd::tts::warmup_engine,
            cmd::voices::list_voices,
            cmd::voices::create_voice,
            cmd::voices::update_voice,
            cmd::voices::delete_voice,
            cmd::voices::resolve_voice_name,
            cmd::voices::save_recorded_sample,
            cmd::history::list_history,
            cmd::history::delete_history,
            cmd::history::export_audio,
            cmd::history::reveal_audio,
            cmd::history::storage_stats,
            cmd::settings::get_settings,
            cmd::settings::set_settings,
            cmd::settings::probe_devices,
            cmd::settings::get_model_status,
            cmd::settings::list_model_status,
            cmd::settings::import_local_model,
            cmd::settings::delete_model,
            cmd::model_dl::download_model,
            cmd::model_dl::cancel_download,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

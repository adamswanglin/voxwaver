//! VoxWeaver desktop app entry point.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
fn main() {
    voxwaver_app_lib::run()
}

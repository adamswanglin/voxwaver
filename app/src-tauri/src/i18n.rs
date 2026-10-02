//! Backend message localization (error strings surfaced in the UI).
//! Locale JSON files live in `src-tauri/i18n/`; the `i18n!` macro is
//! invoked at the crate root (lib.rs) because its generated `t!` macro
//! resolves `_rust_i18n_t` via `crate::`.

/// Apply the UI language to backend translations. Unknown values fall
/// back to the crate fallback locale.
pub fn set_ui_lang(lang: &str) {
    // the legacy value from earlier builds
    let lang = if lang == "zh-CN" { "zh" } else { lang };
    rust_i18n::set_locale(lang);
}

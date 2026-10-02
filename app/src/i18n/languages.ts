/** UI languages; the same set the OmniVoice model accepts as a `lang` tag.
 * Labels stay in each language's own writing (native names). */
export const LANGUAGES = [
  { value: 'zh', label: '简体中文' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: '日本語' },
  { value: 'de', label: 'Deutsch' },
  { value: 'fr', label: 'Français' },
  { value: 'es', label: 'Español' },
  { value: 'ko', label: '한국어' },
  { value: 'ar', label: 'العربية' },
  { value: 'ru', label: 'Русский' },
  { value: 'nl', label: 'Nederlands' },
  { value: 'it', label: 'Italiano' },
  { value: 'pl', label: 'Polski' },
  { value: 'pt', label: 'Português' },
] as const

export type UiLang = (typeof LANGUAGES)[number]['value']

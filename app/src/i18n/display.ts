import i18n from '.'

/** History entries persist the backend-era Chinese sentinels as
 * `voice_name`; translate them at display time, pass user names through. */
export function historyVoiceName(raw: string): string {
  // no reference voice (new entries store ''; legacy ones the old sentinel)
  if (!raw || raw === '默认音色') return '—'
  if (raw === '未知声音') return i18n.t('voices.unknownVoice')
  return raw
}

/** Voice tags: "preset" / "clone" are stable ids; legacy data keeps the
 * Chinese words — translate what we know, show the rest raw. */
export function voiceTagLabel(raw: string): string {
  if (raw === 'preset' || raw === '预置') return i18n.t('voices.preset')
  if (raw === 'clone' || raw === '克隆') return i18n.t('voices.cloneTag')
  return raw
}

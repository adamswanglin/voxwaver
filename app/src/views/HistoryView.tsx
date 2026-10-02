import { useEffect, useState } from 'react'
import { open } from '@tauri-apps/plugin-dialog'
import { useTranslation } from 'react-i18next'
import { apiRevealAudio } from '../api'
import { IconCopy, IconFolder, IconPause, IconPlay, IconSearch, IconTrash } from '../components/Icons'
import { formatBytes, formatDate, formatDuration, isThisWeek, isToday } from '../lib/format'
import { historyVoiceName } from '../i18n/display'
import { useHistory } from '../stores/history'
import { usePlayer } from '../stores/player'
import { toast } from '../stores/toast'
import type { HistoryEntry } from '../types'

type Filter = 'all' | 'today' | 'week'

export function HistoryView() {
  const { t } = useTranslation()
  const [filter, setFilter] = useState<Filter>('all')
  const entries = useHistory((s) => s.entries)
  const selected = useHistory((s) => s.selected)
  const toggle = useHistory((s) => s.toggle)
  const remove = useHistory((s) => s.remove)
  const exportSelected = useHistory((s) => s.exportSelected)
  const clearSelection = useHistory((s) => s.clearSelection)
  const load = useHistory((s) => s.load)
  const search = useHistory((s) => s.search)
  const [query, setQuery] = useState('')

  // debounce keystrokes; empty query falls back to the full list
  useEffect(() => {
    const t = setTimeout(() => void search(query.trim()), 250)
    return () => clearTimeout(t)
  }, [query, search])

  const shown = entries.filter((e) =>
    filter === 'today' ? isToday(e.createdAt) : filter === 'week' ? isThisWeek(e.createdAt) : true,
  )
  const selIds = [...selected].filter((id) => shown.some((e) => e.id === id))

  const onExport = async () => {
    if (!selIds.length) return
    const dir = await open({ directory: true, title: t('common.exportDir') })
    if (!dir) return
    await exportSelected(dir)
  }

  return (
    <>
      <div className="history-header">
        <h2 className="panel-title" style={{ fontSize: 20 }}>
          {t('history.title')}
          <span
            className="panel-subtitle"
            style={{ marginLeft: 10, fontSize: 13 }}
            role="button"
            tabIndex={0}
            onClick={() => void load()}
            onKeyDown={(e) => e.key === 'Enter' && void load()}
          >
            {t('common.refresh')}
          </span>
        </h2>
        <div className="history-toolbar">
          <div className="history-search">
            <IconSearch />
            <input
              value={query}
              placeholder={t('history.searchPlaceholder')}
              aria-label={t('history.searchAria')}
              onChange={(e) => setQuery(e.target.value)}
            />
            {query && (
              <button
                className="history-search-clear"
                aria-label={t('history.clearSearch')}
                onClick={() => setQuery('')}
              >
                ✕
              </button>
            )}
          </div>
          <div className="history-filters">
            <button
              className={`history-filter-btn ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              {t('common.all')}
            </button>
            <button
              className={`history-filter-btn ${filter === 'today' ? 'active' : ''}`}
              onClick={() => setFilter('today')}
            >
              {t('history.today')}
            </button>
            <button
              className={`history-filter-btn ${filter === 'week' ? 'active' : ''}`}
              onClick={() => setFilter('week')}
            >
              {t('history.week')}
            </button>
          </div>
        </div>
      </div>

      {selIds.length > 0 && (
        <div className="batch-bar show">
          {t('history.selectedN', { n: selIds.length })}
          <button className="batch-btn" onClick={onExport}>
            {t('history.exportWav')}
          </button>
          <button
            className="batch-btn"
            onClick={() => {
              void remove(selIds)
            }}
          >
            {t('common.delete')}
          </button>
          <button
            className="batch-btn"
            style={{ marginLeft: 'auto' }}
            onClick={clearSelection}
          >
            {t('history.cancelSel')}
          </button>
        </div>
      )}

      <div className="history-table">
        <div className="table-header">
          <span />
          <span />
          <span>{t('history.colText')}</span>
          <span>{t('history.colVoice')}</span>
          <span>{t('history.colDate')}</span>
          <span>{t('history.colDuration')}</span>
          <span>{t('history.colSize')}</span>
          <span>{t('history.colActions')}</span>
        </div>
        {shown.length === 0 ? (
          <div className="empty-state">{query.trim() ? t('history.noMatch') : t('history.empty')}</div>
        ) : (
          shown.map((e) => (
            <HistoryRow
              key={e.id}
              entry={e}
              checked={selected.has(e.id)}
              onToggle={() => toggle(e.id)}
            />
          ))
        )}
      </div>
    </>
  )
}

function HistoryRow({
  entry,
  checked,
  onToggle,
}: {
  entry: HistoryEntry
  checked: boolean
  onToggle: () => void
}) {
  const { t } = useTranslation()
  const playEntry = usePlayer((s) => s.play)
  const isCurrent = usePlayer((s) => s.entry?.id === entry.id)
  const isPlaying = usePlayer((s) => s.playing)
  const playing = isCurrent && isPlaying

  return (
    <div className="table-row">
      <button
        className={`table-checkbox ${checked ? 'checked' : ''}`}
        role="checkbox"
        aria-checked={checked}
        aria-label={t('history.selectAria')}
        onClick={onToggle}
      >
        {checked && (
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
      </button>
      <button
        className="table-action-btn"
        style={{ width: 28, height: 28, color: playing ? 'var(--seed-primary)' : undefined }}
        aria-label={playing ? t('history.pause') : t('history.play')}
        onClick={() => playEntry(entry)}
      >
        {playing ? <IconPause /> : <IconPlay />}
      </button>
      <span className="text-preview" title={entry.text}>
        {entry.text}
      </span>
      <span className="table-voice">{historyVoiceName(entry.voiceName)}</span>
      <span className="table-voice">{formatDate(entry.createdAt)}</span>
      <span className="table-duration">{formatDuration(entry.durationSec)}</span>
      <span className="table-duration">{formatBytes(entry.fileSize)}</span>
      <span className="table-actions">
        <button
          className="table-action-btn"
          aria-label={t('history.copyText')}
          title={t('history.copyText')}
          onClick={() => {
            navigator.clipboard
              .writeText(entry.text)
              .then(() => toast.success(t('history.copied')))
              .catch((e) => toast.error(String(e)))
          }}
        >
          <IconCopy />
        </button>
        <button
          className="table-action-btn"
          aria-label={t('history.reveal')}
          title={t('history.reveal')}
          onClick={() => {
            apiRevealAudio(entry.id).catch((e) => toast.error(String(e)))
          }}
        >
          <IconFolder />
        </button>
        <button
          className="table-action-btn"
          aria-label={t('common.delete')}
          title={t('common.delete')}
          onClick={() => {
            void useHistory.getState().remove([entry.id])
          }}
        >
          <IconTrash />
        </button>
      </span>
    </div>
  )
}

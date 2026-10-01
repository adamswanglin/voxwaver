import { useState } from 'react'
import { open } from '@tauri-apps/plugin-dialog'
import { apiRevealAudio } from '../api'
import { IconFolder, IconPause, IconPlay, IconTrash } from '../components/Icons'
import { formatBytes, formatDate, formatDuration, isThisWeek, isToday } from '../lib/format'
import { useHistory } from '../stores/history'
import { usePlayer } from '../stores/player'
import { toast } from '../stores/toast'
import type { HistoryEntry } from '../types'

type Filter = 'all' | 'today' | 'week'

export function HistoryView() {
  const [filter, setFilter] = useState<Filter>('all')
  const entries = useHistory((s) => s.entries)
  const selected = useHistory((s) => s.selected)
  const toggle = useHistory((s) => s.toggle)
  const remove = useHistory((s) => s.remove)
  const exportSelected = useHistory((s) => s.exportSelected)
  const clearSelection = useHistory((s) => s.clearSelection)
  const load = useHistory((s) => s.load)

  const shown = entries.filter((e) =>
    filter === 'today' ? isToday(e.createdAt) : filter === 'week' ? isThisWeek(e.createdAt) : true,
  )
  const selIds = [...selected].filter((id) => shown.some((e) => e.id === id))

  const onExport = async () => {
    if (!selIds.length) return
    const dir = await open({ directory: true, title: '选择导出位置' })
    if (!dir) return
    await exportSelected(dir)
  }

  return (
    <>
      <div className="history-header">
        <h2 className="panel-title" style={{ fontSize: 20 }}>
          历史记录
          <span
            className="panel-subtitle"
            style={{ marginLeft: 10, fontSize: 13 }}
            role="button"
            tabIndex={0}
            onClick={() => void load()}
            onKeyDown={(e) => e.key === 'Enter' && void load()}
          >
            刷新
          </span>
        </h2>
        <div className="history-filters">
          <button
            className={`history-filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            全部
          </button>
          <button
            className={`history-filter-btn ${filter === 'today' ? 'active' : ''}`}
            onClick={() => setFilter('today')}
          >
            今天
          </button>
          <button
            className={`history-filter-btn ${filter === 'week' ? 'active' : ''}`}
            onClick={() => setFilter('week')}
          >
            本周
          </button>
        </div>
      </div>

      {selIds.length > 0 && (
        <div className="batch-bar show">
          已选择 {selIds.length} 条
          <button className="batch-btn" onClick={onExport}>
            导出 WAV
          </button>
          <button
            className="batch-btn"
            onClick={() => {
              void remove(selIds)
            }}
          >
            删除
          </button>
          <button
            className="batch-btn"
            style={{ marginLeft: 'auto' }}
            onClick={clearSelection}
          >
            取消选择
          </button>
        </div>
      )}

      <div className="history-table">
        <div className="table-header">
          <span />
          <span />
          <span>文本内容</span>
          <span>声音</span>
          <span>日期</span>
          <span>时长</span>
          <span>大小</span>
          <span>操作</span>
        </div>
        {shown.length === 0 ? (
          <div className="empty-state">暂无生成记录</div>
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
        aria-label="选择"
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
        aria-label={playing ? '暂停' : '播放'}
        onClick={() => playEntry(entry)}
      >
        {playing ? <IconPause /> : <IconPlay />}
      </button>
      <span className="text-preview" title={entry.text}>
        {entry.text}
      </span>
      <span className="table-voice">{entry.voiceName}</span>
      <span className="table-voice">{formatDate(entry.createdAt)}</span>
      <span className="table-duration">{formatDuration(entry.durationSec)}</span>
      <span className="table-duration">{formatBytes(entry.fileSize)}</span>
      <span className="table-actions">
        <button
          className="table-action-btn"
          aria-label="在文件夹中显示"
          title="在文件夹中显示"
          onClick={() => {
            apiRevealAudio(entry.id).catch((e) => toast.error(String(e)))
          }}
        >
          <IconFolder />
        </button>
        <button
          className="table-action-btn"
          aria-label="删除"
          title="删除"
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

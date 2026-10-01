import { useCallback, useEffect, useState } from 'react'
import { apiStorageStats } from '../../api'
import { formatBytes } from '../../lib/format'
import { useHistory } from '../../stores/history'
import { useSettings } from '../../stores/settings'
import { useView } from '../../stores/view'
import { useVoices } from '../../stores/voices'
import { IconClock, IconGear, IconMic, IconWorkspace, Logo } from '../Icons'

const NAV = [
  { view: 'workspace', label: '主工作台', Icon: IconWorkspace },
  { view: 'voices', label: '声音库', Icon: IconMic },
  { view: 'history', label: '历史记录', Icon: IconClock },
] as const

export function Sidebar() {
  const view = useView((s) => s.view)
  const setView = useView((s) => s.setView)
  const setSettingsOpen = useView((s) => s.setSettingsOpen)
  const voices = useVoices((s) => s.voices)
  const historyCount = useHistory((s) => s.entries.length)
  const modelInstalled = useSettings((s) => s.modelStatuses.find((m) => m.active)?.installed)
  const [stats, setStats] = useState<Awaited<ReturnType<typeof apiStorageStats>> | null>(null)

  const refresh = useCallback(() => {
    apiStorageStats().then(setStats).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh, historyCount, modelInstalled])

  const used = stats ? stats.audioBytes + stats.voicesBytes + stats.modelsBytes : 0
  const total = 10 * 1024 ** 3
  const pct = Math.min(100, (used / total) * 100)

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Logo />
        <h1>VoxWeaver</h1>
      </div>

      <nav className="sidebar-nav" aria-label="主导航">
        {NAV.map(({ view: v, label, Icon }) => (
          <button
            key={v}
            className={`nav-item ${view === v ? 'active' : ''}`}
            onClick={() => setView(v)}
            aria-current={view === v ? 'page' : undefined}
          >
            <Icon />
            <span>{label}</span>
            {v === 'voices' && voices.length > 1 && (
              <span className="nav-badge">{voices.length}</span>
            )}
            {v === 'history' && historyCount > 0 && (
              <span className="nav-badge">{historyCount}</span>
            )}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            className="nav-item"
            aria-label="设置"
            style={{ padding: 8, borderRadius: 8, width: 'auto', flexShrink: 0 }}
            onClick={() => setSettingsOpen(true)}
          >
            <IconGear />
          </button>
          <div
            className="storage-pill"
            aria-label="本地存储使用情况"
            style={{ minWidth: 0, fontSize: 11, padding: '5px 8px', gap: 5, margin: 0, flex: 1 }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
              <path d="M22 12H2" />
              <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
            </svg>
            <span style={{ whiteSpace: 'nowrap' }}>{formatBytes(used)}</span>
            <div className="storage-bar">
              <div className="storage-bar-fill" style={{ width: `${pct}%` }} />
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

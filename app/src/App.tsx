import { useEffect } from 'react'
import { CloneModal } from './components/voices/CloneModal'
import { PlayerBar } from './components/player/PlayerBar'
import { SettingsModal } from './components/settings/SettingsModal'
import { Sidebar } from './components/shell/Sidebar'
import { TitleBar } from './components/shell/TitleBar'
import { useHistory } from './stores/history'
import { useSettings, wireDownloadEvents } from './stores/settings'
import { useToast } from './stores/toast'
import { useView } from './stores/view'
import { useVoices } from './stores/voices'
import { HistoryView } from './views/HistoryView'
import { VoicesView } from './views/VoicesView'
import { WorkspaceView } from './views/WorkspaceView'

function ToastHost() {
  const toasts = useToast((s) => s.toasts)
  const dismiss = useToast((s) => s.dismiss)
  return (
    <div className="toast-host">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role="status" onClick={() => dismiss(t.id)}>
          {t.message}
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const view = useView((s) => s.view)

  // initial data load
  useEffect(() => {
    void useSettings.getState().load().catch(() => {})
    void useVoices.getState().load().catch(() => {})
    void useHistory.getState().load().catch(() => {})
    wireDownloadEvents()
  }, [])

  return (
    <div className="app-window">
      <TitleBar />
      <div className="app-body">
        <Sidebar />
        <main className="content">
          <div className={`view ${view === 'workspace' ? 'active' : ''}`}>
            {view === 'workspace' && <WorkspaceView />}
          </div>
          <div className={`view ${view === 'voices' ? 'active' : ''}`}>
            {view === 'voices' && <VoicesView />}
          </div>
          <div className={`view ${view === 'history' ? 'active' : ''}`}>
            {view === 'history' && <HistoryView />}
          </div>
        </main>
      </div>
      <PlayerBar />
      <CloneModal />
      <SettingsModal />
      <ToastHost />
    </div>
  )
}

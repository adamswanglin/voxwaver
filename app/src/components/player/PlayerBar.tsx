import { open } from '@tauri-apps/plugin-dialog'
import { useTranslation } from 'react-i18next'
import { apiExportAudio } from '../../api'
import { formatDuration } from '../../lib/format'
import i18n from '../../i18n'
import { playerSrc, usePlayer } from '../../stores/player'
import { toast } from '../../stores/toast'
import { IconBack10, IconFwd10, IconPause, IconPlay } from '../Icons'
import { Waveform } from './Waveform'

export function PlayerBar() {
  const { t } = useTranslation()
  const entry = usePlayer((s) => s.entry)
  const playing = usePlayer((s) => s.playing)
  const doneNotice = usePlayer((s) => s.doneNotice)
  const position = usePlayer((s) => s.position)
  const duration = usePlayer((s) => s.duration)
  const playPause = usePlayer((s) => s.playPause)
  const skip = usePlayer((s) => s.skip)
  const seek = usePlayer((s) => s.seek)

  const frac = duration > 0 ? position / duration : 0

  const exportCurrent = async () => {
    if (!entry) return
    const dir = await open({ directory: true, title: t('common.exportDir') })
    if (!dir) return
    try {
      const n = await apiExportAudio([entry.id], dir)
      toast.success(i18n.t('player.exported', { n }))
    } catch (e) {
      toast.error(i18n.t('player.exportFail', { msg: String(e) }))
    }
  }

  return (
    <div className="player-bar" data-component="player-bar">
      <div className="player-controls">
        <button className="player-btn" aria-label={t('player.back10')} disabled={!entry} onClick={() => skip(-10)}>
          <IconBack10 />
        </button>
        <button
          className="player-btn primary"
          aria-label={playing ? t('player.pause') : t('player.play')}
          disabled={!entry}
          onClick={playPause}
        >
          {playing ? <IconPause width={18} height={18} /> : <IconPlay width={18} height={18} />}
        </button>
        <button className="player-btn" aria-label={t('player.fwd10')} disabled={!entry} onClick={() => skip(10)}>
          <IconFwd10 />
        </button>
      </div>
      <span className="player-time">
        {formatDuration(position)} / {formatDuration(duration || entry?.durationSec || 0)}
      </span>
      {entry ? (
        <div className="player-track">
          <div className="player-track-text" title={entry.text}>
            {doneNotice && <span className="player-done-flag">{t('player.done')}</span>}
            <span className="player-text-preview">{entry.text}</span>
          </div>
          <Waveform
            key={entry.id}
            url={playerSrc(entry)}
            progress={frac}
            onSeek={seek}
            height={26}
          />
        </div>
      ) : (
        <div className="waveform-container" />
      )}
      <button className="export-btn" aria-label={t('player.exportAria')} disabled={!entry} onClick={exportCurrent}>
        {t('player.exportMp3')}
      </button>
    </div>
  )
}

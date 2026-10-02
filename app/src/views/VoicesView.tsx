import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { VoiceAvatar } from '../components/common/VoiceAvatar'
import { IconEdit, IconPause, IconPlay, IconPlus, IconTrash } from '../components/Icons'
import { formatDate } from '../lib/format'
import { mediaObjectUrl } from '../lib/media'
import i18n from '../i18n'
import { voiceTagLabel } from '../i18n/display'
import { useView } from '../stores/view'
import { useVoices } from '../stores/voices'
import { toast } from '../stores/toast'
import type { VoiceView } from '../types'

export function VoicesView() {
  const { t } = useTranslation()
  const voices = useVoices((s) => s.voices)
  const setCloneOpen = useView((s) => s.setCloneOpen)
  const remove = useVoices((s) => s.remove)
  const openClone = useView((s) => s.openClone)

  return (
    <>
      <div className="voices-header">
        <h2 className="panel-title" style={{ fontSize: 20 }}>
          {t('voices.title')}
        </h2>
        <button className="clone-btn" onClick={() => setCloneOpen(true)}>
          <IconPlus />
          {t('voices.cloneNew')}
        </button>
      </div>

      {voices.length === 0 ? (
        <div className="empty-state" style={{ background: 'var(--seed-surface)', borderRadius: 14 }}>
          {t('voices.empty')}
        </div>
      ) : (
        <div className="voices-grid">
          {voices.map((v) => (
            <VoiceCard
              key={v.id}
              v={v}
              onEdit={() => openClone(v)}
              onDelete={() => remove(v.id)}
            />
          ))}
        </div>
      )}
    </>
  )
}

/** 单例试听：声音卡片直接播放 ref.wav */
let sampleAudio: HTMLAudioElement | null = null
let sampleId: string | null = null

function VoiceCard({
  v,
  onEdit,
  onDelete,
}: {
  v: VoiceView
  onEdit: () => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const [playing, setPlaying] = useState(false)

  const displayName = v.name

  const playSample = () => {
    if (!v.refWav) return
    if (sampleAudio && sampleId === v.id) {
      if (sampleAudio.paused) void sampleAudio.play()
      else sampleAudio.pause()
      return
    }
    sampleAudio?.pause()
    sampleId = v.id
    setPlaying(true)
    // blob: URL, not the media:// URL directly — WKWebView's <audio> can't
    // play custom schemes (see lib/media.ts).
    mediaObjectUrl(v.refWav)
      .then((url) => {
        if (sampleId !== v.id) return // switched to another sample while loading
        sampleAudio = new Audio(url)
        sampleAudio.addEventListener('play', () => setPlaying(true))
        sampleAudio.addEventListener('pause', () => setPlaying(false))
        sampleAudio.addEventListener('ended', () => setPlaying(false))
        sampleAudio.addEventListener('error', () => {
          console.error('sample audio error', sampleAudio?.error?.message, sampleAudio?.src)
          toast.error(i18n.t('voices.samplePlayFail', { msg: sampleAudio?.error?.message ?? v.refWav }))
          setPlaying(false)
        })
        sampleAudio.play().catch((e) => {
          console.error('sample play() rejected', e, sampleAudio?.src)
          setPlaying(false)
        })
      })
      .catch((e) => {
        console.error('sample load failed', e)
        toast.error(i18n.t('voices.samplePlayFail', { msg: String(e) }))
        setPlaying(false)
      })
  }

  return (
    <div className="voice-card">
      {v.isClone && (
        <>
          <button
            className="voice-card-edit"
            aria-label={t('voices.edit', { name: displayName })}
            onClick={onEdit}
          >
            <IconEdit />
          </button>
          <button
            className="voice-card-delete"
            aria-label={t('voices.del', { name: displayName })}
            onClick={onDelete}
          >
            <IconTrash />
          </button>
        </>
      )}
      <div className="voice-card-top">
        <span className="voice-card-avatar">
          <VoiceAvatar name={v.name} icon={v.icon} size={40} />
        </span>
        <div>
          <div className="voice-card-name">{displayName}</div>
        </div>
      </div>
      <div className="voice-card-tags">
        {v.tags.map((tag) => (
          <span key={tag} className={`tag ${v.isClone ? 'cloned' : ''}`}>
            {voiceTagLabel(tag)}
          </span>
        ))}
      </div>
      <div className="voice-card-bottom">
        <button
          className="play-sample"
          onClick={playSample}
          disabled={!v.refWav}
          title={v.refWav ? t('voices.sampleTitle') : t('voices.noSampleHint')}
        >
          {playing ? <IconPause /> : <IconPlay />}
          {playing ? t('voices.pauseSample') : t('voices.playSample')}
        </button>
        <span className="voice-card-date">
          {v.createdAt ? formatDate(v.createdAt) : '—'}
        </span>
      </div>
    </div>
  )
}

import { useState } from 'react'
import { VoiceAvatar } from '../components/common/VoiceAvatar'
import { IconEdit, IconPause, IconPlay, IconPlus, IconTrash } from '../components/Icons'
import { formatDate } from '../lib/format'
import { mediaObjectUrl } from '../lib/media'
import { useView } from '../stores/view'
import { useVoices } from '../stores/voices'
import { toast } from '../stores/toast'
import type { VoiceView } from '../types'

type Filter = 'all' | 'preset' | 'clone'

export function VoicesView() {
  const [filter, setFilter] = useState<Filter>('all')
  const voices = useVoices((s) => s.voices)
  const setCloneOpen = useView((s) => s.setCloneOpen)
  const remove = useVoices((s) => s.remove)

  const shown = voices.filter((v) =>
    filter === 'all' ? true : filter === 'clone' ? v.isClone : !v.isClone,
  )
  const openClone = useView((s) => s.openClone)

  return (
    <>
      <div className="voices-header">
        <h2 className="panel-title" style={{ fontSize: 20 }}>
          声音库
        </h2>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <div className="filter-tabs">
            <button
              className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              全部
            </button>
            <button
              className={`filter-tab ${filter === 'clone' ? 'active' : ''}`}
              onClick={() => setFilter('clone')}
            >
              克隆
            </button>
            <button
              className={`filter-tab ${filter === 'preset' ? 'active' : ''}`}
              onClick={() => setFilter('preset')}
            >
              预置
            </button>
          </div>
          <button className="clone-btn" onClick={() => setCloneOpen(true)}>
            <IconPlus />
            克隆新声音
          </button>
        </div>
      </div>

      {shown.length === 0 ? (
        <div className="empty-state" style={{ background: 'var(--seed-surface)', borderRadius: 14 }}>
          暂无声音。点击「克隆新声音」上传一段 5–10 秒的参考音频。
        </div>
      ) : (
        <div className="voices-grid">
          {shown.map((v) => (
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
  const [playing, setPlaying] = useState(false)

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
          toast.error(`样本播放失败：${sampleAudio?.error?.message ?? v.refWav}`)
          setPlaying(false)
        })
        sampleAudio.play().catch((e) => {
          console.error('sample play() rejected', e, sampleAudio?.src)
          setPlaying(false)
        })
      })
      .catch((e) => {
        console.error('sample load failed', e)
        toast.error(`样本播放失败：${e}`)
        setPlaying(false)
      })
  }

  return (
    <div className="voice-card">
      {v.isClone && (
        <>
          <button
            className="voice-card-edit"
            aria-label={`编辑 ${v.name}`}
            onClick={onEdit}
          >
            <IconEdit />
          </button>
          <button
            className="voice-card-delete"
            aria-label={`删除 ${v.name}`}
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
          <div className="voice-card-name">{v.name}</div>
          <div className="voice-card-lang">
            {v.language} · {v.gender}
            {v.age && v.age !== '—' ? ` · ${v.age}` : ''}
          </div>
        </div>
      </div>
      <div className="voice-card-tags">
        {v.tags.map((t) => (
          <span key={t} className={`tag ${v.isClone ? 'cloned' : ''}`}>
            {t}
          </span>
        ))}
      </div>
      <div className="voice-card-bottom">
        <button
          className="play-sample"
          onClick={playSample}
          disabled={!v.refWav}
          title={v.refWav ? '试听参考样本' : '预置音色无参考样本'}
        >
          {playing ? <IconPause /> : <IconPlay />}
          {playing ? '暂停' : '试听'}
        </button>
        <span className="voice-card-date">
          {v.createdAt ? formatDate(v.createdAt) : '—'}
        </span>
      </div>
    </div>
  )
}

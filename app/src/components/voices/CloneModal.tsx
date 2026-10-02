import { useEffect, useRef, useState } from 'react'
import { convertFileSrc } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/plugin-dialog'
import { useTranslation } from 'react-i18next'
import { IconClose, IconMic, IconUpload } from '../Icons'
import { VoiceAvatar } from '../common/VoiceAvatar'
import { apiSaveSampleAudio } from '../../api'
import { onVoiceEncoding } from '../../api/events'
import i18n from '../../i18n'
import { useView } from '../../stores/view'
import { useVoices } from '../../stores/voices'
import { toast } from '../../stores/toast'

interface CloneForm {
  name: string
  /** emoji icon; '' = name-letter fallback */
  icon: string
  transcript: string
}

const ICON_CHOICES = [
  '😀', '😊', '😎', '🤓', '🥰', '🤔',
  '👩', '👨', '🧑', '👧', '👦', '🧓',
  '🦊', '🐱', '🐶', '🐼', '🦁', '🐸',
  '⭐', '🔥', '🌈', '🎵', '🎙️', '🎧',
  '📢', '🎬', '🚀', '💡', '🌸', '🍀',
]

/** Decode any browser-supported audio to mono f32 samples. */
async function decodeToMono(blob: Blob): Promise<{ samples: number[]; sampleRate: number; duration: number }> {
  const buf = await blob.arrayBuffer()
  const ctx = new AudioContext()
  try {
    const audio = await ctx.decodeAudioData(buf)
    const left = audio.getChannelData(0)
    let samples: Float32Array
    if (audio.numberOfChannels >= 2) {
      const right = audio.getChannelData(1)
      samples = new Float32Array(left.length)
      for (let i = 0; i < left.length; i++) samples[i] = (left[i] + right[i]) / 2
    } else {
      samples = left
    }
    return { samples: Array.from(samples), sampleRate: audio.sampleRate, duration: audio.duration }
  } finally {
    void ctx.close()
  }
}

export function CloneModal() {
  const { t } = useTranslation()
  const openState = useView((s) => s.cloneOpen)
  const setOpen = useView((s) => s.setCloneOpen)
  const editing = useView((s) => s.editingVoice)
  const create = useVoices((s) => s.create)
  const update = useVoices((s) => s.update)
  const [step, setStep] = useState(0)
  const [form, setForm] = useState<CloneForm>({
    name: '',
    icon: '',
    transcript: '',
  })
  const [samplePath, setSamplePath] = useState<string | null>(null)
  /** Display label (original mp3 name / recording), falls back to file name. */
  const [sampleLabel, setSampleLabel] = useState<string | null>(null)
  const [sampleDuration, setSampleDuration] = useState<number | null>(null)
  const [encoding, setEncoding] = useState(false)
  const [encodeLog, setEncodeLog] = useState('')
  const logRef = useRef<HTMLDivElement>(null)
  const [recording, setRecording] = useState(false)
  const [recordSecs, setRecordSecs] = useState(0)
  const [converting, setConverting] = useState(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const timerRef = useRef<number | null>(null)

  const isEdit = !!editing

  // reset everything when reopened; prefill when opened for editing
  useEffect(() => {
    if (openState) {
      setStep(0)
      setEncoding(false)
      setEncodeLog('')
      setSamplePath(null)
      setSampleLabel(null)
      if (editing) {
        setForm({
          name: editing.name,
          icon: editing.icon ?? '',
          transcript: editing.transcript ?? '',
        })
        setSampleDuration(editing.sampleSeconds)
      } else {
        setForm({ name: '', icon: '', transcript: '' })
        setSampleDuration(null)
      }
    }
  }, [openState, editing])

  useEffect(() => {
    const p = onVoiceEncoding((msg) => setEncodeLog(msg))
    return () => {
      void p.then((un) => un())
    }
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: 1e6 })
  }, [encodeLog])

  if (!openState) return null

  const busy = encoding || recording || converting

  const close = () => {
    if (busy) return
    setOpen(false)
  }

  const applyPcmSample = async (blob: Blob, label: string) => {
    setConverting(true)
    try {
      const audio = await decodeToMono(blob)
      const path = await apiSaveSampleAudio(audio.sampleRate, audio.samples)
      setSamplePath(path)
      setSampleLabel(label)
      setSampleDuration(audio.duration)
    } catch (e) {
      toast.error(i18n.t('clone.errAudio', { msg: String(e) }))
    } finally {
      setConverting(false)
    }
  }

  const pickSample = async () => {
    const picked = await open({
      multiple: false,
      title: t('clone.pickTitle'),
      filters: [{ name: t('clone.audioFiles'), extensions: ['wav', 'mp3'] }],
    })
    if (!picked) return
    setSampleDuration(null)
    if (picked.toLowerCase().endsWith('.wav')) {
      // WAV is read directly by the backend
      setSamplePath(picked)
      setSampleLabel(null)
      // 试解码出时长（仅用于展示）
      try {
        const res = await fetch(convertFileSrc(picked))
        const buf = await res.arrayBuffer()
        const ctx = new AudioContext()
        const audio = await ctx.decodeAudioData(buf)
        setSampleDuration(audio.duration)
        void ctx.close()
      } catch {
        /* asset scope 之外解码失败也没关系，时长仅用于展示 */
      }
    } else {
      // MP3 等格式：webview 解码为 PCM 后存为 WAV
      try {
        const res = await fetch(convertFileSrc(picked))
        await applyPcmSample(await res.blob(), fileName(picked))
      } catch {
        toast.error(t('clone.errMp3'))
      }
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      const chunks: Blob[] = []
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }
      rec.onstop = () => {
        stream.getTracks().forEach((tk) => tk.stop())
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
        setRecording(false)
        void applyPcmSample(new Blob(chunks), t('clone.recordSampleLabel'))
      }
      recorderRef.current = rec
      rec.start()
      setRecording(true)
      setRecordSecs(0)
      timerRef.current = window.setInterval(() => setRecordSecs((s) => s + 1), 1000)
    } catch (e) {
      toast.error(i18n.t('clone.errMic', { msg: String(e) }))
    }
  }

  const stopRecording = () => {
    recorderRef.current?.stop()
    recorderRef.current = null
  }

  const sampleDisplay = sampleLabel
    ?? (samplePath ? fileName(samplePath) : isEdit ? t('clone.currentSampleKeep') : '')
  /** edit mode without a newly picked sample keeps the existing reference */
  const keepExisting = isEdit && !samplePath

  const canNext =
    step === 0
      ? form.name.trim().length > 0
      : step === 1
        ? (!!samplePath || keepExisting) && !converting && !recording
        : form.transcript.trim().length > 0

  const submit = async () => {
    if (isEdit) {
      setEncoding(true)
      const ok = await update({
        id: editing.id,
        name: form.name,
        icon: form.icon || null,
        transcript: form.transcript,
        samplePath,
      })
      setEncoding(false)
      if (ok) setOpen(false)
      return
    }
    if (!samplePath) return
    setEncoding(true)
    setEncodeLog('')
    const ok = await create({
      name: form.name,
      icon: form.icon || null,
      samplePath,
      transcript: form.transcript,
    })
    setEncoding(false)
    if (ok) setOpen(false)
  }

  return (
    <div className="modal-overlay show" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={isEdit ? t('clone.titleEdit') : t('clone.titleNew')}>
        <div className="modal-header">
          <div className="modal-title">{isEdit ? t('clone.titleEdit') : t('clone.titleNew')}</div>
          <button className="modal-close" aria-label={t('common.close')} onClick={close}>
            <IconClose />
          </button>
        </div>

        <div className="modal-steps">
          {[t('clone.stepDefine'), t('clone.stepUpload'), t('clone.stepConfirm')].map((label, i) => (
            <div key={label} style={{ display: 'contents' }}>
              {i > 0 && <div className="step-line" style={{ maxWidth: 40 }} />}
              <div className={`step-indicator ${i === step ? 'active' : i < step ? 'done' : ''}`}>
                <span className="step-dot">{i < step ? '✓' : i + 1}</span>
                {label}
              </div>
            </div>
          ))}
        </div>

        <div className="modal-body">
          {step === 0 && (
            <>
              <div className="form-group">
                <label className="form-label">{t('clone.name')}</label>
                <input
                  className="form-input"
                  placeholder={t('clone.namePlaceholder')}
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">{t('clone.icon')}</label>
                <div className="emoji-picker">
                  <button
                    type="button"
                    className={`emoji-choice ${form.icon === '' ? 'active' : ''}`}
                    title={t('clone.iconNone')}
                    onClick={() => setForm((f) => ({ ...f, icon: '' }))}
                  >
                    <VoiceAvatar name={form.name} size={26} />
                  </button>
                  {ICON_CHOICES.map((e) => (
                    <button
                      type="button"
                      key={e}
                      className={`emoji-choice ${form.icon === e ? 'active' : ''}`}
                      onClick={() => setForm((f) => ({ ...f, icon: e }))}
                    >
                      {e}
                    </button>
                  ))}
                </div>
                <div className="slider-hint" style={{ marginTop: 6 }}>
                  {t('clone.iconHint')}
                </div>
              </div>
              <div className="slider-hint">{t('clone.zeroShotHint')}</div>
            </>
          )}

          {step === 1 && (
            <>
              <div className="form-group">
                <label className="form-label">{t('clone.refAudio')}</label>

                {recording ? (
                  <div className="upload-zone" style={{ cursor: 'default' }}>
                    <div className="upload-zone-icon">
                      <IconMic width={26} height={26} />
                    </div>
                    <div className="upload-zone-text" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: '50%',
                          background: 'var(--danger)',
                          display: 'inline-block',
                          animation: 'fadeIn 1s infinite alternate',
                        }}
                      />
                      {t('clone.recording', { time: fmtSecs(recordSecs) })}
                    </div>
                    <div className="upload-zone-hint">{t('clone.recordHint')}</div>
                    <button className="btn-primary" style={{ marginTop: 14 }} onClick={stopRecording}>
                      {t('clone.stopRecording')}
                    </button>
                  </div>
                ) : converting ? (
                  <div className="upload-zone" style={{ cursor: 'default' }}>
                    <div className="upload-zone-text">{t('clone.processing')}</div>
                  </div>
                ) : samplePath ? (
                  <div
                    className={`upload-zone has-file ${busy ? '' : 'clickable'}`}
                    onClick={() => !busy && void pickSample()}
                  >
                    <div className="upload-zone-icon">
                      <IconUpload width={26} height={26} />
                    </div>
                    <div className="upload-zone-text">{sampleDisplay}</div>
                    <div className="upload-zone-hint">
                      {converting
                        ? t('clone.processing')
                        : sampleDuration != null
                          ? `${t('clone.durationSec', { sec: sampleDuration.toFixed(1) })} · ${t('clone.repick')}`
                          : t('clone.repick')}
                    </div>
                    {!busy && (
                      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 12 }}>
                        <button
                          className="btn-secondary"
                          onClick={(e) => {
                            e.stopPropagation()
                            void startRecording()
                          }}
                        >
                          {t('clone.rerecord')}
                        </button>
                        {isEdit && (
                          <button
                            className="btn-secondary"
                            onClick={(e) => {
                              e.stopPropagation()
                              setSamplePath(null)
                              setSampleLabel(null)
                              setSampleDuration(editing.sampleSeconds)
                            }}
                          >
                            {t('clone.removeNewSample')}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ) : keepExisting ? (
                  <div style={{ display: 'grid', gap: 10 }}>
                    <div className="upload-zone has-file" style={{ cursor: 'default' }}>
                      <div className="upload-zone-icon">
                        <IconUpload width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">{t('clone.currentSample')}</div>
                      <div className="upload-zone-hint">
                        {sampleDuration != null ? `${t('clone.durationSec', { sec: sampleDuration.toFixed(1) })} · ` : ''}
                        {t('clone.keepUnchanged')}
                      </div>
                    </div>
                    <div className="upload-zone" onClick={() => void pickSample()}>
                      <div className="upload-zone-icon">
                        <IconUpload width={20} height={20} />
                      </div>
                      <div className="upload-zone-text">{t('clone.uploadNew')}</div>
                    </div>
                    <div className="upload-zone" onClick={() => void startRecording()}>
                      <div className="upload-zone-icon">
                        <IconMic width={20} height={20} />
                      </div>
                      <div className="upload-zone-text">{t('clone.rerecordReplace')}</div>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: 10 }}>
                    <div className="upload-zone" onClick={() => void pickSample()}>
                      <div className="upload-zone-icon">
                        <IconUpload width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">{t('clone.uploadFile')}</div>
                      <div className="upload-zone-hint">
                        {t('clone.uploadHint')}
                      </div>
                    </div>
                    <div className="upload-zone" onClick={() => void startRecording()}>
                      <div className="upload-zone-icon">
                        <IconMic width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">{t('clone.recordNow')}</div>
                      <div className="upload-zone-hint">{t('clone.recordNowHint')}</div>
                    </div>
                  </div>
                )}
              </div>
              <div className="slider-hint">
                {t('clone.cacheHint')}
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div className="form-group">
                <label className="form-label">{t('clone.transcript')}</label>
                <textarea
                  className="form-input"
                  style={{ resize: 'vertical', minHeight: 96, lineHeight: 1.7 }}
                  placeholder={t('clone.transcriptPlaceholder')}
                  value={form.transcript}
                  onChange={(e) => setForm((f) => ({ ...f, transcript: e.target.value }))}
                  disabled={encoding}
                />
              </div>
              <div className="model-dir-row" style={{ marginBottom: 14 }}>
                <span>
                  {form.name} · {sampleDisplay}
                  {sampleDuration != null ? ` · ${sampleDuration.toFixed(1)}s` : ''}
                </span>
              </div>
              {encoding && (
                <div className="dl-progress">
                  <div className="dl-file-label">
                    <span>
                      {keepExisting ? t('clone.savingChanges') : t('clone.encodingRef')}
                    </span>
                  </div>
                  {!keepExisting && (
                    <div className="dl-track">
                      <div className="dl-fill" style={{ width: '40%', animation: 'fadeIn 1s infinite alternate' }} />
                    </div>
                  )}
                  <div className="slider-hint" ref={logRef} style={{ maxHeight: 60, overflow: 'auto', marginTop: 6 }}>
                    {encodeLog}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="modal-footer">
          {step > 0 && !busy && (
            <button className="btn-secondary" onClick={() => setStep((s) => s - 1)}>
              {t('clone.prev')}
            </button>
          )}
          {step < 2 ? (
            <button className="btn-primary" disabled={!canNext} onClick={() => setStep((s) => s + 1)}>
              {t('clone.next')}
            </button>
          ) : (
            <button className="btn-primary" disabled={!canNext || encoding} onClick={() => void submit()}>
              {encoding ? (keepExisting ? t('clone.saving') : t('clone.encoding')) : isEdit ? t('clone.save') : t('clone.start')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

function fmtSecs(s: number): string {
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
}

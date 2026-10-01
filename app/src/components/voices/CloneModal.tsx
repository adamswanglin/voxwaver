import { useEffect, useRef, useState } from 'react'
import { convertFileSrc } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/plugin-dialog'
import { IconClose, IconMic, IconUpload } from '../Icons'
import { apiSaveSampleAudio } from '../../api'
import { onVoiceEncoding } from '../../api/events'
import { useView } from '../../stores/view'
import { useVoices } from '../../stores/voices'
import { toast } from '../../stores/toast'

interface CloneForm {
  name: string
  gender: string
  age: string
  style: string
  language: string
  transcript: string
}

const STEPS = ['定义人物', '上传样本', '确认与编码']

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
  const openState = useView((s) => s.cloneOpen)
  const setOpen = useView((s) => s.setCloneOpen)
  const editing = useView((s) => s.editingVoice)
  const create = useVoices((s) => s.create)
  const update = useVoices((s) => s.update)
  const [step, setStep] = useState(0)
  const [form, setForm] = useState<CloneForm>({
    name: '',
    gender: '女',
    age: '青年',
    style: '',
    language: '中文',
    transcript: '',
  })
  const [samplePath, setSamplePath] = useState<string | null>(null)
  /** Display label (original mp3 name / "录音"), falls back to file name. */
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
          gender: editing.gender,
          age: editing.age === '—' ? '青年' : editing.age,
          style: editing.style,
          language: editing.language,
          transcript: editing.transcript ?? '',
        })
        setSampleDuration(editing.sampleSeconds)
      } else {
        setForm({ name: '', gender: '女', age: '青年', style: '', language: '中文', transcript: '' })
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
      toast.error(`音频处理失败：${String(e)}`)
    } finally {
      setConverting(false)
    }
  }

  const pickSample = async () => {
    const picked = await open({
      multiple: false,
      title: '选择参考音频样本',
      filters: [{ name: '音频文件', extensions: ['wav', 'mp3'] }],
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
        toast.error('无法读取该 MP3 文件，请换一个文件或转换为 WAV')
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
        stream.getTracks().forEach((t) => t.stop())
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
        setRecording(false)
        void applyPcmSample(new Blob(chunks), '录音样本')
      }
      recorderRef.current = rec
      rec.start()
      setRecording(true)
      setRecordSecs(0)
      timerRef.current = window.setInterval(() => setRecordSecs((s) => s + 1), 1000)
    } catch (e) {
      toast.error(`无法访问麦克风：${String(e)}`)
    }
  }

  const stopRecording = () => {
    recorderRef.current?.stop()
    recorderRef.current = null
  }

  const sampleDisplay = sampleLabel
    ?? (samplePath ? fileName(samplePath) : isEdit ? '当前参考样本（保持不变）' : '')
  /** edit mode without a newly picked sample keeps the existing reference */
  const keepExisting = isEdit && !samplePath

  const canNext =
    step === 0
      ? form.name.trim().length > 0
      : step === 1
        ? (!!samplePath || keepExisting) && !converting
        : form.transcript.trim().length > 0

  const submit = async () => {
    if (isEdit) {
      setEncoding(true)
      const ok = await update({
        id: editing.id,
        name: form.name,
        gender: form.gender,
        age: form.age,
        style: form.style,
        language: form.language,
        transcript: form.transcript,
        samplePath,
      })
      setEncoding(false)
      if (ok) setOpen(false)
      return
    }
    if (!samplePath) return
    setEncoding(true)
    setEncodeLog('准备编码器…')
    const ok = await create({
      name: form.name,
      gender: form.gender,
      age: form.age,
      style: form.style,
      language: form.language,
      samplePath,
      transcript: form.transcript,
    })
    setEncoding(false)
    if (ok) setOpen(false)
  }

  return (
    <div className="modal-overlay show" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={isEdit ? '编辑声音' : '克隆新声音'}>
        <div className="modal-header">
          <div className="modal-title">{isEdit ? '编辑声音' : '克隆新声音'}</div>
          <button className="modal-close" aria-label="关闭" onClick={close}>
            <IconClose />
          </button>
        </div>

        <div className="modal-steps">
          {STEPS.map((label, i) => (
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
                <label className="form-label">声音名称 *</label>
                <input
                  className="form-input"
                  placeholder="例如：产品讲解员小雅"
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                />
              </div>
              <div className="form-group" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                <div>
                  <label className="form-label">性别</label>
                  <select
                    className="form-input form-select"
                    value={form.gender}
                    onChange={(e) => setForm((f) => ({ ...f, gender: e.target.value }))}
                  >
                    <option>女</option>
                    <option>男</option>
                    <option>中性</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">年龄</label>
                  <select
                    className="form-input form-select"
                    value={form.age}
                    onChange={(e) => setForm((f) => ({ ...f, age: e.target.value }))}
                  >
                    <option>少年</option>
                    <option>青年</option>
                    <option>中年</option>
                    <option>老年</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">语言</label>
                  <select
                    className="form-input form-select"
                    value={form.language}
                    onChange={(e) => setForm((f) => ({ ...f, language: e.target.value }))}
                  >
                    <option>中文</option>
                    <option>英文</option>
                    <option>中英混合</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">风格描述（可选）</label>
                <input
                  className="form-input"
                  placeholder="例如：温柔、专业播音、纪录片旁白"
                  value={form.style}
                  onChange={(e) => setForm((f) => ({ ...f, style: e.target.value }))}
                />
              </div>
              <div className="slider-hint">zero-shot 克隆无需训练，仅需一段参考音频。</div>
            </>
          )}

          {step === 1 && (
            <>
              <div className="form-group">
                <label className="form-label">参考音频 *</label>

                {samplePath ? (
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
                        ? '正在处理音频…'
                        : sampleDuration != null
                          ? `时长 ${sampleDuration.toFixed(1)} 秒 · 点击重新选择`
                          : '点击重新选择'}
                    </div>
                    {isEdit && !busy && (
                      <button
                        className="btn-secondary"
                        style={{ marginTop: 12 }}
                        onClick={(e) => {
                          e.stopPropagation()
                          setSamplePath(null)
                          setSampleLabel(null)
                          setSampleDuration(editing.sampleSeconds)
                        }}
                      >
                        移除新样本，保留原样本
                      </button>
                    )}
                  </div>
                ) : keepExisting ? (
                  <div style={{ display: 'grid', gap: 10 }}>
                    <div className="upload-zone has-file" style={{ cursor: 'default' }}>
                      <div className="upload-zone-icon">
                        <IconUpload width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">当前参考样本</div>
                      <div className="upload-zone-hint">
                        {sampleDuration != null ? `时长 ${sampleDuration.toFixed(1)} 秒` : ''}
                        保存时保持不变
                      </div>
                    </div>
                    <div className="upload-zone" onClick={() => void pickSample()}>
                      <div className="upload-zone-icon">
                        <IconUpload width={20} height={20} />
                      </div>
                      <div className="upload-zone-text">上传新音频替换</div>
                    </div>
                    <div className="upload-zone" onClick={() => void startRecording()}>
                      <div className="upload-zone-icon">
                        <IconMic width={20} height={20} />
                      </div>
                      <div className="upload-zone-text">重新录音替换</div>
                    </div>
                  </div>
                ) : recording ? (
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
                      录音中 {fmtSecs(recordSecs)}
                    </div>
                    <div className="upload-zone-hint">请朗读 5–10 秒清晰的单人语音</div>
                    <button className="btn-primary" style={{ marginTop: 14 }} onClick={stopRecording}>
                      停止录音
                    </button>
                  </div>
                ) : converting ? (
                  <div className="upload-zone" style={{ cursor: 'default' }}>
                    <div className="upload-zone-text">正在处理音频…</div>
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: 10 }}>
                    <div className="upload-zone" onClick={() => void pickSample()}>
                      <div className="upload-zone-icon">
                        <IconUpload width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">上传音频文件</div>
                      <div className="upload-zone-hint">
                        支持 WAV / MP3 · 建议 5–10 秒清晰单人语音（会自动重采样到 44.1kHz）
                      </div>
                    </div>
                    <div className="upload-zone" onClick={() => void startRecording()}>
                      <div className="upload-zone-icon">
                        <IconMic width={26} height={26} />
                      </div>
                      <div className="upload-zone-text">直接录音</div>
                      <div className="upload-zone-hint">使用麦克风现场录制参考样本</div>
                    </div>
                  </div>
                )}
              </div>
              <div className="slider-hint">
                参考音频会被编码为音色 token 缓存，之后每次生成直接复用，无需重新编码。
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div className="form-group">
                <label className="form-label">样本转写 *</label>
                <textarea
                  className="form-input"
                  style={{ resize: 'vertical', minHeight: 96, lineHeight: 1.7 }}
                  placeholder="逐字写下参考音频里说的话（与音频内容一致可以提升克隆相似度）"
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
                      {keepExisting
                        ? '正在保存修改…'
                        : '正在编码参考音频（zero-shot，无训练）'}
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
              上一步
            </button>
          )}
          {step < 2 ? (
            <button className="btn-primary" disabled={!canNext} onClick={() => setStep((s) => s + 1)}>
              下一步
            </button>
          ) : (
            <button className="btn-primary" disabled={!canNext || encoding} onClick={() => void submit()}>
              {encoding ? (keepExisting ? '保存中…' : '编码中…') : isEdit ? '保存修改' : '开始编码'}
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

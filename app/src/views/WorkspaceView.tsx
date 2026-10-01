import { useRef, useState } from 'react'
import { toast } from '../stores/toast'
import {
  IconChevron,
  IconPlay,
  IconUpload,
} from '../components/Icons'
import { Slider } from '../components/common/Slider'
import { VoiceAvatar } from '../components/common/VoiceAvatar'
import { useGeneration } from '../stores/generation'
import { useSettings } from '../stores/settings'
import { useView } from '../stores/view'
import { useVoices } from '../stores/voices'
import type { VoiceView } from '../types'

export function WorkspaceView() {
  const [text, setText] = useState('')
  const [instruct, setInstruct] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const running = useGeneration((s) => s.running)
  const cancelling = useGeneration((s) => s.cancelling)
  const run = useGeneration((s) => s.run)
  const cancel = useGeneration((s) => s.cancel)
  const fraction = useGeneration((s) => s.fraction)
  const segments = useGeneration((s) => s.segments)
  const stageText = useGeneration((s) => s.stageText)
  const subText = useGeneration((s) => s.subText)

  const voices = useVoices((s) => s.voices)
  const currentId = useVoices((s) => s.currentId)
  const setCurrent = useVoices((s) => s.setCurrent)
  const modelStatuses = useSettings((s) => s.modelStatuses)
  const modelInstalled = modelStatuses.find((m) => m.active)?.installed
  const modelName =
    modelStatuses.find((m) => m.active)?.displayName ?? 'OmniVoice'
  // sampling params live in 设置-模型设置 now
  const settings = useSettings((s) => s.settings)
  const setView = useView((s) => s.setView)
  const setSettingsOpen = useView((s) => s.setSettingsOpen)

  const voice = voices.find((v) => v.id === currentId) ?? voices[0]
  const charCount = text.trim().length
  // 中文 TTS 经验值 ~4.5 字/秒
  const estSeconds = Math.ceil(charCount / 4.5)

  const onImportTxt = (f: File | undefined) => {
    if (!f) return
    const reader = new FileReader()
    reader.onload = () => {
      const s = String(reader.result ?? '').replace(/\r\n/g, '\n')
      setText(s.slice(0, 20000))
    }
    reader.readAsText(f)
  }

  const onGenerate = () => {
    if (!modelInstalled) {
      toast.info('请先安装模型')
      setSettingsOpen(true)
      return
    }
    if (!charCount) {
      toast.info('请输入文本')
      return
    }
    setDropdownOpen(false)
    run({
      text,
      voiceId: voice?.id === 'default' ? null : (voice?.id ?? null),
      instruct: instruct.trim() ? instruct.trim() : null,
      seed: settings?.seed ?? 0,
    })
  }

  return (
    <div className="workspace-layout">
      <div className="text-input-panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">主工作台</h2>
            <p className="panel-subtitle">
              {modelName} · 输入文本，选择声音，即刻生成语音
            </p>
          </div>
        </div>
        <div className="text-editor-card">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="在此输入要转换为语音的文本…&#10;&#10;支持中英混排；长文本会自动分块生成。"
            spellCheck={false}
            disabled={running}
          />
          {running && (
            <div className="gen-overlay">
              <div className="gen-overlay-stage">
                <span>{stageText}</span>
                <span className="gen-overlay-pct">{Math.round(fraction * 100)}%</span>
              </div>
              <div className="gen-progress-track">
                {segments.map((s) => (
                  <div
                    key={s.key}
                    className="gen-progress-seg"
                    style={{ flexGrow: s.weight, background: `${s.color}26` }}
                    title={s.label}
                  >
                    <div
                      className="gen-progress-seg-fill"
                      style={{ width: `${s.fraction * 100}%`, background: s.color }}
                    />
                  </div>
                ))}
              </div>
              <div className="gen-overlay-sub">{subText}</div>
              <button className="gen-cancel-btn" disabled={cancelling} onClick={cancel}>
                {cancelling ? '正在取消…' : '取消生成'}
              </button>
            </div>
          )}
          <div className="editor-footer">
            <span className="char-count">
              {charCount} 字{charCount > 0 && ` · 预计 ${estSeconds}s 音频`}
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="upload-btn" disabled={running} onClick={() => fileInput.current?.click()}>
                <IconUpload />
                导入 .txt
              </button>
              <input
                ref={fileInput}
                type="file"
                accept=".txt,text/plain"
                className="sr-only"
                onChange={(e) => {
                  onImportTxt(e.target.files?.[0])
                  e.target.value = ''
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="params-panel">
        {/* ---- 声音选择 ---- */}
        <div className="param-card" style={{ position: 'relative' }}>
          <div className="param-card-title">声音</div>
          <button
            className="voice-selector"
            disabled={running}
            onClick={() => setDropdownOpen((v) => !v)}
          >
            <span className="voice-avatar">
              <VoiceAvatar name={voice?.name ?? '默认音色'} size={32} />
            </span>
            <span className="voice-info">
              <span className="voice-name">{voice?.name ?? '默认音色'}</span>
              <span className="voice-meta" style={{ display: 'block' }}>
                {voice ? `${voice.gender} · ${voice.language}` : ''}
              </span>
            </span>
            <IconChevron className={`selector-arrow ${dropdownOpen ? 'open' : ''}`} />
          </button>
          {dropdownOpen && (
            <div className="voice-dropdown">
              {voices.map((v) => (
                <VoiceOption
                  key={v.id}
                  v={v}
                  selected={v.id === voice?.id}
                  onPick={() => {
                    setCurrent(v.id)
                    setDropdownOpen(false)
                  }}
                />
              ))}
              <button
                className="voice-option"
                onClick={() => {
                  setDropdownOpen(false)
                  setView('voices')
                }}
              >
                <span className="voice-info">
                  <span className="voice-name" style={{ color: 'var(--seed-primary)' }}>
                    管理声音库 →
                  </span>
                </span>
              </button>
            </div>
          )}
        </div>

        {/* ---- 风格指令（OmniVoice） ---- */}
        <div className="param-card">
          <div className="param-card-title">风格指令（可选）</div>
          <input
            className="form-input"
            style={{ width: '100%', fontSize: 13 }}
            placeholder="例如：用愉快的语气说话"
            value={instruct}
            maxLength={200}
            spellCheck={false}
            disabled={running}
            onChange={(e) => setInstruct(e.target.value)}
          />
          <div className="slider-hint" style={{ marginTop: 6 }}>
            传给 OmniVoice 的 instruct 风格描述，留空则不使用。
          </div>
        </div>

        {/* ---- 韵律（模型暂不支持，保留设计 UI） ---- */}
        <div className="param-card">
          <div className="param-card-title">韵律</div>
          <Slider
            label="语速"
            value={1.0}
            min={0.5}
            max={1.5}
            step={0.05}
            format={(v) => `${v.toFixed(2)}×`}
            onChange={() => {}}
            disabled
            hint="当前模型暂不支持"
          />
          <Slider
            label="音调"
            value={0}
            min={-6}
            max={6}
            step={1}
            format={(v) => (v === 0 ? '0 半音' : `${v > 0 ? '+' : ''}${v}`)}
            onChange={() => {}}
            disabled
            hint="当前模型暂不支持"
          />
          <Slider
            label="情感强度"
            value={0.5}
            min={0}
            max={1}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={() => {}}
            disabled
            hint="当前模型暂不支持"
          />
        </div>

        <button className="generate-btn" disabled={running} onClick={onGenerate}>
          {running ? (
            '生成中…'
          ) : (
            <>
              <IconPlay />
              生成语音
            </>
          )}
        </button>
      </div>
    </div>
  )
}

function VoiceOption({
  v,
  selected,
  onPick,
}: {
  v: VoiceView
  selected: boolean
  onPick: () => void
}) {
  return (
    <button className={`voice-option ${selected ? 'selected' : ''}`} onClick={onPick}>
      <span className="voice-avatar">
        <VoiceAvatar name={v.name} size={32} />
      </span>
      <span className="voice-info">
        <span className="voice-name">{v.name}</span>
        <span className="voice-meta" style={{ display: 'block' }}>
          {v.isClone ? '克隆音色' : '预置'} · {v.language}
        </span>
      </span>
    </button>
  )
}

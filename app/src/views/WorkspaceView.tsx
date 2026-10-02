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
import type { GenOverrides, VoiceView } from '../types'

/** Voice Design instruct categories (same attribute set as the upstream
 * docs; the model auto-normalises Chinese/English/mixed strings). */
const DESIGN_CATEGORIES: { key: string; label: string; options: string[] }[] = [
  { key: 'gender', label: '性别', options: ['男', '女'] },
  { key: 'age', label: '年龄', options: ['儿童', '少年', '青年', '中年', '老年'] },
  {
    key: 'pitch',
    label: '音调',
    options: ['极低音调', '低音调', '中音调', '高音调', '极高音调'],
  },
  { key: 'style', label: '风格', options: ['耳语'] },
  {
    key: 'accent',
    label: '英语口音',
    options: [
      'american accent',
      'british accent',
      'australian accent',
      'canadian accent',
      'indian accent',
      'chinese accent',
      'korean accent',
      'japanese accent',
      'portuguese accent',
      'russian accent',
    ],
  },
  {
    key: 'dialect',
    label: '方言',
    options: [
      '河南话',
      '陕西话',
      '四川话',
      '贵州话',
      '云南话',
      '桂林话',
      '济南话',
      '石家庄话',
      '甘肃话',
      '宁夏话',
      '青岛话',
      '东北话',
    ],
  },
]

/** Advanced generation settings — defaults mirror the official
 * OmniVoiceGenerationConfig. */
const ADV_DEFAULTS = {
  speed: 1,
  duration: 0,
  denoise: true,
  numStep: 32,
  guidanceScale: 2,
  tShift: 0.1,
  positionTemperature: 5,
  classTemperature: 0,
  layerPenaltyFactor: 5,
  postprocessOutput: true,
  padDuration: 0.1,
  fadeDuration: 0.1,
  audioChunkDuration: 15,
  audioChunkThreshold: 30,
}

function ToggleRow({
  label,
  hint,
  value,
  onChange,
  disabled,
}: {
  label: string
  hint?: string
  value: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <div className={`toggle-row ${disabled ? 'disabled' : ''}`}>
      <div className="toggle-info">
        <span className="toggle-label">{label}</span>
        {hint && <span className="slider-hint">{hint}</span>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        aria-label={label}
        className={`toggle-switch ${value ? 'on' : ''}`}
        disabled={disabled}
        onClick={() => onChange(!value)}
      >
        <span className="toggle-knob" />
      </button>
    </div>
  )
}

export function WorkspaceView() {
  const [text, setText] = useState('')
  const [advOpen, setAdvOpen] = useState(false)
  const [adv, setAdv] = useState(ADV_DEFAULTS)
  const [design, setDesign] = useState<Record<string, string>>({})
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const patchAdv = (p: Partial<typeof ADV_DEFAULTS>) => setAdv((a) => ({ ...a, ...p }))

  const running = useGeneration((s) => s.running)
  const cancelling = useGeneration((s) => s.cancelling)
  const run = useGeneration((s) => s.run)
  const cancel = useGeneration((s) => s.cancel)
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
  // 中文 TTS 经验值 ~4.5 字/秒；speed > 1 产出更短的音频；固定时长优先
  const estSeconds = adv.duration > 0
    ? Math.ceil(adv.duration)
    : Math.ceil(charCount / 4.5 / adv.speed)

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
    const attrs = DESIGN_CATEGORIES.map((c) => design[c.key]).filter((v) => v && v !== '不限')
    // 0 / false 等“默认值”字段直接传引擎默认，避免无意义覆盖
    const overrides: GenOverrides = {
      speed: adv.speed,
      duration: adv.duration > 0 ? adv.duration : null,
      denoise: adv.denoise,
      numStep: adv.numStep,
      guidanceScale: adv.guidanceScale,
      tShift: adv.tShift,
      positionTemperature: adv.positionTemperature,
      classTemperature: adv.classTemperature,
      layerPenaltyFactor: adv.layerPenaltyFactor,
      postprocessOutput: adv.postprocessOutput,
      padDuration: adv.padDuration,
      fadeDuration: adv.fadeDuration,
      audioChunkDuration: adv.audioChunkDuration,
      audioChunkThreshold: adv.audioChunkThreshold,
    }
    run({
      text,
      voiceId: voice?.id === 'default' ? null : (voice?.id ?? null),
      instruct: attrs.length ? attrs.join('，') : null,
      seed: settings?.seed ?? 0,
      overrides,
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
              </div>
              <div className="gen-progress-track">
                {segments.map((s) => (
                  <div
                    key={s.key}
                    className="gen-progress-seg"
                    style={{ background: `${s.color}26` }}
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
              <VoiceAvatar name={voice?.name ?? '默认音色'} icon={voice?.icon ?? null} size={32} />
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

        {/* ---- 声音设计（instruct 说话人属性） ---- */}
        <div className="param-card">
          <div className="param-card-title">声音设计（可选）</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {DESIGN_CATEGORIES.map((c) => (
              <label key={c.key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{c.label}</span>
                <select
                  className="form-input form-select"
                  style={{ width: '100%', fontSize: 12, padding: '8px 10px' }}
                  aria-label={c.label}
                  value={design[c.key] ?? '不限'}
                  disabled={running}
                  onChange={(e) =>
                    setDesign((d) => ({ ...d, [c.key]: e.target.value }))
                  }
                >
                  {['不限', ...c.options].map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="slider-hint" style={{ marginTop: 8 }}>
            按属性描述目标音色，无需参考音频；全部留空时使用上方所选声音。英语口音仅对英文文本生效，方言仅对中文生效。
          </div>
        </div>

        {/* ---- 其他设置（默认折叠） ---- */}
        <div className={`param-card advanced-card ${advOpen ? 'open' : ''}`}>
          <button
            className="advanced-toggle"
            onClick={() => setAdvOpen((v) => !v)}
            disabled={running}
            aria-expanded={advOpen}
          >
            <span className="advanced-toggle-text">
              <span className="advanced-toggle-title">其他设置</span>
              <span className="advanced-toggle-sub">语速、时长、解码与后处理</span>
            </span>
            <IconChevron className={`selector-arrow ${advOpen ? 'open' : ''}`} />
          </button>

          {advOpen && (
            <div className="advanced-body">
              {/* -- 时长与速度 -- */}
              <div className="advanced-section">时长与速度</div>
              <Slider
                label="语速"
                value={adv.speed}
                min={0.5}
                max={2}
                step={0.05}
                format={(v) => `${v.toFixed(2)}×`}
                onChange={(v) => patchAdv({ speed: v })}
                disabled={running}
                hint="> 1 更快更短，< 1 更慢更长；默认 1.0"
              />
              <div className="advanced-field">
                <label className="form-label" htmlFor="adv-duration">
                  固定时长（秒）
                </label>
                <input
                  id="adv-duration"
                  className="form-input"
                  type="number"
                  min={0}
                  max={300}
                  step={0.5}
                  placeholder="自动"
                  value={adv.duration > 0 ? adv.duration : ''}
                  disabled={running}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    patchAdv({
                      duration: Number.isFinite(v) && v > 0 ? Math.min(v, 300) : 0,
                    })
                  }}
                />
                <div className="slider-hint">
                  设置后忽略语速、不分块，按此时长生成；留空按文本自动估计
                </div>
              </div>

              {/* -- 解码 -- */}
              <div className="advanced-section">解码</div>
              <Slider
                label="去掩码步数"
                value={adv.numStep}
                min={4}
                max={64}
                step={1}
                format={(v) => String(v)}
                onChange={(v) => patchAdv({ numStep: v })}
                disabled={running}
                hint="迭代去掩码步数，越多质量越好但越慢；默认 32，快速可用 16"
              />
              <Slider
                label="引导强度"
                value={adv.guidanceScale}
                min={0}
                max={5}
                step={0.1}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ guidanceScale: v })}
                disabled={running}
                hint="Classifier-free guidance scale；默认 2.0"
              />
              <Slider
                label="时间偏移"
                value={adv.tShift}
                min={0.02}
                max={0.5}
                step={0.01}
                format={(v) => v.toFixed(2)}
                onChange={(v) => patchAdv({ tShift: v })}
                disabled={running}
                hint="噪声调度时间步偏移，越小越强调早期步；默认 0.1"
              />

              {/* -- 采样 -- */}
              <div className="advanced-section">采样</div>
              <Slider
                label="位置温度"
                value={adv.positionTemperature}
                min={0}
                max={10}
                step={0.5}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ positionTemperature: v })}
                disabled={running}
                hint="掩码位置选择随机度，0 = 贪心；默认 5.0"
              />
              <Slider
                label="Token 温度"
                value={adv.classTemperature}
                min={0}
                max={2}
                step={0.05}
                format={(v) => v.toFixed(2)}
                onChange={(v) => patchAdv({ classTemperature: v })}
                disabled={running}
                hint="Token 采样随机度，0 = 贪心；默认 0"
              />
              <Slider
                label="层级惩罚"
                value={adv.layerPenaltyFactor}
                min={0}
                max={10}
                step={0.5}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ layerPenaltyFactor: v })}
                disabled={running}
                hint="深层码本惩罚，低层码本优先解码；默认 5.0"
              />

              {/* -- 输出处理 -- */}
              <div className="advanced-section">输出处理</div>
              <ToggleRow
                label="降噪标签"
                hint="生成更干净的语音；默认开启"
                value={adv.denoise}
                onChange={(v) => patchAdv({ denoise: v })}
                disabled={running}
              />
              <ToggleRow
                label="输出后处理"
                hint="移除长静音；默认开启"
                value={adv.postprocessOutput}
                onChange={(v) => patchAdv({ postprocessOutput: v })}
                disabled={running}
              />
              <Slider
                label="静音填充"
                value={adv.padDuration}
                min={0}
                max={0.5}
                step={0.01}
                format={(v) => `${v.toFixed(2)}s`}
                onChange={(v) => patchAdv({ padDuration: v })}
                disabled={running}
                hint="首尾每侧静音填充时长；默认 0.1s"
              />
              <Slider
                label="淡入淡出"
                value={adv.fadeDuration}
                min={0}
                max={0.5}
                step={0.01}
                format={(v) => `${v.toFixed(2)}s`}
                onChange={(v) => patchAdv({ fadeDuration: v })}
                disabled={running}
                hint="首尾线性淡入淡出长度；默认 0.1s"
              />

              {/* -- 长文本分块 -- */}
              <div className="advanced-section">长文本分块</div>
              <Slider
                label="分块时长"
                value={adv.audioChunkDuration}
                min={5}
                max={30}
                step={1}
                format={(v) => `${v}s`}
                onChange={(v) => patchAdv({ audioChunkDuration: v })}
                disabled={running}
                hint="长文本每块目标音频时长；默认 15s"
              />
              <Slider
                label="分块阈值"
                value={adv.audioChunkThreshold}
                min={10}
                max={60}
                step={5}
                format={(v) => `${v}s`}
                onChange={(v) => patchAdv({ audioChunkThreshold: v })}
                disabled={running}
                hint="估计时长超过该值才启用分块；默认 30s"
              />

              <button
                className="btn-secondary advanced-reset"
                disabled={running}
                onClick={() => setAdv(ADV_DEFAULTS)}
              >
                恢复默认
              </button>
            </div>
          )}
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
        <VoiceAvatar name={v.name} icon={v.icon} size={32} />
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

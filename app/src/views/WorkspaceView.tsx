import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from '../stores/toast'
import {
  IconChevron,
  IconPlay,
  IconUpload,
} from '../components/Icons'
import { Slider } from '../components/common/Slider'
import { VoiceAvatar } from '../components/common/VoiceAvatar'
import { LangCombobox } from '../components/common/LangCombobox'
import { useGeneration } from '../stores/generation'
import { useSettings } from '../stores/settings'
import { useView } from '../stores/view'
import { useVoices } from '../stores/voices'
import type { GenOverrides, VoiceView } from '../types'

/** "No constraint" sentinel for the design dropdowns (session state only). */
const ANY = '__any__'

/** Voice Design instruct categories. `id` is the stable i18n key;
 * `value` is the canonical model-facing string (the model normalises
 * Chinese/English/mixed — keep it unchanged so prompts stay identical). */
const DESIGN_CATEGORIES: { key: string; options: { id: string; value: string }[] }[] = [
  {
    key: 'gender',
    options: [
      { id: 'male', value: '男' },
      { id: 'female', value: '女' },
    ],
  },
  {
    key: 'age',
    options: [
      { id: 'child', value: '儿童' },
      { id: 'teen', value: '少年' },
      { id: 'young', value: '青年' },
      { id: 'middle', value: '中年' },
      { id: 'senior', value: '老年' },
    ],
  },
  {
    key: 'pitch',
    options: [
      { id: 'veryLow', value: '极低音调' },
      { id: 'low', value: '低音调' },
      { id: 'mid', value: '中音调' },
      { id: 'high', value: '高音调' },
      { id: 'veryHigh', value: '极高音调' },
    ],
  },
  { key: 'style', options: [{ id: 'whisper', value: '耳语' }] },
  {
    key: 'accent',
    options: [
      { id: 'american', value: 'american accent' },
      { id: 'british', value: 'british accent' },
      { id: 'australian', value: 'australian accent' },
      { id: 'canadian', value: 'canadian accent' },
      { id: 'indian', value: 'indian accent' },
      { id: 'chinese', value: 'chinese accent' },
      { id: 'korean', value: 'korean accent' },
      { id: 'japanese', value: 'japanese accent' },
      { id: 'portuguese', value: 'portuguese accent' },
      { id: 'russian', value: 'russian accent' },
    ],
  },
  {
    key: 'dialect',
    options: [
      { id: 'henan', value: '河南话' },
      { id: 'shaanxi', value: '陕西话' },
      { id: 'sichuan', value: '四川话' },
      { id: 'guizhou', value: '贵州话' },
      { id: 'yunnan', value: '云南话' },
      { id: 'guilin', value: '桂林话' },
      { id: 'jinan', value: '济南话' },
      { id: 'shijiazhuang', value: '石家庄话' },
      { id: 'gansu', value: '甘肃话' },
      { id: 'ningxia', value: '宁夏话' },
      { id: 'qingdao', value: '青岛话' },
      { id: 'dongbei', value: '东北话' },
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
  const { t } = useTranslation()
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
  const saveSettings = useSettings((s) => s.save)
  const setView = useView((s) => s.setView)
  const setSettingsOpen = useView((s) => s.setSettingsOpen)

  const voice = voices.find((v) => v.id === currentId)
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
      toast.info(t('toasts.installModelFirst'))
      setSettingsOpen(true)
      return
    }
    if (!charCount) {
      toast.info(t('toasts.enterText'))
      return
    }
    setDropdownOpen(false)
    const attrs = DESIGN_CATEGORIES.map((c) => {
      const o = c.options.find((o) => o.id === design[c.key])
      return o?.value
    }).filter(Boolean) as string[]
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
      voiceId: voice?.id ?? null,
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
            <h2 className="panel-title">{t('workspace.title')}</h2>
            <p className="panel-subtitle">{t('workspace.subtitle', { model: modelName })}</p>
          </div>
        </div>
        <div className="text-editor-card">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t('workspace.placeholder')}
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
                    title={t(s.labelKey)}
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
                {cancelling ? t('workspace.cancelling') : t('workspace.cancelGen')}
              </button>
            </div>
          )}
          <div className="editor-footer">
            <span className="char-count">
              {charCount > 0
                ? `${t('workspace.charCount', { n: charCount })} · ${t('workspace.estAudio', { sec: estSeconds })}`
                : t('workspace.charCount', { n: charCount })}
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="upload-btn" disabled={running} onClick={() => fileInput.current?.click()}>
                <IconUpload />
                {t('workspace.importTxt')}
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
          <div className="param-card-title">{t('workspace.voice')}</div>
          <button
            className="voice-selector"
            disabled={running}
            onClick={() => setDropdownOpen((v) => !v)}
          >
            <span className="voice-avatar">
              <VoiceAvatar name={voice?.name ?? t('voices.none')} icon={voice?.icon ?? null} size={32} />
            </span>
            <span className="voice-info">
              <span
                className="voice-name"
                style={voice ? undefined : { color: 'var(--text-tertiary)' }}
              >
                {voice?.name ?? t('voices.none')}
              </span>
              <span className="voice-meta" style={{ display: 'block' }}>
                {voice && (voice.isClone ? t('voices.cloneTag') : t('voices.preset'))}
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
                  selected={v.id === currentId}
                  onPick={() => {
                    // clicking the selected voice again clears the selection
                    setCurrent(v.id === currentId ? null : v.id)
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
                    {t('workspace.manageVoices')}
                  </span>
                </span>
              </button>
            </div>
          )}
        </div>

        {/* ---- 声音设计（instruct 说话人属性） ---- */}
        <div className="param-card">
          <div className="param-card-title">{t('workspace.designTitle')}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {DESIGN_CATEGORIES.map((c) => (
              <label key={c.key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{t(`design.${c.key}`)}</span>
                <select
                  className="form-input form-select"
                  style={{ width: '100%', fontSize: 12, padding: '8px 10px' }}
                  aria-label={t(`design.${c.key}`)}
                  value={design[c.key] ?? ANY}
                  disabled={running}
                  onChange={(e) =>
                    setDesign((d) => ({ ...d, [c.key]: e.target.value }))
                  }
                >
                  {[ANY, ...c.options.map((o) => o.id)].map((id) => (
                    <option key={id} value={id}>
                      {id === ANY ? t('design.any') : t(`design.${c.key}Opt.${id}`)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="slider-hint" style={{ marginTop: 8 }}>
            {t('workspace.designHint')}
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
              <span className="advanced-toggle-title">{t('workspace.advancedTitle')}</span>
              <span className="advanced-toggle-sub">{t('workspace.advancedSub')}</span>
            </span>
            <IconChevron className={`selector-arrow ${advOpen ? 'open' : ''}`} />
          </button>

          {advOpen && (
            <div className="advanced-body">
              {/* -- 输出语言 -- */}
              <div className="advanced-field">
                <label className="form-label" htmlFor="adv-output-lang">
                  {t('workspace.outputLanguage')}
                </label>
                <LangCombobox
                  id="adv-output-lang"
                  value={settings?.outputLanguage ?? ''}
                  disabled={running}
                  autoLabel={t('workspace.outputLangAuto')}
                  placeholder={t('workspace.outputLangPlaceholder')}
                  onChange={(v) => {
                    if (settings) void saveSettings({ ...settings, outputLanguage: v })
                  }}
                />
                <div className="slider-hint">{t('workspace.outputLangHint')}</div>
              </div>

              {/* -- 时长与速度 -- */}
              <div className="advanced-section">{t('workspace.secSpeed')}</div>
              <Slider
                label={t('workspace.speed')}
                value={adv.speed}
                min={0.5}
                max={2}
                step={0.05}
                format={(v) => `${v.toFixed(2)}×`}
                onChange={(v) => patchAdv({ speed: v })}
                disabled={running}
                hint={t('workspace.speedHint')}
              />
              <div className="advanced-field">
                <label className="form-label" htmlFor="adv-duration">
                  {t('workspace.fixedDuration')}
                </label>
                <input
                  id="adv-duration"
                  className="form-input"
                  type="number"
                  min={0}
                  max={300}
                  step={0.5}
                  placeholder={t('common.auto')}
                  value={adv.duration > 0 ? adv.duration : ''}
                  disabled={running}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    patchAdv({
                      duration: Number.isFinite(v) && v > 0 ? Math.min(v, 300) : 0,
                    })
                  }}
                />
                <div className="slider-hint">{t('workspace.fixedDurationHint')}</div>
              </div>

              {/* -- 解码 -- */}
              <div className="advanced-section">{t('workspace.decoding')}</div>
              <Slider
                label={t('workspace.numStep')}
                value={adv.numStep}
                min={4}
                max={64}
                step={1}
                format={(v) => String(v)}
                onChange={(v) => patchAdv({ numStep: v })}
                disabled={running}
                hint={t('workspace.numStepHint')}
              />
              <Slider
                label={t('workspace.guidance')}
                value={adv.guidanceScale}
                min={0}
                max={5}
                step={0.1}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ guidanceScale: v })}
                disabled={running}
                hint={t('workspace.guidanceHint')}
              />
              <Slider
                label={t('workspace.tShift')}
                value={adv.tShift}
                min={0.02}
                max={0.5}
                step={0.01}
                format={(v) => v.toFixed(2)}
                onChange={(v) => patchAdv({ tShift: v })}
                disabled={running}
                hint={t('workspace.tShiftHint')}
              />

              {/* -- 采样 -- */}
              <div className="advanced-section">{t('workspace.sampling')}</div>
              <Slider
                label={t('workspace.positionTemp')}
                value={adv.positionTemperature}
                min={0}
                max={10}
                step={0.5}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ positionTemperature: v })}
                disabled={running}
                hint={t('workspace.positionTempHint')}
              />
              <Slider
                label={t('workspace.classTemp')}
                value={adv.classTemperature}
                min={0}
                max={2}
                step={0.05}
                format={(v) => v.toFixed(2)}
                onChange={(v) => patchAdv({ classTemperature: v })}
                disabled={running}
                hint={t('workspace.classTempHint')}
              />
              <Slider
                label={t('workspace.layerPenalty')}
                value={adv.layerPenaltyFactor}
                min={0}
                max={10}
                step={0.5}
                format={(v) => v.toFixed(1)}
                onChange={(v) => patchAdv({ layerPenaltyFactor: v })}
                disabled={running}
                hint={t('workspace.layerPenaltyHint')}
              />

              {/* -- 输出处理 -- */}
              <div className="advanced-section">{t('workspace.outputProc')}</div>
              <ToggleRow
                label={t('workspace.denoise')}
                hint={t('workspace.denoiseHint')}
                value={adv.denoise}
                onChange={(v) => patchAdv({ denoise: v })}
                disabled={running}
              />
              <ToggleRow
                label={t('workspace.postprocess')}
                hint={t('workspace.postprocessHint')}
                value={adv.postprocessOutput}
                onChange={(v) => patchAdv({ postprocessOutput: v })}
                disabled={running}
              />
              <Slider
                label={t('workspace.pad')}
                value={adv.padDuration}
                min={0}
                max={0.5}
                step={0.01}
                format={(v) => `${v.toFixed(2)}s`}
                onChange={(v) => patchAdv({ padDuration: v })}
                disabled={running}
                hint={t('workspace.padHint')}
              />
              <Slider
                label={t('workspace.fade')}
                value={adv.fadeDuration}
                min={0}
                max={0.5}
                step={0.01}
                format={(v) => `${v.toFixed(2)}s`}
                onChange={(v) => patchAdv({ fadeDuration: v })}
                disabled={running}
                hint={t('workspace.fadeHint')}
              />

              {/* -- 长文本分块 -- */}
              <div className="advanced-section">{t('workspace.chunking')}</div>
              <Slider
                label={t('workspace.chunkDur')}
                value={adv.audioChunkDuration}
                min={5}
                max={30}
                step={1}
                format={(v) => `${v}s`}
                onChange={(v) => patchAdv({ audioChunkDuration: v })}
                disabled={running}
                hint={t('workspace.chunkDurHint')}
              />
              <Slider
                label={t('workspace.chunkThresh')}
                value={adv.audioChunkThreshold}
                min={10}
                max={60}
                step={5}
                format={(v) => `${v}s`}
                onChange={(v) => patchAdv({ audioChunkThreshold: v })}
                disabled={running}
                hint={t('workspace.chunkThreshHint')}
              />

              <button
                className="btn-secondary advanced-reset"
                disabled={running}
                onClick={() => setAdv(ADV_DEFAULTS)}
              >
                {t('workspace.reset')}
              </button>
            </div>
          )}
        </div>

        <button className="generate-btn" disabled={running} onClick={onGenerate}>
          {running ? (
            t('workspace.generating')
          ) : (
            <>
              <IconPlay />
              {t('workspace.genBtn')}
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
  const { t } = useTranslation()
  return (
    <button className={`voice-option ${selected ? 'selected' : ''}`} onClick={onPick}>
      <span className="voice-avatar">
        <VoiceAvatar name={v.name} icon={v.icon} size={32} />
      </span>
      <span className="voice-info">
        <span className="voice-name">{v.name}</span>
        <span className="voice-meta" style={{ display: 'block' }}>
          {v.isClone ? t('voices.cloneTag') : t('voices.preset')}
        </span>
      </span>
    </button>
  )
}

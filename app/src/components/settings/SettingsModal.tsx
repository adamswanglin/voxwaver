import { useState } from 'react'
import { IconClose, IconCube, IconFolder, IconUpload } from '../Icons'
import { Slider } from '../common/Slider'
import { formatBytes } from '../../lib/format'
import { useSettings } from '../../stores/settings'
import { useView } from '../../stores/view'
import type { ModelStatus } from '../../types'

type Tab = 'basic' | 'model'

// Languages supported by the models; the app UI ships the same set.
const LANGUAGES = [
  { value: 'zh', label: '简体中文' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: '日本語' },
  { value: 'de', label: 'Deutsch' },
  { value: 'fr', label: 'Français' },
  { value: 'es', label: 'Español' },
  { value: 'ko', label: '한국어' },
  { value: 'ar', label: 'العربية' },
  { value: 'ru', label: 'Русский' },
  { value: 'nl', label: 'Nederlands' },
  { value: 'it', label: 'Italiano' },
  { value: 'pl', label: 'Polski' },
  { value: 'pt', label: 'Português' },
] as const

const DEVICE_OPTIONS = [
  { id: 'cpu', title: 'CPU', sub: '纯 CPU 推理，兼容性最佳', recommended: false },
  { id: 'metal', title: 'Metal', sub: 'macOS GPU 加速', recommended: true },
  { id: 'cuda', title: 'CUDA', sub: 'NVIDIA GPU 加速', recommended: false },
] as const

export function SettingsModal() {
  const openState = useView((s) => s.settingsOpen)
  const setOpen = useView((s) => s.setSettingsOpen)
  const [tab, setTab] = useState<Tab>('basic')

  const settings = useSettings((s) => s.settings)
  const devices = useSettings((s) => s.devices)
  const modelStatuses = useSettings((s) => s.modelStatuses)
  const save = useSettings((s) => s.save)
  const activate = useSettings((s) => s.activate)

  if (!openState) return null

  const patch = (p: Partial<NonNullable<typeof settings>>) => {
    if (settings) void save({ ...settings, ...p })
  }

  // 'auto' follows the recommended pick (Metal) for display
  const device = settings?.device === 'cpu' || settings?.device === 'cuda' ? settings.device : 'metal'
  // legacy value from earlier builds
  const uiLang = settings?.language === 'zh-CN' ? 'zh' : (settings?.language ?? 'zh')
  // sampling params, persisted with the rest of the settings
  const omniTemperature = settings?.omniTemperature ?? 0
  const seed = settings?.seed ?? 0

  return (
    <div className="settings-overlay show" onClick={() => setOpen(false)}>
      <div
        className="settings-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="设置"
      >
        <div className="modal-header">
          <div className="modal-title">设置</div>
          <button className="modal-close" aria-label="关闭" onClick={() => setOpen(false)}>
            <IconClose />
          </button>
        </div>
        <div className="settings-tabs">
          <button
            className={`settings-tab ${tab === 'basic' ? 'active' : ''}`}
            onClick={() => setTab('basic')}
          >
            基本设置
          </button>
          <button
            className={`settings-tab ${tab === 'model' ? 'active' : ''}`}
            onClick={() => setTab('model')}
          >
            模型设置
          </button>
        </div>

        <div className="settings-content">
          {tab === 'basic' && (
            <div className="settings-panel active">
              <div className="form-group" style={{ marginBottom: 20 }}>
                <label className="settings-label">界面语言</label>
                <select
                  className="form-input form-select"
                  style={{ width: '100%', fontSize: 13, padding: '10px 14px' }}
                  aria-label="界面语言"
                  value={uiLang}
                  onChange={(e) => patch({ language: e.target.value })}
                >
                  {LANGUAGES.map((l) => (
                    <option key={l.value} value={l.value}>
                      {l.label}
                    </option>
                  ))}
                </select>
                <div className="slider-hint" style={{ marginTop: 6 }}>
                  OmniVoice 模型会把界面语言作为合成语言标签传入。
                </div>
              </div>
            </div>
          )}

          {tab === 'model' && (
            <div className="settings-panel active">
              {/* 模型管理 */}
              <div className="form-group" style={{ marginBottom: 20 }}>
                <label className="settings-label">模型</label>
                {modelStatuses.map((m) => (
                  <ModelCard key={m.model} status={m} onActivate={() => void activate(m.model)} />
                ))}
              </div>

              {/* 推理设备 */}
              <div className="form-group" style={{ marginBottom: 18 }}>
                <label className="settings-label">推理设备</label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {DEVICE_OPTIONS.map((o) => {
                    const unavailable =
                      (o.id === 'metal' && !devices?.metal) || (o.id === 'cuda' && !devices?.cuda)
                    return (
                      <button
                        key={o.id}
                        className={`option-card ${device === o.id ? 'selected' : ''} ${unavailable ? 'disabled' : ''}`}
                        disabled={unavailable}
                        onClick={() => patch({ device: o.id })}
                      >
                        <div style={{ flex: 1 }}>
                          <div className="option-title">{o.title}</div>
                          <div className="option-sub">{o.sub}</div>
                        </div>
                        {unavailable ? (
                          <span className="option-sub">不可用</span>
                        ) : o.recommended ? (
                          <span className="option-badge">推荐</span>
                        ) : null}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* 采样参数 */}
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="settings-label">采样参数</label>
                <Slider
                  label="温度（OmniVoice）"
                  value={omniTemperature}
                  min={0}
                  max={1.5}
                  step={0.05}
                  format={(v) => (v === 0 ? '贪心' : v.toFixed(2))}
                  onChange={(omniTemperature) => patch({ omniTemperature })}
                />
                <div className="slider-hint" style={{ marginTop: 6 }}>
                  0 为贪心解码（与上游 CLI 默认一致）；其余采样超参使用引擎默认值。
                </div>
                <div className="form-group" style={{ marginBottom: 0, marginTop: 14 }}>
                  <label className="form-label">随机种子</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      className="form-input"
                      type="number"
                      min={0}
                      value={seed}
                      onChange={(e) =>
                        patch({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })
                      }
                    />
                    <button
                      className="btn-secondary"
                      style={{ padding: '0 14px' }}
                      onClick={() => patch({ seed: Math.floor(Math.random() * 1_000_000) })}
                      title="随机种子"
                    >
                      🎲
                    </button>
                  </div>
                  <div className="slider-hint">相同种子 + 相同参数 = 完全可复现的输出</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ModelCard({
  status,
  onActivate,
}: {
  status: ModelStatus
  onActivate: () => void
}) {
  const download = useSettings((s) => s.download)
  const importLocal = useSettings((s) => s.importLocal)
  const downloadModel = useSettings((s) => s.downloadModel)
  const cancelDownload = useSettings((s) => s.cancelDownload)
  const deleteModel = useSettings((s) => s.deleteModel)

  const downloadingThis = download?.model === status.model
  const dlPct =
    downloadingThis && download!.total > 0 ? (download!.downloaded / download!.total) * 100 : 0

  return (
    <div
      style={{
        padding: 16,
        borderRadius: 12,
        border: `1px solid ${status.active ? 'var(--seed-primary)' : 'var(--border-color)'}`,
        background: 'var(--seed-bg)',
        marginBottom: 10,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 10,
            background: 'linear-gradient(135deg,#f093fb,#f5576c)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <IconCube width={20} height={20} color="white" />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>
            {status.displayName}
            {status.active && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  color: 'var(--seed-primary)',
                  border: '1px solid var(--seed-primary)',
                  borderRadius: 4,
                  padding: '1px 6px',
                  marginLeft: 8,
                  verticalAlign: 'middle',
                }}
              >
                使用中
              </span>
            )}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
            Qwen3 迭代解码 · 24 kHz · 支持克隆与风格指令 · ~3.3 GB
          </div>
        </div>
        {status.active ? (
          <span className="status-btn installed" style={{ fontSize: 11 }}>
            当前模型
          </span>
        ) : status.installed ? (
          <button
            className="status-btn installed"
            style={{ fontSize: 11, cursor: 'pointer' }}
            onClick={onActivate}
          >
            使用此模型
          </button>
        ) : downloadingThis ? (
          <button className="status-btn downloading" style={{ fontSize: 11 }}>
            下载中
          </button>
        ) : (
          <button className="status-btn downloading" style={{ fontSize: 11 }}>
            未安装
          </button>
        )}
      </div>

      {status.installed ? (
        <div className="model-dir-row">
          <IconFolder />
          <span>{status.path}</span>
          <button className="link-danger" onClick={() => void deleteModel(status.model)}>
            删除
          </button>
        </div>
      ) : downloadingThis ? (
        <div className="dl-progress" style={{ marginTop: 0 }}>
          <div className="dl-file-label">
            <span>
              下载中 {download!.fileIndex + 1}/{download!.fileCount} · {download!.file}
            </span>
            <span>
              {formatBytes(download!.downloaded)}
              {download!.total > 0 ? ` / ${formatBytes(download!.total)}` : ''} ·{' '}
              {formatBytes(download!.speedBps)}/s
            </span>
          </div>
          <div className="dl-track">
            <div className="dl-fill" style={{ width: `${dlPct}%` }} />
          </div>
          <button
            className="btn-secondary"
            style={{ marginTop: 10 }}
            onClick={cancelDownload}
          >
            取消下载
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button className="install-option" onClick={() => void downloadModel(status.model, 'hf')}>
            <IconUpload />
            <div style={{ flex: 1 }}>
              <div className="option-title">从 HuggingFace 下载</div>
              <div className="option-sub">k2-fsa/OmniVoice · 海外源</div>
            </div>
          </button>
          <button className="install-option" onClick={() => void importLocal(status.model)}>
            <IconFolder />
            <div style={{ flex: 1 }}>
              <div className="option-title">选择本地模型文件夹</div>
              <div className="option-sub">需包含 model.safetensors 和 audio_tokenizer/</div>
            </div>
          </button>
        </div>
      )}
    </div>
  )
}

import type { DeviceProbe, ModelStatus, Settings } from '../types'
import {
  apiCancelDownload,
  apiDeleteModel,
  apiDownloadModel,
  apiGetSettings,
  apiImportLocalModel,
  apiListModelStatus,
  apiProbeDevices,
  apiSetSettings,
} from '../api'
import { onModelDownload } from '../api/events'
import i18n, { applyLanguage } from '../i18n'
import { create } from 'zustand'
import { toast } from './toast'

interface DownloadState {
  model: string
  file: string
  fileIndex: number
  fileCount: number
  downloaded: number
  total: number
  speedBps: number
}

interface SettingsStore {
  settings: Settings | null
  modelStatuses: ModelStatus[]
  devices: DeviceProbe | null
  download: DownloadState | null
  load: () => Promise<void>
  save: (s: Settings) => Promise<void>
  /** switch the active engine (applies on next generation) */
  activate: (model: string) => Promise<void>
  importLocal: (model: string) => Promise<void>
  downloadModel: (model: string, source: 'hf' | 'modelscope') => Promise<void>
  cancelDownload: () => void
  deleteModel: (model: string) => Promise<void>
  /** status of the currently active model (convenience selector) */
  activeStatus: () => ModelStatus | null
}

export const useSettings = create<SettingsStore>((set, get) => ({
  settings: null,
  modelStatuses: [],
  devices: null,
  download: null,

  load: async () => {
    const [settings, modelStatuses, devices] = await Promise.all([
      apiGetSettings(),
      apiListModelStatus(),
      apiProbeDevices(),
    ])
    // legacy value from earlier builds
    if (settings.language === 'zh-CN') settings.language = 'zh'
    applyLanguage(settings.language)
    set({ settings, modelStatuses, devices })
  },

  save: async (s) => {
    await apiSetSettings(s)
    if (s.language !== get().settings?.language) applyLanguage(s.language)
    set({ settings: s })
    const modelStatuses = await apiListModelStatus()
    set({ modelStatuses })
  },

  activate: async (model) => {
    const s = get().settings
    if (!s || s.model === model) return
    await get().save({ ...s, model })
    toast.info(i18n.t('toasts.modelSwitched'))
  },

  importLocal: async (model) => {
    try {
      const path = await apiImportLocalModel(model)
      if (path) {
        toast.success(i18n.t('toasts.modelImported', { path }))
        await get().load()
      }
    } catch (e) {
      toast.error(String(e))
    }
  },

  downloadModel: async (model, source) => {
    try {
      await apiDownloadModel(model, source)
      set({ download: null })
      toast.success(i18n.t('toasts.downloadDone'))
      await get().load()
    } catch (e) {
      set({ download: null })
      const msg = String(e)
      if (!msg.toLowerCase().includes('cancelled')) toast.error(i18n.t('toasts.downloadFail', { msg }))
      await get().load()
    }
  },

  cancelDownload: () => {
    apiCancelDownload().catch(() => {})
  },

  deleteModel: async (model) => {
    try {
      await apiDeleteModel(model)
      toast.info(i18n.t('toasts.modelDeleted'))
      await get().load()
    } catch (e) {
      toast.error(String(e))
    }
  },

  activeStatus: () => {
    const s = get()
    return s.modelStatuses.find((m) => m.active) ?? null
  },
}))

// live download progress -> store
let wired = false
export function wireDownloadEvents() {
  if (wired) return
  wired = true
  onModelDownload((e) => {
    useSettings.setState({ download: { ...e } })
  })
}

import type {
  DeviceProbe,
  EngineStatus,
  GenerateReq,
  HistoryEntry,
  ModelStatus,
  Settings,
  StorageStats,
  VoiceView,
} from '../types'
import { invoke } from '@tauri-apps/api/core'

// ---- tts ----
export const apiGenerate = (req: GenerateReq) =>
  invoke<HistoryEntry>('generate', { req })
export const apiCancelGenerate = () => invoke<void>('cancel_generate')
export const apiGetEngineStatus = () => invoke<EngineStatus>('get_engine_status')
export const apiWarmupEngine = () => invoke<void>('warmup_engine')

// ---- voices ----
export const apiListVoices = () => invoke<VoiceView[]>('list_voices')
export interface CreateVoiceReq {
  name: string
  /** optional emoji icon; null/'' = none */
  icon: string | null
  samplePath: string
  transcript: string
}
export const apiCreateVoice = (req: CreateVoiceReq) =>
  invoke<VoiceView>('create_voice', { req })
export interface UpdateVoiceReq {
  id: string
  name: string
  /** optional emoji icon; null/'' = clear (name-letter fallback) */
  icon: string | null
  transcript: string
  /** new reference sample; null/empty = keep the existing one */
  samplePath: string | null
}
export const apiUpdateVoice = (req: UpdateVoiceReq) =>
  invoke<VoiceView>('update_voice', { req })
export const apiDeleteVoice = (id: string) => invoke<void>('delete_voice', { id })
/** Persist webview-side PCM (recording / decoded MP3) as a WAV, returns its path. */
export const apiSaveSampleAudio = (sampleRate: number, samples: number[]) =>
  invoke<string>('save_recorded_sample', { sampleRate, samples })

// ---- history ----
export const apiListHistory = () => invoke<HistoryEntry[]>('list_history')
/** Grep-filter history entries by raw JSON text (case-insensitive substring). */
export const apiSearchHistory = (query: string) =>
  invoke<HistoryEntry[]>('search_history', { query })
export const apiDeleteHistory = (ids: string[]) => invoke<void>('delete_history', { ids })
export const apiExportAudio = (ids: string[], destDir: string) =>
  invoke<number>('export_audio', { ids, destDir })
export const apiRevealAudio = (id: string) => invoke<void>('reveal_audio', { id })
export const apiStorageStats = () => invoke<StorageStats>('storage_stats')

// ---- settings & model ----
export const apiGetSettings = () => invoke<Settings>('get_settings')
export const apiSetSettings = (settings: Settings) =>
  invoke<boolean>('set_settings', { settings })
export const apiProbeDevices = () => invoke<DeviceProbe>('probe_devices')
/** `model` omitted = the active model */
export const apiGetModelStatus = (model?: string) =>
  invoke<ModelStatus>('get_model_status', { model: model ?? null })
export const apiListModelStatus = () => invoke<ModelStatus[]>('list_model_status')
export const apiImportLocalModel = (model: string) =>
  invoke<string | null>('import_local_model', { model })
export const apiDeleteModel = (model: string) => invoke<void>('delete_model', { model })
export const apiDownloadModel = (model: string, source: 'hf' | 'modelscope') =>
  invoke<void>('download_model', { model, source })
export const apiCancelDownload = () => invoke<void>('cancel_download')

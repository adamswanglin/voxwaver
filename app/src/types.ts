export type ViewName = 'workspace' | 'voices' | 'history'

export interface Settings {
  device: string // auto | cpu | metal | cuda
  model: string // active engine
  modelDirs: Record<string, string> // model id -> directory
  language: string
  /** OmniVoice class temperature; 0 = greedy (upstream CLI default) */
  omniTemperature: number
  seed: number
}

export interface VoiceView {
  id: string
  name: string
  gender: string
  age: string
  style: string
  language: string
  tags: string[]
  createdAt: number
  isClone: boolean
  refWav: string | null
  transcript: string | null
  sampleSeconds: number | null
}

export interface GenParams {
  temperature: number
  topP: number
  repetitionPenalty: number
  seed: number
}

export interface HistoryEntry {
  id: string
  text: string
  voiceId: string | null
  voiceName: string
  model: string
  durationSec: number
  fileSize: number
  wavRel: string
  wavAbs: string
  createdAt: number
  params: GenParams
}

export interface ModelStatus {
  model: string
  displayName: string
  installed: boolean
  path: string | null
  missingFiles: string[]
  downloading: boolean
  /** true when this is the active engine */
  active: boolean
}

export interface EngineStatus {
  ready: boolean
  model: string
  modelDir: string | null
  device: string
}

export interface DeviceProbe {
  metal: boolean
  cuda: boolean
}

export interface StorageStats {
  audioBytes: number
  voicesBytes: number
  modelsBytes: number
}

/** serde tag="stage", snake_case — mirrors tts-common engine::Progress */
export type GenProgress =
  | { stage: 'loading_lm' }
  | { stage: 'loading_codec' }
  | { stage: 'encoding_ref'; seconds: number }
  | { stage: 'chunks'; total: number }
  | { stage: 'prefill'; chunk: number; total: number; tokens: number }
  | { stage: 'generating'; chunk: number; total: number; frames: number; max_frames: number; fps: number }
  | { stage: 'unmasking'; chunk: number; total: number; step: number; total_steps: number }
  | { stage: 'decoding'; chunk: number; total: number; frames_total: number }
  | { stage: 'writing_wav' }
  | { stage: 'done'; frames: number; seconds: number }

export interface DownloadEvent {
  model: string
  file: string
  fileIndex: number
  fileCount: number
  downloaded: number
  total: number
  speedBps: number
  done: boolean
  error: string | null
}

export interface GenerateReq {
  text: string
  voiceId: string | null
  /** OmniVoice style instruction */
  instruct: string | null
  seed: number
}

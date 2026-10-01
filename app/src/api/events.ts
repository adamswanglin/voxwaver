import type { DownloadEvent, GenProgress, HistoryEntry } from '../types'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'

export function onGenProgress(cb: (p: GenProgress) => void): Promise<UnlistenFn> {
  return listen<GenProgress>('gen://progress', (e) => cb(e.payload))
}

export function onGenLog(cb: (msg: string) => void): Promise<UnlistenFn> {
  return listen<string>('gen://log', (e) => cb(e.payload))
}

export function onGenError(
  cb: (err: { message: string; cancelled: boolean }) => void,
): Promise<UnlistenFn> {
  return listen<{ message: string; cancelled: boolean }>('gen://error', (e) => cb(e.payload))
}

export function onGenDone(cb: (entry: HistoryEntry) => void): Promise<UnlistenFn> {
  return listen<HistoryEntry>('gen://done', (e) => cb(e.payload))
}

export function onModelDownload(cb: (e: DownloadEvent) => void): Promise<UnlistenFn> {
  return listen<DownloadEvent>('model://download', (e) => cb(e.payload))
}

export function onVoiceEncoding(cb: (msg: string) => void): Promise<UnlistenFn> {
  return listen<string>('voice://encoding', (e) => cb(e.payload))
}

import { convertFileSrc } from '@tauri-apps/api/core'
import { mediaObjectUrl, setActivePlaybackSrc } from '../lib/media'
import type { HistoryEntry } from '../types'
import { create } from 'zustand'
import { toast } from './toast'

interface PlayerStore {
  entry: HistoryEntry | null
  playing: boolean
  position: number // seconds
  duration: number
  /** 生成完成、尚未播放的提示标记；开始播放后清除 */
  doneNotice: boolean
  /** 0..1 playhead */
  seek: (fraction: number) => void
  play: (entry: HistoryEntry) => void
  /** 载入条目但不自动播放（生成完成后调用） */
  load: (entry: HistoryEntry) => void
  playPause: () => void
  skip: (deltaSec: number) => void
}

let audio: HTMLAudioElement | null = null
let wired = false

function ensureAudio(): HTMLAudioElement {
  if (audio) return audio
  audio = new Audio()
  audio.addEventListener('timeupdate', () => {
    const a = audio!
    usePlayer.setState({
      position: a.currentTime,
      duration: Number.isFinite(a.duration) ? a.duration : usePlayer.getState().entry?.durationSec ?? 0,
    })
  })
  audio.addEventListener('ended', () => usePlayer.setState({ playing: false }))
  audio.addEventListener('error', () => {
    const a = audio!
    const code = a.error?.code
    const kind =
      code === 1 ? 'aborted' : code === 2 ? 'network' : code === 3 ? 'decode' : code === 4 ? 'not-supported' : String(code)
    console.error('audio error', kind, a.error?.message, a.src)
    toast.error(`音频加载失败（${kind}）：${a.error?.message ?? a.src}`)
  })
  audio.addEventListener('play', () => usePlayer.setState({ playing: true, doneNotice: false }))
  audio.addEventListener('pause', () => usePlayer.setState({ playing: false }))
  wired = true
  return audio
}

export function playerSrc(entry: HistoryEntry): string {
  // Fetch-friendly media:// URL (waveform decoding). NOT for the <audio>
  // element directly — see lib/media.ts.
  return convertFileSrc(entry.wavAbs, 'media')
}

function reportPlayError(e: unknown) {
  console.error('play() rejected', e)
  // NotAllowedError: the fetch before play() dropped the user-gesture
  // context. Stay silent — the blob is cached now, so the next playPause
  // click plays instantly from inside a real gesture.
  if ((e as DOMException)?.name === 'NotAllowedError') return
  toast.error(`播放失败：${(e as DOMException)?.name ?? ''} ${(e as DOMException)?.message ?? e}`)
}

async function startEntry(a: HTMLAudioElement, entry: HistoryEntry, autoplay: boolean) {
  try {
    const url = await mediaObjectUrl(entry.wavAbs)
    if (usePlayer.getState().entry?.id !== entry.id) return // switched away while loading
    setActivePlaybackSrc(url)
    a.src = url
  } catch (e) {
    console.error('load audio failed', e)
    toast.error(`音频加载失败：${e}`)
    return
  }
  if (autoplay) a.play().catch(reportPlayError)
}

export const usePlayer = create<PlayerStore>((set, get) => ({
  entry: null,
  playing: false,
  position: 0,
  duration: 0,
  doneNotice: false,

  play: (entry) => {
    const a = ensureAudio()
    if (get().entry?.id !== entry.id) {
      set({ entry, position: 0, duration: entry.durationSec })
      void startEntry(a, entry, true)
    } else {
      a.play().catch(reportPlayError)
    }
  },

  load: (entry) => {
    const a = ensureAudio()
    set({ doneNotice: true })
    if (get().entry?.id !== entry.id) {
      set({ entry, position: 0, duration: entry.durationSec })
      void startEntry(a, entry, false)
    }
  },

  playPause: () => {
    const a = ensureAudio()
    if (!get().entry) return
    if (a.paused) a.play().catch(reportPlayError)
    else a.pause()
  },

  seek: (fraction) => {
    const a = ensureAudio()
    const d = a.duration || get().entry?.durationSec || 0
    if (Number.isFinite(d)) {
      a.currentTime = Math.max(0, Math.min(1, fraction)) * d
      set({ position: a.currentTime })
    }
  },

  skip: (delta) => {
    const a = ensureAudio()
    a.currentTime = Math.max(0, a.currentTime + delta)
    set({ position: a.currentTime })
  },
}))

export function playerReady(): boolean {
  return wired
}

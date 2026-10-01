import type { GenerateReq, GenProgress } from '../types'
import { apiCancelGenerate, apiGenerate } from '../api'
import { onGenError, onGenLog, onGenProgress } from '../api/events'
import { create } from 'zustand'
import { useHistory } from './history'
import { usePlayer } from './player'
import { useView } from './view'
import { toast } from './toast'

/** One colored segment of the total progress bar. */
export interface GenSegment {
  key: 'load' | 'encode' | 'gen' | 'decode' | 'write'
  label: string
  color: string
  /** relative width within the bar (all weights sum to 1) */
  weight: number
  /** fill of this segment, 0..1 */
  fraction: number
}

/** Pipeline phases in bar order; weights sum to 1. */
const SEGMENT_DEFS: Omit<GenSegment, 'fraction'>[] = [
  { key: 'load', label: '加载模型', color: '#8b5cf6', weight: 0.05 },
  { key: 'encode', label: '编码参考', color: '#06b6d4', weight: 0.05 },
  { key: 'gen', label: '生成语音', color: '#1b61c9', weight: 0.68 },
  { key: 'decode', label: '音频解码', color: '#f59e0b', weight: 0.17 },
  { key: 'write', label: '写入文件', color: '#10b981', weight: 0.05 },
]

/** Which segment each progress event belongs to. */
const PHASE_OF: Record<GenProgress['stage'], GenSegment['key']> = {
  loading_lm: 'load',
  loading_codec: 'load',
  encoding_ref: 'encode',
  chunks: 'encode',
  prefill: 'gen',
  generating: 'gen',
  unmasking: 'gen',
  decoding: 'decode',
  writing_wav: 'write',
  done: 'write',
}

const freshSegments = (): GenSegment[] =>
  SEGMENT_DEFS.map((d) => ({ ...d, fraction: 0 }))

/** Field defaults so an older backend payload (missing `chunk`/`total`)
 * degrades to "single chunk" instead of NaN. */
const chunkOf = (p: { chunk?: number }): number => p.chunk ?? 1
const totalOf = (p: { total?: number }): number => Math.max(1, p.total ?? 1)

/**
 * Progress within a phase, 0..1. Chunked phases (gen/decode) advance across
 * all chunks — `chunk` is 1-based — so the bar never restarts at 0 per chunk.
 */
function phaseFraction(p: GenProgress): number {
  switch (p.stage) {
    case 'loading_lm':
      return 0.4
    case 'loading_codec':
      return 0.95
    case 'encoding_ref':
      return 0.6
    case 'chunks':
      return 1
    case 'prefill':
      return (chunkOf(p) - 1) / totalOf(p)
    case 'generating':
      return (chunkOf(p) - 1 + p.frames / Math.max(1, p.max_frames)) / totalOf(p)
    case 'unmasking':
      return (chunkOf(p) - 1 + p.step / Math.max(1, p.total_steps)) / totalOf(p)
    case 'decoding':
      // emitted before each chunk's DAC decode; count it as half-done
      return (chunkOf(p) - 0.5) / totalOf(p)
    case 'writing_wav':
      return 0.5
    case 'done':
      return 1
  }
}

/** Apply an event: fill the phase's segment (monotonic) and auto-complete
 * every earlier segment (e.g. model load skipped because it is resident). */
function applyToSegments(segs: GenSegment[], p: GenProgress): GenSegment[] {
  const idx = SEGMENT_DEFS.findIndex((d) => d.key === PHASE_OF[p.stage])
  const f = phaseFraction(p)
  return segs.map((s, i) => ({
    ...s,
    fraction: i < idx || p.stage === 'done' ? 1 : i === idx ? Math.max(s.fraction, f) : s.fraction,
  }))
}

const overallFraction = (segs: GenSegment[]): number =>
  segs.reduce((acc, s) => acc + s.weight * s.fraction, 0)

export interface GenState {
  running: boolean
  /** cancel requested; waiting for the backend loop to notice */
  cancelling: boolean
  /** 0..1 across the whole pipeline */
  fraction: number
  /** colored per-phase segments of the total progress bar */
  segments: GenSegment[]
  stageText: string
  subText: string
}

interface GenerationStore extends GenState {
  run: (req: GenerateReq) => Promise<boolean>
  cancel: () => void
}

let wired = false
function wireEvents() {
  if (wired) return
  wired = true
  onGenProgress((p) => {
    const segments = applyToSegments(useGeneration.getState().segments, p)
    const patch: Partial<GenState> = { segments, fraction: overallFraction(segments) }
    Object.assign(patch, stageText(p))
    useGeneration.setState(patch)
  })
  onGenLog(() => {
    /* logs are diagnostics; the UI renders stage text */
  })
  onGenError(({ cancelled }) => {
    if (cancelled) {
      useGeneration.setState({
        running: false,
        cancelling: false,
        fraction: 0,
        segments: freshSegments(),
        stageText: '',
        subText: '',
      })
      toast.info('已取消生成')
    }
  })
}

function stageText(p: GenProgress): Partial<GenState> {
  switch (p.stage) {
    case 'loading_lm':
      return { stageText: '正在加载语言模型…', subText: '首次加载约需数秒' }
    case 'loading_codec':
      return { stageText: '正在加载编解码器…', subText: '' }
    case 'encoding_ref':
      return { stageText: '正在编码参考音频…', subText: `${p.seconds.toFixed(1)}s 样本` }
    case 'chunks':
      return { stageText: '文本分块完成', subText: `共 ${p.total} 块` }
    case 'prefill':
      return {
        stageText: `正在预填充 (${chunkOf(p)}/${totalOf(p)})`,
        subText: `${p.tokens} tokens`,
      }
    case 'generating':
      return {
        stageText: `正在生成语音 (${chunkOf(p)}/${totalOf(p)})`,
        subText: `${p.frames}/${p.max_frames} 帧 · ${p.fps.toFixed(1)} 帧/秒`,
      }
    case 'unmasking':
      return {
        stageText: `正在生成语音 (${chunkOf(p)}/${totalOf(p)})`,
        subText: `迭代解码 ${p.step + 1}/${p.total_steps} 步`,
      }
    case 'decoding':
      return {
        stageText: `正在解码音频 (${chunkOf(p)}/${totalOf(p)})`,
        subText: `${p.frames_total} 帧`,
      }
    case 'writing_wav':
      return { stageText: '正在写入音频文件…', subText: '' }
    case 'done':
      return { stageText: '完成', subText: `${p.seconds.toFixed(1)}s 音频` }
  }
}

export const useGeneration = create<GenerationStore>((set) => ({
  running: false,
  cancelling: false,
  fraction: 0,
  segments: freshSegments(),
  stageText: '',
  subText: '',

  run: async (req) => {
    wireEvents()
    set({
      running: true,
      cancelling: false,
      fraction: 0,
      segments: freshSegments(),
      stageText: '准备中…',
      subText: '',
    })
    try {
      const entry = await apiGenerate(req)
      set({ running: false, cancelling: false, fraction: 1, stageText: '', subText: '' })
      toast.success('语音生成完成')
      await useHistory.getState().load()
      usePlayer.getState().play(entry)
      return true
    } catch (e) {
      set({
        running: false,
        cancelling: false,
        fraction: 0,
        segments: freshSegments(),
        stageText: '',
        subText: '',
      })
      const msg = String(e)
      if (!msg.includes('取消') && !msg.includes('cancelled')) toast.error(`生成失败：${msg}`)
      return false
    }
  },

  cancel: () => {
    apiCancelGenerate()
    set({ cancelling: true, stageText: '正在取消…' })
  },
}))

// keep view store referenced for auto-switch convenience
void useView

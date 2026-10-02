import type { GenerateReq, GenProgress } from '../types'
import { apiCancelGenerate, apiGenerate } from '../api'
import { onGenError, onGenLog, onGenProgress } from '../api/events'
import i18n from '../i18n'
import { create } from 'zustand'
import { useHistory } from './history'
import { usePlayer } from './player'
import { useView } from './view'
import { toast } from './toast'

/** One colored segment of the total progress bar. */
export interface GenSegment {
  key: 'load' | 'encode' | 'gen' | 'decode' | 'write'
  /** i18n key, resolved at render time */
  labelKey: string
  color: string
  /** fill of this segment, 0..1 */
  fraction: number
}

/** Pipeline phases in bar order; every segment renders equal width. */
const SEGMENT_DEFS: Omit<GenSegment, 'fraction'>[] = [
  { key: 'load', labelKey: 'gen.seg.load', color: '#8b5cf6' },
  { key: 'encode', labelKey: 'gen.seg.encode', color: '#06b6d4' },
  { key: 'gen', labelKey: 'gen.seg.gen', color: '#1b61c9' },
  { key: 'decode', labelKey: 'gen.seg.decode', color: '#f59e0b' },
  { key: 'write', labelKey: 'gen.seg.write', color: '#10b981' },
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

export interface GenState {
  running: boolean
  /** cancel requested; waiting for the backend loop to notice */
  cancelling: boolean
  /** colored per-phase segments of the progress bar */
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
    const patch: Partial<GenState> = { segments }
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
        segments: freshSegments(),
        stageText: '',
        subText: '',
      })
      toast.info(i18n.t('toasts.genCancelled'))
    }
  })
}

function stageText(p: GenProgress): Partial<GenState> {
  const t = i18n.t.bind(i18n)
  switch (p.stage) {
    case 'loading_lm':
      return { stageText: t('gen.loadLm'), subText: t('gen.loadLmSub') }
    case 'loading_codec':
      return { stageText: t('gen.loadCodec'), subText: '' }
    case 'encoding_ref':
      return { stageText: t('gen.encodeRef'), subText: t('gen.encodeRefSub', { sec: p.seconds.toFixed(1) }) }
    case 'chunks':
      return { stageText: t('gen.chunksDone'), subText: t('gen.chunksSub', { n: p.total }) }
    case 'prefill':
      return {
        stageText: t('gen.prefill', { chunk: chunkOf(p), total: totalOf(p) }),
        subText: t('gen.prefillSub', { n: p.tokens }),
      }
    case 'generating':
      return {
        stageText: t('gen.generating', { chunk: chunkOf(p), total: totalOf(p) }),
        subText: t('gen.generatingSub', { frames: p.frames, max: p.max_frames, fps: p.fps.toFixed(1) }),
      }
    case 'unmasking':
      return {
        stageText: t('gen.generating', { chunk: chunkOf(p), total: totalOf(p) }),
        subText: t('gen.unmaskSub', { step: p.step + 1, total: p.total_steps }),
      }
    case 'decoding':
      return {
        stageText: t('gen.decoding', { chunk: chunkOf(p), total: totalOf(p) }),
        subText: t('gen.decodingSub', { n: p.frames_total }),
      }
    case 'writing_wav':
      return { stageText: t('gen.writing'), subText: '' }
    case 'done':
      return { stageText: t('gen.done'), subText: t('gen.doneSub', { sec: p.seconds.toFixed(1) }) }
  }
}

export const useGeneration = create<GenerationStore>((set) => ({
  running: false,
  cancelling: false,
  segments: freshSegments(),
  stageText: '',
  subText: '',

  run: async (req) => {
    wireEvents()
    set({
      running: true,
      cancelling: false,
      segments: freshSegments(),
      stageText: i18n.t('gen.prepare'),
      subText: '',
    })
    try {
      const entry = await apiGenerate(req)
      set({ running: false, cancelling: false, stageText: '', subText: '' })
      await useHistory.getState().load()
      // 只载入不自动播放；底部播放条显示「生成完成」提示
      usePlayer.getState().load(entry)
      return true
    } catch (e) {
      set({
        running: false,
        cancelling: false,
        segments: freshSegments(),
        stageText: '',
        subText: '',
      })
      const msg = String(e)
      if (!msg.toLowerCase().includes('cancelled')) toast.error(i18n.t('toasts.genFail', { msg }))
      return false
    }
  },

  cancel: () => {
    apiCancelGenerate()
    set({ cancelling: true, stageText: i18n.t('gen.cancelling') })
  },
}))

// keep view store referenced for auto-switch convenience
void useView

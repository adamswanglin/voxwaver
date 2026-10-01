import { useEffect, useMemo, useState } from 'react'
import { computePeaks } from '../../lib/wav'

interface WaveformProps {
  url: string
  /** playback fraction 0..1 */
  progress: number
  onSeek?: (fraction: number) => void
  height?: number
  buckets?: number
}

const BAR_W = 4
const BAR_GAP = 2

/** Real waveform rendered from decoded peaks; bars before the playhead are
 *  highlighted, with a playhead line. */
export function Waveform({ url, progress, onSeek, height = 32, buckets = 140 }: WaveformProps) {
  const [peaks, setPeaks] = useState<number[] | null>(null)

  useEffect(() => {
    let cancelled = false
    setPeaks(null)
    computePeaks(url, buckets)
      .then((p) => !cancelled && setPeaks(p))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [url, buckets])

  const vbW = (buckets - 1) * (BAR_W + BAR_GAP) + BAR_W

  const bars = useMemo(() => {
    if (!peaks) return null
    const minH = height * 0.08
    return peaks.map((p) => Math.max(minH, p * height * 0.92))
  }, [peaks, height])

  return (
    <div
      className="waveform-container"
      style={{ height }}
      onClick={(e) => {
        if (!onSeek) return
        const rect = e.currentTarget.getBoundingClientRect()
        onSeek((e.clientX - rect.left) / rect.width)
      }}
    >
      <svg viewBox={`0 0 ${vbW} ${height}`} preserveAspectRatio="none">
        {bars?.map((h, i) => {
          const x = i * (BAR_W + BAR_GAP)
          const played = i / buckets <= progress
          return (
            <rect
              key={i}
              x={x}
              y={(height - h) / 2}
              width={BAR_W}
              height={h}
              rx={1.5}
              fill="var(--seed-primary)"
              opacity={played ? 0.75 : 0.25}
            />
          )
        })}
        {peaks && (
          <line
            x1={progress * vbW}
            y1={0}
            x2={progress * vbW}
            y2={height}
            stroke="var(--seed-primary)"
            strokeWidth="2"
            strokeLinecap="round"
          />
        )}
      </svg>
    </div>
  )
}

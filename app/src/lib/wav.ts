/** Waveform helpers: decode a wav URL into downsampled peaks. */

const peaksCache = new Map<string, number[]>()

let audioCtx: AudioContext | null = null

export async function computePeaks(url: string, buckets = 600): Promise<number[]> {
  const cached = peaksCache.get(url)
  if (cached) return cached
  const res = await fetch(url)
  const buf = await res.arrayBuffer()
  audioCtx ??= new AudioContext()
  const audio = await audioCtx.decodeAudioData(buf)
  const data = audio.getChannelData(0)
  const peaks: number[] = []
  const step = Math.floor(data.length / buckets) || 1
  for (let i = 0; i < buckets; i++) {
    let max = 0
    const start = i * step
    for (let j = start; j < start + step && j < data.length; j += 4) {
      const v = Math.abs(data[j])
      if (v > max) max = v
    }
    peaks.push(max)
  }
  // normalize to [0, 1]
  const top = Math.max(...peaks, 0.01)
  const norm = peaks.map((p) => p / top)
  peaksCache.set(url, norm)
  return norm
}

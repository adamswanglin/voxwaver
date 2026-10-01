/** Deterministic pastel avatar SVG from a voice id/name. */

const PALETTES: Array<[string, string]> = [
  ['#DBEAFE', '#3B82F6'],
  ['#FEF3C7', '#D97706'],
  ['#D1FAE5', '#059669'],
  ['#E8D5F5', '#8B5CF6'],
  ['#FCE7F3', '#EC4899'],
  ['#E0E7FF', '#4F46E5'],
]

function hash(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

export function VoiceAvatar({ name, size = 40 }: { name: string; size?: number }) {
  const [bg, fg] = PALETTES[hash(name) % PALETTES.length]
  const r = size / 2
  const headR = size * 0.16
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={r} cy={r} r={r} fill={bg} />
      <circle cx={r} cy={r * 0.78} r={headR} fill={fg} />
      <ellipse cx={r} cy={r * 1.65} rx={headR * 1.75} ry={headR * 1.3} fill={fg} />
    </svg>
  )
}

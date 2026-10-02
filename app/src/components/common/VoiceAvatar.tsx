/** Voice avatar: chosen emoji icon, or the first character of the name on a
 * deterministic pastel background (hash of the name → stable "random" color). */

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

export function VoiceAvatar({
  name,
  icon,
  size = 40,
}: {
  name: string
  icon?: string | null
  size?: number
}) {
  const [bg, fg] = PALETTES[hash(name) % PALETTES.length]
  const r = size / 2
  if (icon) {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={r} cy={r} r={r} fill={bg} />
        <text
          x={r}
          y={r}
          fill={fg}
          fontSize={size * 0.5}
          dominantBaseline="central"
          textAnchor="middle"
        >
          {icon}
        </text>
      </svg>
    )
  }
  // first grapheme (works for CJK, latin and surrogate pairs alike)
  const ch = Array.from(name.trim())[0] ?? '?'
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={r} cy={r} r={r} fill={bg} />
      <text
        x={r}
        y={r}
        fill={fg}
        fontSize={size * 0.45}
        fontWeight={600}
        dominantBaseline="central"
        textAnchor="middle"
      >
        {ch}
      </text>
    </svg>
  )
}

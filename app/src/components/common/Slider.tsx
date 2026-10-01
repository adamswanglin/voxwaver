import { useRef } from 'react'

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  format: (v: number) => string
  onChange: (v: number) => void
  disabled?: boolean
  hint?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
  hint,
}: SliderProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const pct = ((value - min) / (max - min)) * 100

  const setFromEvent = (clientX: number) => {
    const el = trackRef.current
    if (!el || disabled) return
    const rect = el.getBoundingClientRect()
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const raw = min + frac * (max - min)
    const snapped = Math.round(raw / step) * step
    onChange(Math.max(min, Math.min(max, Number(snapped.toFixed(4)))))
  }

  return (
    <div className={`slider-group ${disabled ? 'disabled' : ''}`}>
      <div className="slider-label">
        <span>{label}</span>
        <span className="slider-value">{format(value)}</span>
      </div>
      <div
        className="slider-track"
        role="slider"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
        ref={trackRef}
        onPointerDown={(e) => {
          ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          setFromEvent(e.clientX)
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) setFromEvent(e.clientX)
        }}
      >
        <div className="slider-fill" style={{ width: `${pct}%` }} />
        <div className="slider-thumb" style={{ left: `${pct}%` }} />
      </div>
      {hint && <div className="slider-hint">{hint}</div>}
    </div>
  )
}

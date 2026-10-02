import { useEffect, useMemo, useRef, useState } from 'react'
import { OMNIVOICE_LANGUAGES } from '../../data/omnivoiceLanguages'

interface Props {
  /** '' = auto (no lang tag passed to the model) */
  value: string
  onChange: (v: string) => void
  disabled?: boolean
  /** i18n label for the auto ('') entry */
  autoLabel: string
  placeholder: string
  id?: string
}

/**
 * Filterable language picker over the 646-entry OmniVoice table.
 * Type to filter by (English) name or id; ids are matched case-insensitively.
 */
export function LangCombobox({ value, onChange, disabled, autoLabel, placeholder, id }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const selectedName =
    value === '' ? '' : OMNIVOICE_LANGUAGES.find((l) => l.id === value)?.name ?? value

  const items = useMemo(() => {
    const q = query.trim().toLowerCase()
    const head: { id: string; name: string }[] = []
    if (!q || autoLabel.toLowerCase().includes(q)) head.push({ id: '', name: autoLabel })
    // legacy saved id not in the model's language table (e.g. the old 'ar')
    if (value && value !== '' && !OMNIVOICE_LANGUAGES.some((l) => l.id === value)) {
      if (!q || value.toLowerCase().includes(q)) head.push({ id: value, name: value })
    }
    const tail = q
      ? OMNIVOICE_LANGUAGES.filter(
          (l) => l.name.toLowerCase().includes(q) || l.id.toLowerCase().includes(q),
        )
      : OMNIVOICE_LANGUAGES
    return [...head, ...tail]
  }, [query, value, autoLabel])

  // close on outside click
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  // keep the keyboard-active option in view
  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const openList = () => {
    setOpen(true)
    setQuery('')
    // items still reflects the old query here; recompute the empty-query
    // list ([auto, ...all]) to place the highlight on the current value
    const idx = OMNIVOICE_LANGUAGES.findIndex((l) => l.id === value)
    setActive(idx >= 0 ? idx + 1 : 0)
    requestAnimationFrame(() => inputRef.current?.select())
  }

  const pick = (id: string) => {
    onChange(id)
    setOpen(false)
    inputRef.current?.blur()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) openList()
      else setActive((i) => Math.min(i + 1, items.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && open) {
      e.preventDefault()
      const item = items[active]
      if (item) pick(item.id)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="lang-combobox" ref={rootRef}>
      <input
        ref={inputRef}
        id={id}
        className="form-input"
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        disabled={disabled}
        placeholder={placeholder}
        /* closed: selected language ('auto' when unset); open: filter text */
        value={open ? query : value === '' ? autoLabel : selectedName}
        onChange={(e) => {
          setQuery(e.target.value)
          setActive(0)
          if (!open) setOpen(true)
        }}
        onFocus={openList}
        onKeyDown={onKeyDown}
      />
      {open && (
        <div className="lang-dropdown" role="listbox" ref={listRef}>
          {items.length === 0 && <div className="lang-empty">—</div>}
          {items.map((l, i) => (
            <button
              key={l.id === '' ? '__auto' : l.id}
              type="button"
              role="option"
              aria-selected={l.id === value}
              data-index={i}
              className={`lang-option ${i === active ? 'active' : ''} ${l.id === value ? 'selected' : ''}`}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick(l.id)}
            >
              <span className="lang-option-name">{l.name}</span>
              {l.id !== '' && <span className="lang-option-id">{l.id}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

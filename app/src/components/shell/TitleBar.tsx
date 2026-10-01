import { IconSearch } from '../Icons'

export function TitleBar() {
  return (
    <header className="titlebar" data-tauri-drag-region>
      <div className="titlebar-center" data-tauri-drag-region>
        VoxWeaver
      </div>
      <div className="titlebar-actions">
        <div className="search-pill" role="search" aria-label="全局搜索">
          <IconSearch />
          <span>搜索声音、配置…</span>
          <kbd>⌘K</kbd>
        </div>
      </div>
    </header>
  )
}

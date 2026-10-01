import type { HistoryEntry } from '../types'
import { apiDeleteHistory, apiExportAudio, apiListHistory } from '../api'
import { create } from 'zustand'
import { toast } from './toast'

interface HistoryStore {
  entries: HistoryEntry[]
  selected: Set<string>
  load: () => Promise<void>
  remove: (ids: string[]) => Promise<void>
  toggle: (id: string) => void
  clearSelection: () => void
  exportSelected: (destDir: string) => Promise<void>
}

export const useHistory = create<HistoryStore>((set, get) => ({
  entries: [],
  selected: new Set(),

  load: async () => {
    const entries = await apiListHistory()
    set({ entries })
  },

  remove: async (ids) => {
    await apiDeleteHistory(ids)
    const sel = new Set(get().selected)
    ids.forEach((id) => sel.delete(id))
    set({ selected: sel })
    await get().load()
    toast.info(`已删除 ${ids.length} 条记录`)
  },

  toggle: (id) => {
    const sel = new Set(get().selected)
    if (sel.has(id)) sel.delete(id)
    else sel.add(id)
    set({ selected: sel })
  },

  clearSelection: () => set({ selected: new Set() }),

  exportSelected: async (destDir) => {
    const ids = [...get().selected]
    if (!ids.length) return
    try {
      const n = await apiExportAudio(ids, destDir)
      toast.success(`已导出 ${n} 个音频文件`)
      get().clearSelection()
    } catch (e) {
      toast.error(`导出失败：${e}`)
    }
  },
}))

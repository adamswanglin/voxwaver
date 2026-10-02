import type { HistoryEntry } from '../types'
import { apiDeleteHistory, apiExportAudio, apiListHistory, apiSearchHistory } from '../api'
import i18n from '../i18n'
import { create } from 'zustand'
import { toast } from './toast'

interface HistoryStore {
  entries: HistoryEntry[]
  selected: Set<string>
  load: () => Promise<void>
  /** Empty query = full list; otherwise grep-filtered by the backend. */
  search: (query: string) => Promise<void>
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

  search: async (query) => {
    const entries = query ? await apiSearchHistory(query) : await apiListHistory()
    set({ entries })
  },

  remove: async (ids) => {
    await apiDeleteHistory(ids)
    const sel = new Set(get().selected)
    ids.forEach((id) => sel.delete(id))
    set({ selected: sel })
    await get().load()
    toast.info(i18n.t('toasts.historyDeleted', { n: ids.length }))
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
      toast.success(i18n.t('toasts.exported', { n }))
      get().clearSelection()
    } catch (e) {
      toast.error(i18n.t('toasts.exportFail', { msg: String(e) }))
    }
  },
}))

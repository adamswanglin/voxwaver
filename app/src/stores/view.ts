import type { ViewName, VoiceView } from '../types'
import { create } from 'zustand'

interface ViewStore {
  view: ViewName
  settingsOpen: boolean
  cloneOpen: boolean
  /** voice being edited in the clone modal (null = create new) */
  editingVoice: VoiceView | null
  setView: (v: ViewName) => void
  setSettingsOpen: (open: boolean) => void
  setCloneOpen: (open: boolean) => void
  openClone: (voice?: VoiceView | null) => void
}

export const useView = create<ViewStore>((set) => ({
  view: 'workspace',
  settingsOpen: false,
  cloneOpen: false,
  editingVoice: null,
  setView: (view) => set({ view }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setCloneOpen: (cloneOpen) => set({ cloneOpen, editingVoice: null }),
  openClone: (voice = null) => set({ cloneOpen: true, editingVoice: voice }),
}))

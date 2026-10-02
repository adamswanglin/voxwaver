import type { VoiceView } from '../types'
import {
  apiCreateVoice,
  apiDeleteVoice,
  apiListVoices,
  apiUpdateVoice,
  type CreateVoiceReq,
  type UpdateVoiceReq,
} from '../api'
import i18n from '../i18n'
import { create } from 'zustand'
import { toast } from './toast'

interface VoicesStore {
  voices: VoiceView[]
  /** currently selected voice in the workspace (null = no voice selected) */
  currentId: string | null
  load: () => Promise<void>
  setCurrent: (id: string | null) => void
  create: (req: CreateVoiceReq) => Promise<boolean>
  update: (req: UpdateVoiceReq) => Promise<boolean>
  remove: (id: string) => Promise<void>
}

export const useVoices = create<VoicesStore>((set, get) => ({
  voices: [],
  currentId: null,

  load: async () => {
    const voices = await apiListVoices()
    set({ voices })
  },

  setCurrent: (currentId) => set({ currentId }),

  create: async (req) => {
    try {
      const v = await apiCreateVoice(req)
      await get().load()
      toast.success(i18n.t('toasts.voiceCreated', { name: v.name }))
      return true
    } catch (e) {
      toast.error(i18n.t('toasts.voiceCreateFail', { msg: String(e) }))
      return false
    }
  },

  update: async (req) => {
    try {
      await apiUpdateVoice(req)
      await get().load()
      toast.success(i18n.t('toasts.voiceUpdated', { name: req.name }))
      return true
    } catch (e) {
      toast.error(i18n.t('toasts.voiceUpdateFail', { msg: String(e) }))
      return false
    }
  },

  remove: async (id) => {
    try {
      await apiDeleteVoice(id)
      if (get().currentId === id) set({ currentId: null })
      await get().load()
      toast.info(i18n.t('toasts.voiceDeleted'))
    } catch (e) {
      toast.error(String(e))
    }
  },
}))

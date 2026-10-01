import type { VoiceView } from '../types'
import {
  apiCreateVoice,
  apiDeleteVoice,
  apiListVoices,
  apiUpdateVoice,
  type CreateVoiceReq,
  type UpdateVoiceReq,
} from '../api'
import { create } from 'zustand'
import { toast } from './toast'

interface VoicesStore {
  voices: VoiceView[]
  /** currently selected voice in the workspace (null = default) */
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
      toast.success(`声音「${v.name}」创建成功`)
      return true
    } catch (e) {
      toast.error(`创建失败：${e}`)
      return false
    }
  },

  update: async (req) => {
    try {
      await apiUpdateVoice(req)
      await get().load()
      toast.success(`声音「${req.name}」已更新`)
      return true
    } catch (e) {
      toast.error(`保存失败：${e}`)
      return false
    }
  },

  remove: async (id) => {
    try {
      await apiDeleteVoice(id)
      if (get().currentId === id) set({ currentId: null })
      await get().load()
      toast.info('声音已删除')
    } catch (e) {
      toast.error(String(e))
    }
  },
}))

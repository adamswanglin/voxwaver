import { create } from 'zustand'

export interface Toast {
  id: number
  kind: 'info' | 'error' | 'success'
  message: string
}

interface ToastStore {
  toasts: Toast[]
  push: (kind: Toast['kind'], message: string) => void
  dismiss: (id: number) => void
}

let nextId = 1

export const useToast = create<ToastStore>((set, get) => ({
  toasts: [],
  push: (kind, message) => {
    const id = nextId++
    set({ toasts: [...get().toasts, { id, kind, message }] })
    setTimeout(() => get().dismiss(id), kind === 'error' ? 6000 : 3200)
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
}))

export const toast = {
  info: (m: string) => useToast.getState().push('info', m),
  error: (m: string) => useToast.getState().push('error', m),
  success: (m: string) => useToast.getState().push('success', m),
}

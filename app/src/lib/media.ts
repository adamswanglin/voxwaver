import { convertFileSrc } from '@tauri-apps/api/core'

/** WKWebView's `<audio>` media stack (AVFoundation) refuses resources served
 *  over custom URI schemes (`NotSupportedError: The operation is not
 *  supported.`) even though `fetch()` and `decodeAudioData()` work fine on the
 *  very same `media://` URL — see tauri-apps/tauri#4826. Fetching the bytes
 *  and handing the element a `blob:` URL, which it does support, sidesteps
 *  the whole class of scheme problems. Object URLs are cached per path. */

const objectUrls = new Map<string, string>()
const MAX_CACHED = 16

export async function mediaObjectUrl(absPath: string, fallbackMime = 'audio/wav'): Promise<string> {
  const cached = objectUrls.get(absPath)
  if (cached) return cached

  const res = await fetch(convertFileSrc(absPath, 'media'))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const type = res.headers.get('content-type') ?? fallbackMime
  const blob = new Blob([await res.arrayBuffer()], { type })
  const url = URL.createObjectURL(blob)

  // Evict oldest entries, but never the URL the shared <audio> is playing.
  while (objectUrls.size >= MAX_CACHED) {
    const oldest = objectUrls.keys().next().value
    if (oldest === undefined) break
    const stale = objectUrls.get(oldest)!
    objectUrls.delete(oldest)
    if (stale !== activePlaybackSrc) URL.revokeObjectURL(stale)
  }
  objectUrls.set(absPath, url)
  return url
}

/** Set by the player store so eviction never revokes the playing source. */
let activePlaybackSrc: string | null = null

export function setActivePlaybackSrc(url: string | null) {
  activePlaybackSrc = url
}

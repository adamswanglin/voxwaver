// Generates a simple brand icon (rounded square + waveform) as a PNG,
// used as input for `tauri icon`. Pure Node (zlib), no image deps.
import zlib from 'node:zlib'
import fs from 'node:fs'

const SIZE = 1024
const BG = [27, 97, 201] // --seed-primary #1b61c9
const FG = [255, 255, 255]

// waveform heights (normalized), mirrors the design logo path
const wave = [0.35, 0.55, 0.42, 0.75, 0.5, 0.85, 0.4]
const bars = 7
const barW = Math.round(SIZE * 0.052)
const gap = Math.round(SIZE * 0.032)
const totalW = bars * barW + (bars - 1) * gap
const x0 = Math.round((SIZE - totalW) / 2)

const radius = Math.round(SIZE * 0.22) // ~ app icon squircle
const inRounded = (x, y) => {
  const r = radius
  const cx = Math.min(Math.max(x, r), SIZE - r)
  const cy = Math.min(Math.max(y, r), SIZE - r)
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
}

const raw = Buffer.alloc(SIZE * (SIZE * 4 + 1))
for (let y = 0; y < SIZE; y++) {
  const row = y * (SIZE * 4 + 1)
  raw[row] = 0 // filter: none
  for (let x = 0; x < SIZE; x++) {
    const o = row + 1 + x * 4
    let c = null
    if (inRounded(x + 0.5, y + 0.5)) {
      c = BG
      // bar test
      const i = Math.floor((x - x0) / (barW + gap))
      if (i >= 0 && i < bars) {
        const bx = x0 + i * (barW + gap)
        if (x >= bx && x < bx + barW) {
          const h = wave[i] * SIZE * 0.42
          const cy = SIZE / 2
          if (Math.abs(y + 0.5 - cy) <= h / 2) c = FG
        }
      }
    }
    if (c) raw.set([...c, 255], o)
    else raw.set([0, 0, 0, 0], o)
  }
}

function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crcTable = []
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    crcTable[n] = c >>> 0
  }
  let crc = 0xffffffff
  for (const b of body) crc = crcTable[(crc ^ b) & 0xff] ^ (crc >>> 8)
  const crcBuf = Buffer.alloc(4)
  crcBuf.writeUInt32BE((crc ^ 0xffffffff) >>> 0)
  return Buffer.concat([len, body, crcBuf])
}

const ihdr = Buffer.alloc(13)
ihdr.writeUInt32BE(SIZE, 0)
ihdr.writeUInt32BE(SIZE, 4)
ihdr[8] = 8 // bit depth
ihdr[9] = 6 // RGBA
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
  chunk('IEND', Buffer.alloc(0)),
])
fs.writeFileSync('app-icon.png', png)
console.log('wrote app-icon.png', png.length, 'bytes')

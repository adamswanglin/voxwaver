//! voxwaver: fishaudio/s1-mini TTS inference CLI (dual-AR LM + modded-DAC codec).
mod config;
mod dac;
mod dual_ar;
mod prof;
mod prompt;
mod sampling;
mod tokenizer;
mod wavio;

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::{Args, Parser, Subcommand};
use config::{CodecConfig, DualArConfig};
use sampling::{SampleParams, Sampler};
use std::path::PathBuf;
use std::time::Instant;

/// Repetition-penalty window: the last REP_WIN_SIZE drawn tokens, zero-padded
/// on the right while fewer exist (upstream reads `previous_tokens[:, :16]`
/// out of a zero-initialized buffer). `None`-equivalent is expressed by the
/// caller not passing a window at all (the prefill sample).
fn rep_window(hist: &[u32]) -> Option<Vec<u32>> {
    let n = hist.len();
    let mut w = vec![0u32; sampling::REP_WIN_SIZE];
    let take = n.min(sampling::REP_WIN_SIZE);
    let src: &[u32] = if n <= sampling::REP_WIN_SIZE {
        &hist[..take]
    } else {
        &hist[n - take..]
    };
    w[..take].copy_from_slice(src);
    Some(w)
}

/// `rep_window` uploaded to the device as a `[REP_WIN_SIZE]` u32 tensor,
/// for the on-device sampling chain (a 64-byte H2D copy per row per frame).
fn rep_window_t(hist: &[u32], dev: &Device) -> Result<Tensor> {
    let w = rep_window(hist).unwrap_or_else(|| vec![0u32; sampling::REP_WIN_SIZE]);
    Ok(Tensor::from_vec(w, sampling::REP_WIN_SIZE, dev)?)
}

/// Split text into chunks of at most `max_bytes` UTF-8 bytes, preferring
/// sentence boundaries (mirrors upstream `group_turns_into_batches`'s
/// ~300-byte batching in `generate_long`).
fn split_chunks(text: &str, max_bytes: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?' | ';' | ':') {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    // hard-split overlong sentences on char boundaries
    let mut pieces: Vec<String> = Vec::new();
    for s in sentences {
        if s.len() <= max_bytes {
            pieces.push(s);
            continue;
        }
        let mut start = 0;
        while start < s.len() {
            let mut end = (start + max_bytes).min(s.len());
            while end > start && !s.is_char_boundary(end) {
                end -= 1;
            }
            pieces.push(s[start..end].to_string());
            start = end;
        }
    }
    // greedy packing
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for piece in pieces {
        if !cur.is_empty() && cur.len() + piece.len() > max_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&piece);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

#[derive(Parser)]
#[command(name = "voxwaver", about = "fishaudio/s1-mini TTS inference in Rust (candle)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate speech from text (optionally cloned from reference audio).
    Generate(GenerateArgs),
    /// Decode a JSON codes file ([[10 codebooks][T]]) to a WAV.
    Decode(DecodeArgs),
    /// Dump the tensor names in a .pth checkpoint (weight debugging).
    Keys(KeysArgs),
    /// Decode the semantic row of a codes JSON back to text (debug).
    Detok(DetokArgs),
}

#[derive(Args)]
struct GenerateArgs {
    /// Directory containing model.pth / codec.pth / config.json / tokenizer files
    #[arg(long, default_value = ".")]
    model_dir: PathBuf,
    /// Text to synthesize
    #[arg(long)]
    text: String,
    /// Reference audio for voice cloning (WAV, any common sample rate)
    #[arg(long)]
    ref_audio: Option<PathBuf>,
    /// Transcript of the reference audio
    #[arg(long)]
    ref_text: Option<String>,
    /// Output WAV path
    #[arg(long, default_value = "out.wav")]
    out: PathBuf,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 0.7)]
    top_p: f64,
    #[arg(long, default_value_t = 1.5)]
    repetition_penalty: f64,
    /// Max codec frames to generate (2048 samples each, ~46 ms)
    #[arg(long, default_value_t = 1024)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// cpu | cuda | metal | auto
    #[arg(long, default_value = "auto")]
    device: String,
    /// auto | bf16 | f16 | f32
    #[arg(long, default_value = "auto")]
    dtype: String,
    /// Codec decode chunk size in frames (0 = decode at once)
    #[arg(long, default_value_t = 256)]
    chunk_frames: usize,
    /// Write 16-bit PCM instead of float32
    #[arg(long)]
    pcm16: bool,
    /// Dump the prompt token layout and exit (debug)
    #[arg(long)]
    print_prompt: bool,
    /// Write the generated codebook rows to a JSON file
    #[arg(long)]
    dump_codes: Option<PathBuf>,
}

#[derive(Args)]
struct DecodeArgs {
    /// Directory containing codec.pth
    #[arg(long, default_value = ".")]
    model_dir: PathBuf,
    /// JSON file with one array of u32 per codebook
    #[arg(long)]
    codes: PathBuf,
    /// Output WAV path
    #[arg(long, default_value = "out.wav")]
    out: PathBuf,
    /// cpu | cuda | metal | auto
    #[arg(long, default_value = "auto")]
    device: String,
    /// Codec decode chunk size in frames (0 = decode at once)
    #[arg(long, default_value_t = 256)]
    chunk_frames: usize,
    /// Write 16-bit PCM instead of float32
    #[arg(long)]
    pcm16: bool,
}

#[derive(Args)]
struct DetokArgs {
    /// Directory containing tokenizer files
    #[arg(long, default_value = ".")]
    model_dir: PathBuf,
    /// JSON file written by `generate --dump-codes`
    #[arg(long)]
    codes: PathBuf,
}

#[derive(Args)]
struct KeysArgs {
    /// Path to a .pth file
    path: PathBuf,
}

fn select_device(sel: &str) -> Result<Device> {
    match sel {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0).context("CUDA device requested but unavailable"),
        "metal" => Device::new_metal(0).context("Metal device requested but unavailable"),
        "auto" => {
            #[cfg(feature = "metal")]
            {
                if let Ok(d) = Device::new_metal(0) {
                    return Ok(d);
                }
            }
            #[cfg(feature = "cuda")]
            {
                if let Ok(d) = Device::new_cuda(0) {
                    return Ok(d);
                }
            }
            Ok(Device::Cpu)
        }
        other => bail!("unknown device {other:?} (cpu|cuda|metal|auto)"),
    }
}

fn select_dtype(sel: &str, dev: &Device) -> Result<DType> {
    match sel {
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "auto" => Ok(match dev {
            // f32 is the fast path on Metal and CPU
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        }),
        other => bail!("unknown dtype {other:?} (auto|bf16|f16|f32)"),
    }
}

fn run_generate(args: &GenerateArgs) -> Result<()> {
    let dev = select_device(&args.device)?;
    let dtype = select_dtype(&args.dtype, &dev)?;
    eprintln!("[voxwaver] device={:?} dtype={:?}", dev.location(), dtype);

    let codec_cfg = CodecConfig::default();
    let mut tok = tokenizer::Tokenizer::load(&args.model_dir)?;
    let cfg = DualArConfig::load(&args.model_dir.join("config.json"))?;

    // ---- reference audio -> codec codes ----
    let ref_codes: Option<Vec<Vec<u32>>> = match &args.ref_audio {
        Some(path) => {
            let ref_text = args
                .ref_text
                .as_deref()
                .context("--ref-text is required with --ref-audio")?;
            ensure!(!ref_text.is_empty(), "--ref-text must not be empty");
            let (samples, sr) = wavio::read_wav_mono(path)?;
            let samples = wavio::resample(&samples, sr, codec_cfg.sample_rate);
            eprintln!(
                "[voxwaver] reference: {:.1}s @ {} Hz -> {:.1}s @ {} Hz",
                samples.len() as f64 / codec_cfg.sample_rate as f64,
                sr,
                samples.len() as f64 / codec_cfg.sample_rate as f64,
                codec_cfg.sample_rate
            );
            let t0 = Instant::now();
            let codec = dac::Dac::load(&args.model_dir.join("codec.pth"), &codec_cfg, &dev)?;
            let codes = codec.encode(&samples).context("encoding reference audio")?;
            eprintln!(
                "[voxwaver] encoded reference: {} frames (load+encode {:.1}s)",
                codes[0].len(),
                t0.elapsed().as_secs_f64()
            );
            if let Ok(out) = std::env::var("ROUNDTRIP") {
                // codec sanity: decode the reference codes back to audio
                let codec = dac::Dac::load(&args.model_dir.join("codec.pth"), &codec_cfg, &dev)?;
                let audio = codec.decode_codes_chunked(&codes, args.chunk_frames)?;
                wavio::write_wav(
                    std::path::Path::new(&out),
                    &audio,
                    codec_cfg.sample_rate,
                    args.pcm16,
                )?;
                eprintln!("[voxwaver] roundtrip wrote {out}");
                return Ok(());
            }
            Some(codes)
        }
        None => None,
    };

    // ---- text chunking ----
    // Upstream generate_long never feeds a whole article as one prompt: text is
    // split into ~300-byte batches and each batch is generated to <|im_end|,
    // with the generated codes fed back as prior turns (iterative prompting).
    // Long single prompts measurably degrade s1-mini's output.
    let chunks = split_chunks(&args.text, 300);
    eprintln!("[voxwaver] text split into {} chunks", chunks.len());
    let mut history: Vec<(String, Vec<Vec<u32>>)> = match (&args.ref_text, &ref_codes) {
        (Some(t), Some(c)) => vec![(t.clone(), c.clone())],
        _ => Vec::new(),
    };
    if args.print_prompt {
        let hist: Vec<prompt::Turn> = history
            .iter()
            .map(|(t, c)| prompt::Turn { text: t, codes: c })
            .collect();
        let p = prompt::build(&mut tok, chunks.first().map(String::as_str).unwrap_or(""), &hist, cfg.max_seq_len, cfg.num_codebooks)?;
        println!("prompt len = {}", p.len);
        println!("tokens = {:?}", p.tokens);
        return Ok(());
    }

    // ---- model ----
    let t0 = Instant::now();
    let mut model = dual_ar::DualArModel::load(
        &args.model_dir.join("model.pth"),
        &cfg,
        tok.semantic_begin,
        tok.semantic_end,
        dtype,
        &dev,
    )?;
    eprintln!(
        "[voxwaver] model loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let params = SampleParams {
        temperature: args.temperature,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
    };
    // seed the device RNG (Metal tausworthe/lcg) used by the GPU sampling
    // chain so runs are reproducible on the same device
    dev.set_seed(args.seed)?;
    let sampler = Sampler::new(args.seed, dtype == DType::BF16);

    // Prefill runs in slices so the manual causal attention (used on Metal
    // for bf16, see dual_ar::attention) never materializes a score matrix
    // much larger than [16, SLICE, prompt_len]; synchronize between slices
    // to let the Metal buffer pool release the intermediates.
    let prefill_chunk = std::env::var("PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let rows = cfg.num_codebooks + 1;
    let t_total = Instant::now();
    let mut gen_codes: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
    let mut budget = args.max_new_tokens;
    // global generated-frame counter (across text chunks) for prof reports
    let mut prof_step: u64 = 0;

    for (ci, chunk_text) in chunks.iter().enumerate() {
        if budget == 0 {
            eprintln!("[voxwaver] --max-new-tokens budget exhausted");
            break;
        }
        // rebuild the prompt with the generated turns fed back; drop the
        // oldest generated turns if the conversation outgrows the context
        let (p, prompt_len) = loop {
            let hist: Vec<prompt::Turn> = history
                .iter()
                .map(|(t, c)| prompt::Turn { text: t, codes: c })
                .collect();
            match prompt::build(&mut tok, chunk_text, &hist, cfg.max_seq_len, cfg.num_codebooks) {
                Ok(p) => {
                    let len = p.len;
                    break (p, len);
                }
                Err(e) => {
                    // keep the reference turn (index 0) if present
                    let droppable = history.len() > usize::from(args.ref_audio.is_some());
                    if droppable {
                        let dropped = history.remove(usize::from(args.ref_audio.is_some()));
                        eprintln!(
                            "[voxwaver] context full: dropping oldest generated turn ({} frames)",
                            dropped.1[0].len()
                        );
                    } else {
                        return Err(e);
                    }
                }
            }
        };
        let max_new = budget.min(cfg.max_seq_len.saturating_sub(prompt_len + 1));
        ensure!(max_new > 0, "prompt fills the context window");
        model.setup_caches(prompt_len, max_new)?;

        // flat values, row-major [11, T]: row 0 tokens, rows 1.. codes
        let mut values: Vec<u32> = Vec::with_capacity(prompt_len * rows);
        values.extend_from_slice(&p.tokens);
        for row in &p.codes {
            values.extend_from_slice(row);
        }

        let t0 = Instant::now();
        let (mut logits, mut hidden) = (None, None);
        for pos0 in (0..prompt_len).step_by(prefill_chunk) {
            let len = prefill_chunk.min(prompt_len - pos0);
            let slice: Vec<u32> = (0..rows)
                .flat_map(|row| {
                    let base = row * prompt_len + pos0;
                    values[base..base + len].iter().copied()
                })
                .collect();
            let slice = Tensor::from_vec(slice, (rows, len), &dev)?;
            let (l, h) = model.forward(&slice, pos0)?;
            logits = Some(l);
            hidden = Some(h);
            dev.synchronize()?;
        }
        let (mut logits, mut hidden) = (logits.unwrap(), hidden.unwrap());
        eprintln!(
            "[voxwaver] chunk {}/{}: prefill {} tokens in {:.1}s",
            ci + 1,
            chunks.len(),
            prompt_len,
            t0.elapsed().as_secs_f64()
        );
    if let Ok(path) = std::env::var("DUMP_PREFILL") {
        let l = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let h = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write(format!("{path}.logits.f32"), bytemuck::cast_slice(&l))?;
        std::fs::write(format!("{path}.hidden.f32"), bytemuck::cast_slice(&h))?;
        eprintln!("[voxwaver] dumped prefill logits/hidden to {path}.*.f32");
    }
    if std::env::var("DUMP_FAST").is_ok() {
        let row = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let token = row
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| p.total_cmp(q))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let sem_code = token.saturating_sub(tok.semantic_begin);
        let (codes, rows) = model.fast_frame_greedy(&hidden, sem_code)?;
        eprintln!("[debug] sem_code {sem_code} greedy fast codes: {codes:?}");
        for (i, r) in rows.iter().enumerate() {
            std::fs::write(
                format!("/tmp/vox_fast_{i}.f32"),
                bytemuck::cast_slice(r),
            )?;
        }
        std::process::exit(0);
    }
    eprintln!(
        "[voxwaver] prefill {} tokens in {:.1}s",
        prompt_len,
        t0.elapsed().as_secs_f64()
    );

        // ---- generation loop ----
        let mut pos = prompt_len;
        let mut chunk_codes: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
        // per-row repetition-penalty history (main token row + fast codebook
        // rows), mirroring `previous_tokens` in `decode_n_tokens`
        let mut main_hist: Vec<u32> = Vec::new();
        let mut cb_hist: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
        let t0 = Instant::now();

        for step in 0..max_new {
        // GPU sampling chain: submit the sampler kernels, then a single
        // blocking scalar read for the token id (the im_end control flow
        // and the history bookkeeping need it on the CPU)
        let window = rep_window_t(&main_hist, &dev)?;
        let token_t = {
            let _g = prof::scope(prof::MAIN_SAMPLE);
            sampler.sample_gpu(&logits, &params, Some(&window))?
        };
        let token = {
            let _g = prof::scope(prof::MAIN_READ);
            token_t.to_vec1::<u32>()?[0]
        };
            if token == tok.im_end {
                eprintln!("[voxwaver] chunk {} done (<|im_end|>) after {step} frames", ci + 1);
                break;
            }
            if step > 0 {
                // upstream `previous_tokens` only records tokens drawn inside
                // decode_n_tokens; the prefill sample (step 0) never enters it
                main_hist.push(token);
            }
            let sem_code = token - tok.semantic_begin;

            let cb_windows: Vec<Option<Tensor>> = (1..cfg.num_codebooks)
                .map(|k| rep_window_t(&cb_hist[k], &dev).map(Some))
                .collect::<Result<_>>()?;
            let codes = model.fast_frame(&hidden, sem_code, &sampler, &params, &cb_windows)?;
            for ((cb_gen, cb_win), &c) in chunk_codes
                .iter_mut()
                .zip(cb_hist.iter_mut())
                .zip(&codes)
            {
                cb_gen.push(c);
                if step > 0 {
                    cb_win.push(c);
                }
            }
            budget -= 1;

            if step + 1 == max_new {
                break;
            }
            // feed [token, codes...] back
            let mut col = Vec::with_capacity(cfg.num_codebooks + 1);
            col.push(token);
            col.extend_from_slice(&codes);
            let col = Tensor::from_vec(col, (cfg.num_codebooks + 1, 1), &dev)?;
            let (new_logits, new_hidden) = {
                let _g = prof::scope(prof::MAIN_FWD);
                model.forward(&col, pos)?
            };
            logits = new_logits;
            hidden = new_hidden;
            pos += 1;

            prof_step += 1;
            prof::maybe_report(prof_step, false);
            // Periodic full synchronize: the Metal cross-encoder fence map
            // (untracked-hazard ordering) and the buffer pool only get
            // cleaned on `synchronize()`; without this, command encoding
            // slows down monotonically over the decode loop (every new
            // encoder waits on more stale fences). The GPU is already drained
            // by the per-frame readbacks, so the extra wait is ~free.
            if dev.is_metal() && prof_step % 25 == 0 {
                dev.synchronize()?;
            }
            if step % 50 == 0 {
                let fps = (step + 1) as f64 / t0.elapsed().as_secs_f64().max(1e-9);
                eprintln!("[voxwaver] frame {step}/{max_new} ({fps:.1} frames/s)");
            }
            if pos >= cfg.max_seq_len {
                eprintln!("[voxwaver] context limit reached");
                break;
            }
        }

        prof::maybe_report(prof_step, true);

        let chunk_frames = chunk_codes[0].len();
        ensure!(chunk_frames > 0, "no frames generated for chunk {}", ci + 1);
        eprintln!(
            "[voxwaver] chunk {}/{}: {} frames ({:.1}s audio) in {:.1}s",
            ci + 1,
            chunks.len(),
            chunk_frames,
            chunk_frames as f64 * codec_cfg.frame_length() as f64 / codec_cfg.sample_rate as f64,
            t0.elapsed().as_secs_f64()
        );
        for (g, c) in gen_codes.iter_mut().zip(chunk_codes.iter()) {
            g.extend_from_slice(c);
        }
        history.push((chunk_text.clone(), chunk_codes));
    }

    let frames = gen_codes[0].len();
    ensure!(frames > 0, "no frames generated");
    eprintln!(
        "[voxwaver] generated {} frames ({:.1}s audio) in {:.1}s",
        frames,
        frames as f64 * codec_cfg.frame_length() as f64 / codec_cfg.sample_rate as f64,
        t_total.elapsed().as_secs_f64()
    );

    // ---- codec decode ----
    let t0 = Instant::now();
    // Drop the transformer + caches and free GPU buffers pooled during
    // generation before the memory-hungry codec decode (synchronize also
    // drops unused pooled Metal buffers).
    drop(model);
    dev.synchronize()?;
    let codec = dac::Dac::load(&args.model_dir.join("codec.pth"), &codec_cfg, &dev)?;
    let audio = codec.decode_codes_chunked(&gen_codes, args.chunk_frames)?;
    eprintln!(
        "[voxwaver] decoded {} samples in {:.1}s",
        audio.len(),
        t0.elapsed().as_secs_f64()
    );

    if let Some(path) = &args.dump_codes {
        std::fs::write(path, serde_json::to_string(&gen_codes)?)?;
        eprintln!("[voxwaver] wrote codes to {}", path.display());
    }
    wavio::write_wav(&args.out, &audio, codec_cfg.sample_rate, args.pcm16)?;
    eprintln!("[voxwaver] wrote {}", args.out.display());
    Ok(())
}

fn run_decode(args: &DecodeArgs) -> Result<()> {
    let dev = select_device(&args.device)?;
    let codec_cfg = CodecConfig::default();
    let raw = std::fs::read_to_string(&args.codes)
        .with_context(|| format!("read {}", args.codes.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let rows = value
        .as_array()
        .with_context(|| "codes JSON must be an array of arrays")?;
    let mut codes: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
    for row in rows {
        codes.push(
            row.as_array()
                .context("each codebook row must be an array")?
                .iter()
                .map(|v| {
                    let v = v.as_u64().context("codes must be integers")?;
                    Ok(v as u32)
                })
                .collect::<Result<Vec<u32>>>()?,
        );
    }
    ensure!(!codes.is_empty() && !codes[0].is_empty(), "empty codes");
    ensure!(
        codes.iter().all(|r| r.len() == codes[0].len()),
        "ragged codebook rows"
    );
    eprintln!(
        "[voxwaver] codes: {} rows x {} frames",
        codes.len(),
        codes[0].len()
    );
    let codec = dac::Dac::load(&args.model_dir.join("codec.pth"), &codec_cfg, &dev)?;
    let audio = codec.decode_codes_chunked(&codes, args.chunk_frames)?;
    wavio::write_wav(&args.out, &audio, codec_cfg.sample_rate, args.pcm16)?;
    eprintln!("[voxwaver] wrote {}", args.out.display());
    Ok(())
}

fn run_detok(args: &DetokArgs) -> Result<()> {
    let tok = tokenizer::Tokenizer::load(&args.model_dir)?;
    let raw = std::fs::read_to_string(&args.codes)
        .with_context(|| format!("read {}", args.codes.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let sem = value
        .as_array()
        .and_then(|rows| rows.first())
        .and_then(|r| r.as_array())
        .context("codes JSON must be [[...]] with the semantic row first")?;
    let ids: Vec<u32> = sem
        .iter()
        .map(|v| Ok(v.as_u64().context("codes must be integers")? as u32))
        .collect::<Result<Vec<_>>>()?;
    let ids: Vec<u32> = ids.iter().map(|&c| c + tok.semantic_begin).collect();
    println!("{}", tok.decode(&ids));
    Ok(())
}

fn run_keys(args: &KeysArgs) -> Result<()> {
    let pth = candle_core::pickle::PthTensors::new(&args.path, None)
        .with_context(|| format!("open {}", args.path.display()))?;
    let infos = pth.tensor_infos();
    let mut names: Vec<&String> = infos.keys().collect();
    names.sort();
    for name in names {
        let info = &infos[name];
        println!("{name}\t{:?}\t{:?}", info.layout.shape(), info.dtype);
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match &cli.cmd {
        Cmd::Generate(args) => run_generate(args),
        Cmd::Decode(args) => run_decode(args),
        Cmd::Keys(args) => run_keys(args),
        Cmd::Detok(args) => run_detok(args),
    };
    if let Err(e) = result {
        eprintln!("[voxwaver] error: {e:#}");
        std::process::exit(1);
    }
}

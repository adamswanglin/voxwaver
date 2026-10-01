//! voxwaver: fishaudio/s1-mini TTS inference CLI (dual-AR LM + modded-DAC codec).
use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use voxwaver_core::engine::{
    CancelFlag, Engine, EngineConfig, GenerateRequest, Progress, ProgressSink, RefTurn,
};
use voxwaver_core::{config, dac, prompt, select_device, select_dtype, tokenizer, wavio};

/// CLI sink: logs to stderr with the historical `[voxwaver]` prefix.
struct CliSink;
impl ProgressSink for CliSink {
    fn progress(&self, p: Progress) {
        match p {
            Progress::LoadingLm => eprintln!("[voxwaver] loading model…"),
            Progress::LoadingCodec => eprintln!("[voxwaver] loading codec…"),
            Progress::EncodingRef { seconds } => {
                eprintln!("[voxwaver] encoding reference ({seconds:.1}s)")
            }
            Progress::Chunks { total } => eprintln!("[voxwaver] text split into {total} chunks"),
            Progress::Prefill {
                chunk,
                total,
                tokens,
            } => eprintln!("[voxwaver] chunk {chunk}/{total}: prefill {tokens} tokens"),
            Progress::Generating {
                chunk,
                total,
                frames,
                max_frames,
                fps,
            } => eprintln!(
                "[voxwaver] chunk {chunk}/{total}: frame {frames}/{max_frames} ({fps:.1} frames/s)"
            ),
            Progress::Decoding { chunk, total, frames_total } => {
                eprintln!("[voxwaver] decoding {frames_total} frames (chunk {chunk}/{total})")
            }
            // OmniVoice-only stage; the s1-mini CLI never emits it.
            Progress::Unmasking { .. } => {}
            Progress::WritingWav => {}
            Progress::Done { frames, seconds } => {
                eprintln!("[voxwaver] done: {frames} frames ({seconds:.1}s audio)")
            }
        }
    }
    fn log(&self, msg: &str) {
        eprintln!("[voxwaver] {msg}");
    }
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
    /// Max codec frames to generate per text chunk (2048 samples each, ~46 ms); 0 = unlimited
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

fn run_generate(args: &GenerateArgs) -> Result<()> {
    let dev = select_device(&args.device)?;
    let dtype = select_dtype(&args.dtype, &dev)?;
    let sink = CliSink;

    let mut cfg = EngineConfig::new(&args.model_dir, dev, dtype);
    cfg.chunk_frames = args.chunk_frames;
    cfg.max_new_tokens = args.max_new_tokens;
    let mut engine = Engine::new(cfg)?;

    // ---- reference audio -> codec codes ----
    let ref_turn = match (&args.ref_audio, &args.ref_text) {
        (Some(path), Some(ref_text)) => {
            ensure!(!ref_text.is_empty(), "--ref-text must not be empty");
            let codes = engine.encode_reference(path, &sink)?;
            Some(RefTurn {
                text: ref_text.clone(),
                codes,
            })
        }
        (Some(_), None) => bail!("--ref-text is required with --ref-audio"),
        _ => None,
    };

    if args.print_prompt {
        let cfg_ar = config::DualArConfig::load(&args.model_dir.join("config.json"))?;
        let mut tok = tokenizer::Tokenizer::load(&args.model_dir)?;
        let hist: Vec<prompt::Turn> = match &ref_turn {
            Some(r) => vec![prompt::Turn { text: &r.text, codes: &r.codes }],
            None => Vec::new(),
        };
        let p = prompt::build(
            &mut tok,
            args.text.lines().next().unwrap_or(""),
            &hist,
            cfg_ar.max_seq_len,
            cfg_ar.num_codebooks,
        )?;
        println!("prompt len = {}", p.len);
        println!("tokens = {:?}", p.tokens);
        return Ok(());
    }

    let t0 = Instant::now();
    let req = GenerateRequest {
        text: args.text.clone(),
        ref_turn,
        params: voxwaver_core::sampling::SampleParams {
            temperature: args.temperature,
            top_p: args.top_p,
            repetition_penalty: args.repetition_penalty,
        },
        seed: args.seed,
    };
    let out = engine.generate(&req, &CancelFlag::new(), &sink)?;

    if let Some(path) = &args.dump_codes {
        std::fs::write(path, serde_json::to_string(&out.codes)?)?;
        eprintln!("[voxwaver] wrote codes to {}", path.display());
    }
    sink.progress(Progress::WritingWav);
    wavio::write_wav(&args.out, &out.samples, out.sample_rate, args.pcm16)?;
    sink.log(&format!(
        "wrote {} in {:.1}s total",
        args.out.display(),
        t0.elapsed().as_secs_f64()
    ));
    Ok(())
}

fn run_decode(args: &DecodeArgs) -> Result<()> {
    let dev = select_device(&args.device)?;
    let codec_cfg = config::CodecConfig::default();
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
    let audio = codec.decode_codes_chunked(&codes, args.chunk_frames, None)?;
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

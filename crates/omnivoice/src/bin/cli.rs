//! omnivoice-cli: text-to-speech over the OmniVoice two-stage pipeline.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tts_common::{CancelFlag, Progress, ProgressSink};

use omnivoice::config as cfg;
use omnivoice::engine::{Engine, SpeakOptions};
use omnivoice::generator::GenParams;

/// CLI sink: logs to stderr with an `[omnivoice]` prefix.
struct CliSink;
impl ProgressSink for CliSink {
    fn progress(&self, p: Progress) {
        match p {
            Progress::Chunks { total } => {
                eprintln!("[omnivoice] text split into {total} chunks")
            }
            Progress::Unmasking { chunk, step, total_steps, .. }
                if step % 8 == 0 || step + 1 == total_steps =>
            {
                eprintln!("[omnivoice] chunk {chunk}: unmasking {}/{total_steps}", step + 1);
            }
            Progress::Done { frames, seconds } => {
                eprintln!("[omnivoice] done: {frames} frames ({seconds:.1}s audio)")
            }
            _ => {}
        }
    }
    fn log(&self, msg: &str) {
        eprintln!("{msg}");
    }
}

#[derive(Parser)]
#[command(
    name = "omnivoice-cli",
    about = "OmniVoice TTS: Qwen3 iterative-unmasking generator + HiggsAudioV2 RVQ/DAC decoder"
)]
struct Cli {
    /// OmniVoice model directory (config.json / model.safetensors / tokenizer.json / audio_tokenizer/).
    #[arg(long)]
    model_dir: PathBuf,
    /// Text to synthesize.
    #[arg(long)]
    text: String,
    /// Output wav path.
    #[arg(long, default_value = "omnivoice.wav")]
    out: PathBuf,
    /// Language tag (e.g. "en", "zh"); "None" leaves it unset.
    #[arg(long)]
    lang: Option<String>,
    /// Style instruction (e.g. "Speak with a cheerful tone"); "None" leaves it unset.
    #[arg(long)]
    instruct: Option<String>,
    /// Random seed; omit for entropy-seeded sampling.
    #[arg(long)]
    seed: Option<u64>,
    /// Force CPU inference.
    #[arg(long)]
    cpu: bool,
    /// Iterative unmasking steps.
    #[arg(long, default_value_t = cfg::NUM_STEP)]
    steps: usize,
    /// Classifier-free guidance scale.
    #[arg(long, default_value_t = cfg::GUIDANCE_SCALE)]
    guidance: f64,
    /// Time shift of the unmasking schedule.
    #[arg(long, default_value_t = cfg::T_SHIFT)]
    t_shift: f64,
    /// Per-codebook layer penalty.
    #[arg(long, default_value_t = cfg::LAYER_PENALTY_FACTOR)]
    layer_penalty: f64,
    /// Gumbel temperature for position selection.
    #[arg(long, default_value_t = cfg::POSITION_TEMPERATURE)]
    pos_temp: f64,
    /// Token sampling temperature; 0 selects greedy decoding.
    #[arg(long, default_value_t = cfg::CLASS_TEMPERATURE)]
    class_temp: f64,
    /// Reference audio wav for voice cloning (any sample rate; resampled internally).
    #[arg(long)]
    ref_audio: Option<PathBuf>,
    /// Transcript of the reference audio (optional; helps cloning fidelity).
    #[arg(long)]
    ref_text: Option<String>,
    /// Speaking speed; >1 faster, <1 slower.
    #[arg(long, default_value_t = 1.0)]
    speed: f64,
    /// Target chunk duration (seconds) for long text; 0 disables chunking.
    #[arg(long, default_value_t = cfg::AUDIO_CHUNK_DURATION)]
    audio_chunk_duration: f64,
    /// Estimated audio duration (seconds) above which chunking is activated.
    #[arg(long, default_value_t = cfg::AUDIO_CHUNK_THRESHOLD)]
    audio_chunk_threshold: f64,
    /// Skip output post-processing (silence removal / normalization / fades).
    #[arg(long)]
    no_postprocess: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let params = GenParams {
        num_step: cli.steps,
        guidance_scale: cli.guidance,
        t_shift: cli.t_shift,
        layer_penalty_factor: cli.layer_penalty,
        position_temperature: cli.pos_temp,
        class_temperature: cli.class_temp,
    };
    let engine = Engine::load(&cli.model_dir, cli.cpu)?;
    let ref_audio = match &cli.ref_audio {
        Some(p) => {
            let (wav, sr) = omnivoice::wavio::read_wav(p)?;
            eprintln!("[omnivoice] reference audio {} samples @ {sr} Hz", wav.len());
            Some((wav, sr))
        }
        None => None,
    };
    let started = Instant::now();
    let (ref_codes, ref_rms) = match &ref_audio {
        Some((wav, sr)) => {
            let codes = engine.encode_ref(wav, *sr)?;
            eprintln!(
                "[omnivoice] reference audio encoded to 8 x {} tokens",
                codes[0].len()
            );
            (Some(codes), Some(omnivoice::engine::ref_rms(wav, *sr)))
        }
        None => (None, None),
    };
    let opts = SpeakOptions {
        speed: cli.speed,
        audio_chunk_duration: cli.audio_chunk_duration,
        audio_chunk_threshold: cli.audio_chunk_threshold,
        postprocess_output: !cli.no_postprocess,
        ref_rms,
        ..Default::default()
    };
    let wave = engine.tts_with(
        &cli.text,
        cli.lang.as_deref(),
        cli.instruct.as_deref(),
        cli.seed,
        &params,
        ref_codes.as_deref(),
        cli.ref_text.as_deref(),
        &opts,
        &CancelFlag::new(),
        &CliSink,
    )?;
    omnivoice::wavio::write_wav(&cli.out, &wave, cfg::SAMPLE_RATE)?;
    eprintln!(
        "[omnivoice] wrote {} ({:.2}s of audio) in {:.1}s",
        cli.out.display(),
        wave.len() as f32 / cfg::SAMPLE_RATE as f32,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

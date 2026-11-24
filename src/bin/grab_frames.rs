//! Captures N static frames from the camera into `assets/test/` (Phase 2a).
//!
//! Usage: `cargo run --release --bin grab_frames -- [--count N] [--outdir DIR]`
//!
//! (Fixtures are gitignored; see NOTES.md, Phase 2a.)

use anyhow::{Context, Result};
use opencv::core::Mat;
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture};
use std::path::PathBuf;

/// Reads `camera_index` from the holox config (mirrors the app's config).
fn camera_index_from_config() -> Result<i32> {
    #[derive(serde::Deserialize, Default)]
    struct Cfg {
        #[serde(default)]
        camera_index: i32,
    }
    let path = match dirs::config_dir() {
        Some(d) => d.join("holox/config.toml"),
        None => PathBuf::from(".holox/config.toml"),
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    Ok(toml::from_str::<Cfg>(&text).unwrap_or_default().camera_index)
}

fn main() -> Result<()> {
    let mut count: u32 = 10;
    let mut outdir = PathBuf::from("assets/test");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--count" => {
                count = args
                    .next()
                    .context("--count needs a value")?
                    .parse()
                    .context("--count value must be a number")?
            }
            "--outdir" => {
                outdir = PathBuf::from(args.next().context("--outdir needs a value")?)
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }
    std::fs::create_dir_all(&outdir)?;

    let index = camera_index_from_config()?;
    let mut cap = VideoCapture::new(index, videoio::CAP_V4L2)?;
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
    let mut seq: u32 = 0;
    for _ in 0..count {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.rows() == 0 || frame.cols() == 0 {
            continue;
        }
        let path = outdir.join(format!("frame_{seq:03}.png"));
        imgcodecs::imwrite_def(path.to_string_lossy().as_ref(), &frame)?;
        seq += 1;
    }
    println!("captured {seq} frames into {}", outdir.display());
    Ok(())
}

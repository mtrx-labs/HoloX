//! HoloX: a simulated, Jarvis-style holographic desktop OS controlled by hand
//! gestures via a webcam. See NOTES.md for the phased build plan.

mod app;
mod config;
mod vision;

use anyhow::Result;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("HoloX scaffolding (phase 0)");

    let _config = config::load()?;

    let _vision_rx = vision::spawn_vision_thread();

    app::run()
}

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

    info!("HoloX (phase 1)");

    let config = config::load()?;

    // Channel shared between the vision thread (Sender) and the Bevy side
    // (Receiver), wrapped in the `VisionRx` resource.
    let (tx, rx) = crossbeam_channel::unbounded::<vision::VisionMsg>();
    let _vision_handle = vision::spawn_vision_thread(config.camera_index, tx);

    app::run(rx)
}

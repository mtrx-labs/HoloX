//! HoloX binary: entry point. The library (see `lib.rs`) holds the modules;
//! the binary wires them together.

use anyhow::Result;
use holox::{app, config, vision};
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("HoloX (phase 2a)");

    let settings = config::load()?;

    // Channel shared between the vision thread (Sender) and the Bevy side
    // (Receiver), wrapped in the `VisionRx` resource.
    let (tx, rx) = crossbeam_channel::unbounded::<vision::VisionMsg>();
    let _vision_handle = vision::spawn_vision_thread(settings.camera_index, tx);

    app::run(rx)
}

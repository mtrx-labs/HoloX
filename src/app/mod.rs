//! Bevy-based holographic desktop application (Vulkan default backend).

use anyhow::Result;
use tracing::info;

/// Placeholder for later phases: builds the Bevy `App` with default plugins,
/// receives `VisionEvent`s over the crossbeam channel, and renders the
/// holographic desktop.
pub fn run() -> Result<()> {
    // Scaffold reference to the Bevy dependency; phases 2+ add plugins,
    // systems and the render loop here.
    let _app = bevy::app::App::new();
    info!("app::run() called (scaffolding; Bevy desktop deferred to later phases)");
    Ok(())
}

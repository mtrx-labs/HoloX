//! Computer-vision pipeline: camera capture and hand-pose estimation via OpenCV DNN.
//!
//! The vision thread owns the OpenCV objects; the Bevy main thread receives
//! lightweight events over a `crossbeam-channel` (see [`VisionEvent`]).

pub mod camera;
pub mod hand_landmarks;
pub mod palm_detector;

use crossbeam_channel::Receiver;
use std::time::Duration;
use tracing::info;

/// An event produced by the vision thread and consumed by the Bevy loop.
///
/// The concrete payloads are filled in as later phases add detection results.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VisionEvent {
    /// Wall-clock timestamp of the frame, milliseconds since the UNIX epoch.
    pub timestamp_ms: u64,
    /// Number of hands detected in the frame.
    pub hands: u8,
}

/// Spawns the dedicated vision thread.
///
/// Later phases will have this thread open the camera, run palm detection and
/// hand-landmark estimation, and send [`VisionEvent`]s over an unbounded
/// channel. This placeholder thread keeps the concurrency skeleton in place;
/// the render loop never blocks on it.
pub fn spawn_vision_thread() -> Receiver<VisionEvent> {
    let (tx, rx) = crossbeam_channel::unbounded::<VisionEvent>();
    std::thread::Builder::new()
        .name("holox-vision".to_string())
        .spawn(move || {
            // Keep the sender alive for the thread's lifetime; the camera
            // loop and detection pipeline are wired in later phases.
            let _sender = tx;
            info!("vision thread started (placeholder; camera pipeline deferred)");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        })
        .expect("spawn holox-vision thread");
    rx
}

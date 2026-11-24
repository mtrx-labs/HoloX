//! V4L2 camera capture via OpenCV.

use anyhow::Result;
use tracing::debug;

/// Path to the camera device, matching the environment (`/dev/video0`).
#[allow(dead_code)]
pub const CAMERA_PATH: &str = "/dev/video0";

/// Placeholder for later phases: opens the camera with `opencv::videoio::VideoCapture`
/// and returns the handle used by the vision thread.
#[allow(dead_code)]
pub fn open_camera() -> Result<()> {
    debug!(camera = CAMERA_PATH, "camera open: deferred to later phases");
    Ok(())
}

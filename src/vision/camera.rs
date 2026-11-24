//! V4L2 camera capture via OpenCV.

use anyhow::{Context, Result};
use opencv::core::Mat;
use opencv::prelude::{VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::videoio::{self, VideoCapture};
use tracing::{info, warn};

/// A V4L2 camera device wrapped by `opencv::videoio::VideoCapture`.
///
/// This struct is deliberately **not** `Sync`: it is owned by the vision
/// thread, and `next_frame` reads into a caller-provided buffer to avoid
/// per-frame allocation.
pub struct Camera {
    cap: VideoCapture,
    /// Frame width in pixels, as the driver actually accepted.
    pub width: i32,
    /// Frame height in pixels, as the driver actually accepted.
    pub height: i32,
}

impl Camera {
    /// Opens the V4L2 camera at `index` (0 for `/dev/video0`) and requests
    /// 640x480 @ 30 FPS. Logs the values the driver actually accepted; V4L2
    /// often silently negotiates a different mode, so we read them back.
    pub fn open(index: i32) -> Result<Self> {
        let mut cap = VideoCapture::new(index, videoio::CAP_V4L2)?;
        cap.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
        cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
        cap.set(videoio::CAP_PROP_FPS, 30.0)?;
        if !cap.is_opened()? {
            anyhow::bail!("camera {index} did not open (V4L2)");
        }
        let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
        let fps = cap.get(videoio::CAP_PROP_FPS)?;
        info!(index, width, height, fps, "camera opened (V4L2)");
        Ok(Self { cap, width, height })
    }

    /// Reads the next frame into the caller-provided buffer.
    ///
    /// `VideoCapture::read` blocks until a frame is available, so this
    /// throttles the vision thread to the camera's FPS naturally.
    pub fn next_frame(&mut self, buf: &mut Mat) -> Result<()> {
        let ok = self.cap.read(buf).context("camera read failed")?;
        if !ok {
            warn!("camera read returned no frame");
        }
        Ok(())
    }
}

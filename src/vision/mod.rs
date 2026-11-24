//! Computer-vision pipeline: camera capture and hand-pose estimation via OpenCV DNN.
//!
//! The vision thread owns the OpenCV objects; the Bevy main thread receives
//! lightweight, OpenCV-free messages over a `crossbeam-channel` ([`VisionMsg`])
//! and drains them each frame (see `crate::app::drain_vision`).

pub mod camera;
pub mod hand_landmarks;
pub mod palm_detector;

use camera::Camera;
use crossbeam_channel::Sender;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

/// A frame crossing the thread boundary. Deliberately free of OpenCV types.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Monotonically increasing frame sequence number (per vision thread).
    pub seq: u64,
    /// Frame width in pixels, as the driver actually accepted.
    #[allow(dead_code)] // read by later phases / the debug preview.
    pub width: u32,
    /// Frame height in pixels, as the driver actually accepted.
    #[allow(dead_code)] // read by later phases / the debug preview.
    pub height: u32,
    /// Wall-clock frame timestamp, nanoseconds since the UNIX epoch.
    #[allow(dead_code)] // read by later phases / the debug preview.
    pub timestamp_ns: u64,
    /// Downscaled RGB8 preview (row-major), 320x240. Only present with the
    /// `debug_preview` feature.
    #[cfg(feature = "debug_preview")]
    pub rgb: Vec<u8>,
}

/// Hand gesture labels. Most variants are produced by later phases.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gesture {
    #[default]
    None,
    Pinch,
    OpenPalm,
    Fist,
    Point,
}

/// Dummy hand data (21 landmarks, MediaPipe layout). Real palm-detection and
/// hand-landmark estimation arrive in Phase 2.
#[derive(Debug, Clone)]
pub struct HandData {
    pub landmarks: [glam::Vec3; 21],
    #[allow(dead_code)] // read by later phases.
    pub gesture: Gesture,
    #[allow(dead_code)] // read by later phases.
    pub frame_seq: u64,
    #[allow(dead_code)] // read by later phases.
    pub timestamp_ns: u64,
}

/// One vision message per frame: the frame plus a `HandData` snapshot.
#[derive(Debug, Clone)]
pub struct VisionMsg {
    pub frame: Frame,
    pub hand: HandData,
}

/// Preview dimensions used by the feature-gated RGB data.
pub const PREVIEW_WIDTH: u32 = 320;
/// Preview height, counterpart of [`PREVIEW_WIDTH`].
pub const PREVIEW_HEIGHT: u32 = 240;

/// Spawns the dedicated vision thread.
pub fn spawn_vision_thread(
    camera_index: i32,
    tx: Sender<VisionMsg>,
) -> std::thread::JoinHandle<anyhow::Result<()>> {
    std::thread::Builder::new()
        .name("holox-vision".to_string())
        .spawn(move || vision_loop(camera_index, tx))
        .expect("spawn holox-vision thread")
}

/// Main loop of the vision thread: capture, dummy hand data, channel send.
fn vision_loop(camera_index: i32, tx: Sender<VisionMsg>) -> anyhow::Result<()> {
    // Open the camera on the vision thread (never blocks Bevy startup).
    let mut camera = match Camera::open(camera_index) {
        Ok(camera) => Some(camera),
        Err(e) => {
            error!(camera_index, "camera open failed: {e:#}; entering no-camera mode");
            None
        }
    };
    info!(camera_index, "vision thread started");
    let mut buf = opencv::core::Mat::default();
    let mut seq: u64 = 0;
    loop {
        seq += 1;
        let (width, height) = if let Some(camera) = camera.as_mut() {
            camera.next_frame(&mut buf)?;
            (camera.width.unsigned_abs(), camera.height.unsigned_abs())
        } else {
            // No-camera mode: synthetic frames at ~30 Hz.
            std::thread::sleep(Duration::from_millis(33));
            (PREVIEW_WIDTH, PREVIEW_HEIGHT)
        };

        let timestamp_ns = now_ns();
        let hand = dummy_hand(seq, timestamp_ns);
        let frame = Frame {
            seq,
            width,
            height,
            timestamp_ns,
            #[cfg(feature = "debug_preview")]
            rgb: if camera.is_none() { checkerboard() } else { preview(&buf) },
        };
        if tx.send(VisionMsg { frame, hand }).is_err() {
            warn!("vision channel disconnected; stopping vision thread");
            break;
        }
    }
    Ok(())
}

/// Dummy hand data with a fixed open-hand pose, so later phases have
/// something visible to track.
fn dummy_hand(frame_seq: u64, timestamp_ns: u64) -> HandData {
    // MediaPipe hand-landmark layout: 0 = wrist, then thumb (1-4),
    // index (5-8), middle (9-12), ring (13-16), pinky (17-20).
    let mut landmarks = [glam::Vec3::ZERO; 21];
    for (i, lm) in landmarks.iter_mut().enumerate() {
        let t = i as f32 / 20.0;
        *lm = glam::Vec3::new(0.2 + 0.05 * t, -0.1 + 0.3 * t, 0.02 * (i as f32 - 10.0));
    }
    HandData {
        landmarks,
        gesture: Gesture::None,
        frame_seq,
        timestamp_ns,
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Downscales the frame to a 320x240 RGB8 preview (feature-gated).
#[cfg(feature = "debug_preview")]
fn preview(buf: &opencv::core::Mat) -> Vec<u8> {
    use opencv::core::Size;
    use opencv::imgproc::{self, COLOR_BGR2RGB};
    use opencv::prelude::MatTraitConst;
    let mut rgb = opencv::core::Mat::default();
    if imgproc::cvt_color_def(buf, &mut rgb, COLOR_BGR2RGB).is_err() {
        return Vec::new();
    }
    let mut small = opencv::core::Mat::default();
    if imgproc::resize_def(&rgb, &mut small, Size::new(PREVIEW_WIDTH as i32, PREVIEW_HEIGHT as i32))
        .is_err()
    {
        return Vec::new();
    }
    // `Mat::data_bytes` is crate-private in opencv 0.102, so read the raw
    // pixels through the public `MatTraitConst::data` pointer instead.
    let expected = PREVIEW_WIDTH as usize * PREVIEW_HEIGHT as usize * 3;
    let ptr = small.data();
    if ptr.is_null() {
        return Vec::new();
    }
    // Safety: `small` is a continuous 320x240 CV_8UC3 Mat produced by
    // `resize_def` above, so `expected` is its exact byte length.
    unsafe { std::slice::from_raw_parts(ptr, expected) }.to_vec()
}

/// Checkerboard preview for the no-camera mode (feature-gated).
#[cfg(feature = "debug_preview")]
fn checkerboard() -> Vec<u8> {
    let (w, h) = (PREVIEW_WIDTH as usize, PREVIEW_HEIGHT as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / 40) + (y / 40)) % 2 == 0 { 255 } else { 32 };
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    rgb
}

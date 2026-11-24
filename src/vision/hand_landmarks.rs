//! Hand-landmark estimation via the MediaPipe hand-landmarks model.

use anyhow::Result;

/// Placeholder for later phases: estimates hand landmarks in one frame using
/// the MediaPipe hand-landmarks ONNX model through `opencv::dnn`. Later
/// phases return one vector of `(x, y, z)` landmark coordinates per hand.
#[allow(dead_code)]
pub fn detect_landmarks(_frame: &opencv::core::Mat) -> Result<Vec<Vec<[f32; 3]>>> {
    let _ = _frame;
    Ok(Vec::new())
}

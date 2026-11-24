//! Palm/hand detection via the MediaPipe palm-detection model.

use anyhow::Result;

/// Placeholder for later phases: detects palm/hand boxes in one frame using the
/// MediaPipe palm-detection ONNX model through `opencv::dnn`. Later phases
/// return the detected boxes as `(x, y, w, h)` floats in pixel coordinates.
#[allow(dead_code)]
pub fn detect_palms(_frame: &opencv::core::Mat) -> Result<Vec<[f32; 4]>> {
    let _ = _frame;
    Ok(Vec::new())
}

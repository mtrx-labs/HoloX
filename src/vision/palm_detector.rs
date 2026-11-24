//! Palm detection with the OpenCV-Zoo MediaPipe palm detector
//! (`palm_detection_mediapipe_2023feb.onnx`, 2023 Feb).
//!
//! This is a faithful port of OpenCV's official demo `mp_palmdet.py` from
//! `opencv/palm_detection_mediapipe` on HuggingFace: SSD anchors,
//! non-exponential regression, and the same preprocessing/decode math. All
//! constants were verified empirically against the model's actual
//! input/output shapes (recorded in NOTES.md, Phase 2a).

use anyhow::Result;
use opencv::core::{copy_make_border, BORDER_CONSTANT, DataLayout, Mat, Scalar, Size};
use opencv::dnn;
use opencv::imgproc;
use opencv::prelude::*;
use std::path::Path;

/// Number of anchors emitted by the model at 192x192 (introspected; the
/// model's `Identity_1` output has this many elements).
const NUM_ANCHORS: usize = 2016;

/// Values per anchor in the regression output: 4 bbox values + 7 keypoints
/// x 2 coordinates.
const REGRESSION_LEN: usize = 18;

/// Verified anchor spec for this model (GenMediaPipePalmDectionSSDAnchors):
/// see NOTES.md, Phase 2a.
const PALM_STRIDES: [u32; 4] = [8, 16, 16, 16];
const PALM_ASPECT_RATIOS: [f32; 2] = [1.0, 0.5];
const PALM_OFFSET: f32 = 0.5;
const PALM_FIXED_ANCHOR_SIZE: bool = true;
const PALM_MIN_SCALE: f32 = 0.1484375;
const PALM_MAX_SCALE: f32 = 0.75;

/// One SSD anchor: center `(cx, cy)` in `[0,1]` input-image space; `w`/`h`
/// are in `[0,1]` input space too (1.0,1.0 for fixed-size anchors).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Generates the SSD anchors for the palm detector, mirroring
/// `GenMediaPipePalmDectionSSDAnchors`
/// (github.com/VimalMollyn/GenMediaPipePalmDectionSSDAnchors).
pub fn generate_anchors(
    input_size: u32,
    strides: &[u32],
    aspect_ratios: &[f32],
    offset: f32,
    fixed_anchor_size: bool,
    min_scale: f32,
    max_scale: f32,
) -> Vec<Anchor> {
    let num_layers = strides.len();
    let mut anchors = Vec::new();
    let mut layer_id = 0usize;
    while layer_id < num_layers {
        // Per-position widths/heights from this stride group (mirrors the JS).
        let mut anchor_wh: Vec<(f32, f32)> = Vec::new();
        let mut idx = layer_id;
        while idx < strides.len() && strides[idx] == strides[layer_id] {
            let scale = min_scale + (max_scale - min_scale) * idx as f32 / (num_layers as f32 - 1.0);
            for &ar in aspect_ratios {
                let sqrt_ar = ar.sqrt();
                anchor_wh.push((scale * sqrt_ar, scale / sqrt_ar));
            }
            idx += 1;
        }
        let feature = input_size.div_ceil(strides[layer_id]);
        let (fw, fh) = (feature as i32, feature as i32);
        for y in 0..fh {
            for x in 0..fw {
                for &(w, h) in &anchor_wh {
                    let (w, h) = if fixed_anchor_size { (1.0, 1.0) } else { (w, h) };
                    anchors.push(Anchor {
                        cx: (x as f32 + offset) / fw as f32,
                        cy: (y as f32 + offset) / fh as f32,
                        w,
                        h,
                    });
                }
            }
        }
        layer_id = idx;
    }
    anchors
}

/// A detected palm: bbox in original-frame coords + 7 palm keypoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PalmDetection {
    /// Bounding box as `(x_min, y_min, width, height)` in original coords.
    pub bbox: (f32, f32, f32, f32),
    /// Detection score in [0, 1].
    pub score: f32,
    /// 7 palm keypoints in original coords.
    pub keypoints: [(f32, f32); 7],
}

/// One detection candidate before NMS: score + bbox corners.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    keypoints: [(f32, f32); 7],
}

/// Runs the palm detector over frames.
pub struct PalmDetector {
    net: dnn::Net,
    anchors: Vec<Anchor>,
    input_size: u32,
}

impl PalmDetector {
    /// Loads the model and generates anchors; asserts the anchor count
    /// against the introspection result ([`NUM_ANCHORS`]).
    pub fn new(model_path: &Path, input_size: u32) -> Result<Self> {
        let net = dnn::read_net_from_onnx_def(model_path)?;
        let anchors = generate_anchors(
            input_size,
            &PALM_STRIDES,
            &PALM_ASPECT_RATIOS,
            PALM_OFFSET,
            PALM_FIXED_ANCHOR_SIZE,
            PALM_MIN_SCALE,
            PALM_MAX_SCALE,
        );
        anyhow::ensure!(
            anchors.len() == NUM_ANCHORS,
            "generated {} anchors, expected {NUM_ANCHORS} (model introspection)",
            anchors.len()
        );
        Ok(Self {
            net,
            anchors,
            input_size,
        })
    }

    /// Detects the highest-scoring palm in `bgr` (any size). Returns `None`
    /// when nothing passes `score_threshold`.
    pub fn detect(&mut self, bgr: &Mat, score_threshold: f32) -> Result<Option<PalmDetection>> {
        let (rgb, pad_bias) = self.preprocess(bgr)?;
        // Blob: scalefactor 1/255, ddepth CV_32F, no swap (already RGB),
        // NHWC layout — matches the official Python demo's input blob.
        let mut params = dnn::Image2BlobParams::default()?;
        params.scalefactor = Scalar::all(1.0 / 255.0);
        params.datalayout = DataLayout::DATA_LAYOUT_NHWC;
        let blob = dnn::blob_from_image_with_params(&rgb, params)?;
        self.net.set_input(&blob, "", 1.0, Scalar::default())?;

        let out_names = self.net.get_unconnected_out_layers_names()?;
        let mut regression: Option<Mat> = None;
        let mut scores: Option<Mat> = None;
        for name in out_names.iter() {
            let out = self.net.forward_single(&name)?;
            let total = out.total();
            if total == self.anchors.len() {
                scores = Some(out);
            } else if total == self.anchors.len() * REGRESSION_LEN {
                regression = Some(out);
            } else {
                anyhow::bail!("unexpected palm model output size: {total}");
            }
        }
        let (Some(regression), Some(scores)) = (regression, scores) else {
            anyhow::bail!("palm model did not expose both regression and score outputs");
        };

        let regs = mat_f32s(&regression);
        let raw_scores = mat_f32s(&scores);
        let scale = f32::max(bgr.cols() as f32, bgr.rows() as f32);
        let mut candidates: Vec<Candidate> = Vec::new();
        for (i, anchor) in self.anchors.iter().enumerate() {
            let score = sigmoid(*raw_scores.get(i).unwrap_or(&f32::NAN));
            if score < score_threshold {
                continue;
            }
            let row = &regs[i * REGRESSION_LEN..(i + 1) * REGRESSION_LEN];
            // Non-exponential decode (mp_palmdet.py::_postprocess):
            // xy1 = (cxy_delta - wh_delta/2 + anchors) * scale;
            // xy2 = (cxy_delta + wh_delta/2 + anchors) * scale.
            let input_size = self.input_size as f32;
            let x1 = (row[0] / input_size - row[2] / input_size / 2.0 + anchor.cx) * scale
                - pad_bias.0 as f32;
            let y1 = (row[1] / input_size - row[3] / input_size / 2.0 + anchor.cy) * scale
                - pad_bias.1 as f32;
            let x2 = (row[0] / input_size + row[2] / input_size / 2.0 + anchor.cx) * scale
                - pad_bias.0 as f32;
            let y2 = (row[1] / input_size + row[3] / input_size / 2.0 + anchor.cy) * scale
                - pad_bias.1 as f32;
            let mut keypoints = [(0.0f32, 0.0f32); 7];
            for k in 0..7 {
                let kx = (row[4 + 2 * k] / input_size + anchor.cx) * scale - pad_bias.0 as f32;
                let ky = (row[5 + 2 * k] / input_size + anchor.cy) * scale - pad_bias.1 as f32;
                keypoints[k] = (kx, ky);
            }
            candidates.push(Candidate {
                score,
                x1,
                y1,
                x2,
                y2,
                keypoints,
            });
        }
        let kept = nms(&candidates, 0.3);
        let Some(best) = kept.first().copied() else {
            return Ok(None);
        };
        let c = &candidates[best];
        Ok(Some(PalmDetection {
            bbox: (
                c.x1,
                c.y1,
                (c.x2 - c.x1).max(0.0),
                (c.y2 - c.y1).max(0.0),
            ),
            score: c.score,
            keypoints: c.keypoints,
        }))
    }

    /// Official-demo preprocessing (`mp_palmdet.py::_preprocess`):
    /// aspect-preserving resize to the model input + symmetric padding,
    /// then BGR→RGB. Returns the RGB Mat and the padding bias in
    /// original-frame pixels.
    fn preprocess(&self, image: &Mat) -> Result<(Mat, (i32, i32))> {
        let input = self.input_size as i32;
        let (h, w) = (image.rows(), image.cols());
        let ratio = (input as f32 / h as f32).min(input as f32 / w as f32);
        let mut pad_bias = (0i32, 0i32);
        let mut img = Mat::default();
        if h != input || w != input {
            let rh = (h as f32 * ratio) as i32;
            let rw = (w as f32 * ratio) as i32;
            imgproc::resize(image, &mut img, Size::new(rw, rh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
            let (pad_h, pad_w) = (input - rh, input - rw);
            let left = pad_w / 2;
            let top = pad_h / 2;
            let mut padded = Mat::default();
            copy_make_border(
                &img,
                &mut padded,
                top,
                pad_h - top,
                left,
                pad_w - left,
                BORDER_CONSTANT,
                Scalar::default(),
            )?;
            img = padded;
            pad_bias = ((left as f32 / ratio) as i32, (top as f32 / ratio) as i32);
        } else {
            img = image.clone();
        }
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(&img, &mut rgb, imgproc::COLOR_BGR2RGB)?;
        Ok((rgb, pad_bias))
    }
}

/// IoU between two axis-aligned boxes given as `(x1, y1, x2, y2)`.
fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, ax2, ay2) = a;
    let (bx1, by1, bx2, by2) = b;
    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (ax2 - ax1) * (ay2 - ay1);
    let area_b = (bx2 - bx1) * (by2 - by1);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// IoU-based NMS; returns kept indices in descending score order.
fn nms(candidates: &[Candidate], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b].score.total_cmp(&candidates[a].score));
    let mut keep: Vec<usize> = Vec::new();
    while let Some(first) = order.first().copied() {
        keep.push(first);
        order.retain(|&i| {
            if i == first {
                return false;
            }
            let a = (
                candidates[first].x1,
                candidates[first].y1,
                candidates[first].x2,
                candidates[first].y2,
            );
            let b = (
                candidates[i].x1,
                candidates[i].y1,
                candidates[i].x2,
                candidates[i].y2,
            );
            iou(a, b) < iou_threshold
        });
    }
    keep
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reads a CV_32F Mat's data as f32 values (used on model outputs).
fn mat_f32s(mat: &Mat) -> Vec<f32> {
    let ptr = mat.data();
    if ptr.is_null() {
        return Vec::new();
    }
    let n = mat.total();
    // Safety: the Mat is continuous and holds `n` 32-bit float elements
    // (the caller checks `total() == anchors * 18` or `== anchors`).
    unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), n).to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_anchors_matches_introspection() {
        // Verified empirically in Phase 2a against the ONNX model's
        // `Identity_1` output (total = 2016 at 192x192).
        let anchors = generate_anchors(
            192,
            &PALM_STRIDES,
            &PALM_ASPECT_RATIOS,
            PALM_OFFSET,
            PALM_FIXED_ANCHOR_SIZE,
            PALM_MIN_SCALE,
            PALM_MAX_SCALE,
        );
        assert_eq!(anchors.len(), 2016);
        // Spot check against GenMediaPipePalmDectionSSDAnchors: the first
        // anchor center is (0.5, 0.5)/24 on the stride-8 feature map.
        assert_eq!(anchors[0].cx, 0.5 / 24.0);
        assert_eq!(anchors[0].cy, 0.5 / 24.0);
        assert!(anchors.iter().all(|a| a.w == 1.0 && a.h == 1.0));
    }

    #[test]
    fn nms_keeps_best_and_drops_overlaps() {
        fn c(score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Candidate {
            Candidate {
                score,
                x1,
                y1,
                x2,
                y2,
                keypoints: [(0.0, 0.0); 7],
            }
        }
        let candidates = vec![
            c(0.9, 0.0, 0.0, 10.0, 10.0),
            c(0.8, 0.5, 0.0, 10.5, 10.0), // IoU with #0 ≈ 0.90
            c(0.7, 100.0, 100.0, 110.0, 110.0), // disjoint
        ];
        let keep = nms(&candidates, 0.3);
        assert_eq!(keep, vec![0, 2]);
    }
}

//! Runs `PalmDetector` on one image and draws the result (Phase 2a).
//!
//! Usage: `cargo run --release --bin palm_test -- <image_path> [--out <path>]`

use anyhow::Result;
use holox::vision::palm_detector::PalmDetector;
use opencv::core::{Point, Rect, Scalar};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        anyhow::bail!("usage: palm_test <image_path> [--out <path>]");
    };
    let mut out: Option<String> = None;
    let mut threshold: f32 = 0.5;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next(),
            "--threshold" => threshold = args.next().unwrap_or_default().parse()?,
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let image = imgcodecs::imread_def(&path)?;
    if image.rows() == 0 || image.cols() == 0 {
        anyhow::bail!("could not read {path}");
    }

    let mut detector = PalmDetector::new(&PathBuf::from("assets/models/palm_detection_mediapipe_2023feb.onnx"), 192)?;
    let detection = detector.detect(&image, threshold)?;
    match detection {
        Some(det) => {
            println!("score: {:.4}", det.score);
            let (x1, y1, w, h) = det.bbox;
            println!("bbox: x_min={x1:.1} y_min={y1:.1} w={w:.1} h={h:.1}");
            for (i, kp) in det.keypoints.iter().enumerate() {
                println!("  kp[{i}]: ({:.1}, {:.1})", kp.0, kp.1);
            }
            let mut annotated = image.clone();
            imgproc::rectangle(
                &mut annotated,
                Rect::new(x1 as i32, y1 as i32, w as i32, h as i32),
                Scalar::new(0.0, 255.0, 0.0, 0.0), // green (image is BGR)
                2,
                imgproc::LINE_8,
                0,
            )?;
            for kp in &det.keypoints {
                imgproc::circle(
                    &mut annotated,
                    Point::new(kp.0 as i32, kp.1 as i32),
                    4,
                    Scalar::new(0.0, 0.0, 255.0, 0.0), // red
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
            let out_path = out.unwrap_or_else(|| format!("{path}_detected.png"));
            imgcodecs::imwrite_def(out_path.clone(), &annotated)?;
            println!("saved annotated image to {out_path}");
        }
        None => println!("no palm detected in {path}"),
    }
    Ok(())
}

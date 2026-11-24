# HoloX

A simulated, Jarvis-style holographic desktop interface controlled by hand
gestures via a webcam. Everything is written in Rust: rendering via Bevy
(Vulkan by default), computer vision via the OpenCV DNN module running
MediaPipe ONNX models, and a dedicated vision thread talking to the render
loop over `crossbeam-channel`.

## Build & run

```bash
cargo build --release
cargo run --release
```

Useful commands:

- `cargo check` / `cargo clippy -- -D warnings` — the project's quality
  gates; run both with and without `--features debug_preview`.
- `RUST_LOG=holox=info cargo run --release` — live vision logs
  (`vision: received=N dropped=M last_seq=K landmarks=21` about once per
  second).
- `cargo run --release --features debug_preview` — adds a live camera
  preview plus debug rendering (bbox, skeleton, gesture name) in the corner.

## Project layout

- `src/main.rs` — entry point: tracing init, config load, vision thread
  spawn, run the Bevy app.
- `src/config.rs` — `AppConfig` persisted under `~/.config/holox/`.
- `src/vision/` — camera capture (`camera.rs`), palm detection
  (`palm_detector.rs`), hand landmarks (`hand_landmarks.rs`); the OpenCV-free
  `Frame`/`HandData`/`Gesture`/`VisionMsg` types and thread spawn live in
  `mod.rs`.
- `src/app/` — the Bevy app: `VisionRx` / `LatestVision` resources and the
  `drain_vision` system.
- `assets/models/` — MediaPipe ONNX models used by the vision thread
  (gitignored; fetched by the download script in Phase 2).
- `NOTES.md` — the phase-by-phase build log, including model sha256s.

## Phases

The build is phased; see `NOTES.md` for status:

- **Phase 0** — scaffolding (Bevy app skeleton, config, module structure).
- **Phase 1** — live camera capture + vision channel plumbing.
- **Phase 2** — MediaPipe hand tracking: palm detection + landmark
  inference (in progress).
- **Phase 3+** — holographic UI, gestures in use, OS control — planned.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE).

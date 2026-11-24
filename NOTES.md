# HoloX — Notes

Build log for HoloX, a simulated, Jarvis-style holographic desktop interface
controlled by hand gestures via a webcam (Linux, Vulkan rendering).

## Phase 0 — Scaffolding

- `cargo init --name holox`; explicit `[[bin]] name = "holox" path = "src/main.rs"`
  so `cargo run --release` produces `target/release/holox`.
- Dependencies added via `cargo add` (versions resolved from crates.io on
  2026-11-24):
  - bevy v0.20.1 (default features; Vulkan backend)
  - opencv v0.102.0 with feature `clang-runtime` (system OpenCV via pkg-config;
    pulls clang v2.1.0, clang-sys v1.9.1, opencv-binding-generator v0.108.0)
  - crossbeam-channel v0.5.17, once_cell v1.21.4, anyhow v1.0.105
  - tracing v0.1.44, tracing-subscriber v0.3.23 (env-filter feature)
  - serde v1.0.229 (derive), serde_json v1.0.154, toml v1.2.2
  - dirs v7.0.0 (config paths under `~/.config/holox/`)
  - glam v0.33.12 (matches the glam version Bevy 0.20.1 uses)
- Module skeleton:
  - `src/main.rs` — entry point: init tracing, load config, spawn the vision
    thread, then `app::run()`.
  - `src/config.rs` — `AppConfig`, `config_root()` → `~/.config/holox/`,
    `load()`/`save()` via TOML, global `CONFIG_ROOT` (once_cell).
  - `src/vision/mod.rs` — `VisionEvent`, `spawn_vision_thread()` on a
    dedicated named `std::thread` (`holox-vision`) with an unbounded
    `crossbeam-channel` receiver.
  - `src/vision/camera.rs` — `CAMERA_PATH` (`/dev/video0`), `open_camera()`
    placeholder.
  - `src/vision/palm_detector.rs` — `detect_palms()` placeholder.
  - `src/vision/hand_landmarks.rs` — `detect_landmarks()` placeholder.
  - `src/app/mod.rs` — `run()` placeholder (Bevy scaffold).
- `cargo check` passes; `cargo build --release` verified — produces
  `target/release/holox` (~95 MB, 2026-11-24).

### Deferred to later phases

- Real camera loop (`/dev/video0` via OpenCV `videoio`) in the vision thread.
- MediaPipe ONNX models: palm detection + hand landmarks via `opencv::dnn`.
- The Bevy holographic desktop (render loop, gesture state machine,
  calibration tooling).
- Verification with `cargo build --release`.

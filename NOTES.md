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

## Phase 1 — Camera capture & vision channel plumbing

- `src/vision/camera.rs`: real `Camera` wrapping `opencv::videoio::VideoCapture`:
  - `Camera::open(index)` uses `videoio::CAP_V4L2` (reliable path on CachyOS)
    and requests 640x480 @ 30 FPS via `CAP_PROP_FRAME_WIDTH/HEIGHT/FPS`
    **before** the first read, then reads the accepted values back and logs
    them (V4L2 often negotiates silently; observed on this host:
    `camera opened (V4L2) index=0 width=640 height=480 fps=30.0`).
  - `next_frame(&mut self, buf: &mut Mat)` reads into a reused buffer;
    `VideoCapture::read` blocks, so the thread self-throttles to the camera
    FPS (observed ~30 Hz on `/dev/video0`).
- `src/vision/mod.rs`: OpenCV-free `Frame` / `HandData` / `Gesture` /
  `VisionMsg`; `dummy_hand()` returns a fixed open-hand pose (MediaPipe
  layout: wrist + thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20)
  with `gesture: Gesture::None`.
- `src/vision/mod.rs::spawn_vision_thread(camera_index, tx)` spawns the
  dedicated `holox-vision` thread; on camera-open failure it logs and enters
  a no-camera mode that emits synthetic frames at ~30 Hz.
- `src/app/mod.rs`:
  - `VisionRx(pub Receiver<VisionMsg>)` resource (crossbeam 0.5.17's
    `Receiver` is `Send + Sync`).
  - `LatestVision { frame, hand, received, dropped, disconnected }` resource.
  - `drain_vision` system: loops `try_recv()` until `Empty`/`Disconnected`,
    keeps only the last message of each batch, counts older ones as `dropped`,
    and logs `vision: received=N dropped=M last_seq=K landmarks=21` about
    once per second via `Time::elapsed_secs`.
  - `run(rx)` builds the Bevy App (`DefaultPlugins`, UI included) and runs it.
- `set_window_title`: primary window title set to "HoloX"
  (`Query<&mut Window, With<PrimaryWindow>>`).
- Logger: `main` initializes our `tracing_subscriber`; bevy's `LogPlugin` is
  disabled to avoid the "Could not set global logger" error.
- Debug preview (feature-gated): `Frame.rgb` (RGB8, downscaled to 320x240) is
  filled on the vision thread; `preview_system` builds a `bevy::image::Image`
  and updates an `ImageNode` spawned bottom-right. Byte extraction uses the
  public `MatTraitConst::data` pointer because `Mat::data_bytes` is
  crate-private in opencv 0.102.

### Runtime sample (`RUST_LOG=holox=info`, ~5s)

    2026-11-24T05:24:23.2 INFO holox: HoloX (phase 1)
    2026-11-24T05:24:23.3 INFO holox::config: wrote default holox config path=/home/mtrx/.config/holox/config.toml
    2026-11-24T05:24:23.4 INFO holox::vision::camera: camera opened (V4L2) index=0 width=640 height=480 fps=30.0
    2026-11-24T05:24:23.4 INFO holox::vision: vision thread started camera_index=0
    2026-11-24T05:24:24.3 INFO holox::app: vision: received=15 dropped=0 last_seq=15 landmarks=21
    2026-11-24T05:24:25.3 INFO holox::app: vision: received=46 dropped=0 last_seq=46 landmarks=21
    2026-11-24T05:24:26.3 INFO holox::app: vision: received=77 dropped=0 last_seq=77 landmarks=21
    2026-11-24T05:24:27.3 INFO holox::app: vision: received=107 dropped=0 last_seq=107 landmarks=21

### Deferred (Phase 1)

- **Unplug test**: this sandbox has no way to physically unplug the camera, so
  the `Disconnected` path is documented/inspected but not exercised live. It
  is wired like this: on camera-open failure or a read error the vision loop
  exits and drops the sender; `drain_vision` then logs
  `vision channel disconnected` exactly once and sets
  `LatestVision::disconnected = true` — no panic, no log spam, render loop
  unaffected.
- Real hand tracking (`palm_detector.rs` + `hand_landmarks.rs`) — Phase 2+.
- ONNX/MediaPipe models, calibration tooling, 60 FPS verification,
  live verification of the debug preview — later.

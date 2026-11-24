# Phase 2a test fixtures

Frames captured from the camera with
`cargo run --release --bin grab_frames` (`--count 10` by default,
`--outdir assets/test`).

The `*_detected.png` files next to them are annotated by
`cargo run --release --bin palm_test` (green bbox + red keypoint circles).

Detection scores are recorded in NOTES.md (Phase 2a): the best-scoring palm
per fixture was ~0.025 — no detections reached the default 0.5 threshold in
this environment.

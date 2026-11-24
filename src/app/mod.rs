//! Bevy-based holographic desktop application (Vulkan default backend).

use anyhow::Result;
use bevy::ecs::system::Local;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::time::Time;
use bevy::window::{PrimaryWindow, Window};
use crossbeam_channel::{Receiver, TryRecvError};
use tracing::info;

#[cfg(feature = "debug_preview")]
use bevy::asset::{Assets, RenderAssetUsages};
#[cfg(feature = "debug_preview")]
use bevy::image::Image;
#[cfg(feature = "debug_preview")]
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
#[cfg(feature = "debug_preview")]
use bevy::ui::prelude::ImageNode;
#[cfg(feature = "debug_preview")]
use bevy::ui::{Node, PositionType, Val};

use crate::vision::{Frame, HandData, VisionMsg};

#[cfg(feature = "debug_preview")]
use crate::vision::{PREVIEW_HEIGHT, PREVIEW_WIDTH};

/// Receives `VisionMsg`s from the vision thread (unbounded channel).
#[derive(Resource)]
pub struct VisionRx(pub Receiver<VisionMsg>);

/// Latest state consumed by the Bevy side; see the `drain_vision` system.
#[derive(Resource, Default)]
pub struct LatestVision {
    pub frame: Option<Frame>,
    pub hand: Option<HandData>,
    pub received: u64,
    pub dropped: u64,
    pub disconnected: bool,
}

/// Builds and runs the Bevy app with the vision resources wired in.
pub fn run(rx: Receiver<VisionMsg>) -> Result<()> {
    let mut app = App::new();
    // Our `tracing_subscriber` is initialized in `main`; disable bevy's
    // `LogPlugin` so both don't try to set the global logger.
    app.add_plugins(DefaultPlugins.build().disable::<LogPlugin>());
    app.insert_resource(VisionRx(rx));
    app.insert_resource(LatestVision::default());
    app.add_systems(Startup, set_window_title);
    app.add_systems(Update, drain_vision);
    #[cfg(feature = "debug_preview")]
    {
        app.add_systems(Startup, spawn_preview_node);
        app.add_systems(Update, preview_system);
    }
    app.run();
    Ok(())
}

/// Sets the primary window title to "HoloX".
fn set_window_title(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    for mut window in windows.iter_mut() {
        window.title = "HoloX".into();
    }
}

/// Drains the vision channel every frame: keeps only the last message of each
/// batch, updates [`LatestVision`], and logs roughly once per second.
fn drain_vision(
    rx: Res<VisionRx>,
    mut latest: ResMut<LatestVision>,
    time: Res<Time>,
    mut last_log: Local<f32>,
    mut logged_disconnect: Local<bool>,
) {
    let mut batch = 0u64;
    let mut last: Option<VisionMsg> = None;
    loop {
        match rx.0.try_recv() {
            Ok(msg) => {
                latest.received += 1;
                batch += 1;
                last = Some(msg);
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                latest.disconnected = true;
                if !*logged_disconnect {
                    info!("vision channel disconnected");
                    *logged_disconnect = true;
                }
                break;
            }
        }
    }
    if let Some(msg) = last {
        let seq = msg.frame.seq;
        latest.dropped += batch.saturating_sub(1);
        latest.frame = Some(msg.frame);
        latest.hand = Some(msg.hand);
        let now = time.elapsed_secs();
        if now - *last_log >= 1.0 {
            let landmarks = latest.hand.as_ref().map_or(0, |h| h.landmarks.len());
            info!(
                "vision: received={} dropped={} last_seq={} landmarks={}",
                latest.received, latest.dropped, seq, landmarks
            );
            *last_log = now;
        }
    }
}

/// Spawns the debug preview UI node (feature-gated).
#[cfg(feature = "debug_preview")]
fn spawn_preview_node(mut commands: Commands) {
    commands.spawn((
        ImageNode::default(),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Auto,
            right: Val::Px(0.0),
            bottom: Val::Px(0.0),
            width: Val::Px(crate::vision::PREVIEW_WIDTH as f32),
            height: Val::Px(crate::vision::PREVIEW_HEIGHT as f32),
            ..Default::default()
        },
    ));
}

/// Updates the debug preview image when the latest frame changes (feature-gated).
#[cfg(feature = "debug_preview")]
fn preview_system(
    latest: Res<LatestVision>,
    mut images: ResMut<Assets<Image>>,
    mut nodes: Query<&mut ImageNode>,
    mut last_seq: Local<Option<u64>>,
) {
    let Some(frame) = latest.frame.as_ref() else {
        return;
    };
    if frame.rgb.is_empty() || *last_seq == Some(frame.seq) {
        return;
    }
    // The image holds the preview (`Frame.rgb`), which `preview()` produced
    // at PREVIEW_WIDTH x PREVIEW_HEIGHT; `Frame.width`/`height` keep the
    // camera's accepted dimensions.
    let extent = Extent3d {
        width: PREVIEW_WIDTH,
        height: PREVIEW_HEIGHT,
        depth_or_array_layers: 1,
    };
    let mut rgba = Vec::with_capacity(frame.rgb.len() * 4 / 3 + 4);
    for px in frame.rgb.as_chunks::<3>().0 {
        rgba.extend_from_slice(px);
        rgba.push(255);
    }
    let image = Image::new(
        extent,
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    let handle = images.add(image);
    for mut node in nodes.iter_mut() {
        node.image = handle.clone();
    }
    *last_seq = Some(frame.seq);
}

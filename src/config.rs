//! Application configuration, persisted under `~/.config/holox/`.

use anyhow::{Context, Result};
use dirs::config_dir;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

/// Root directory for HoloX user configuration, resolved as `~/.config/holox/`.
pub fn config_root() -> PathBuf {
    match config_dir() {
        Some(dir) => dir.join("holox"),
        None => PathBuf::from(".holox"),
    }
}

/// Lazily resolved config root, shared across the process (later phases read
/// the global config from it).
#[allow(dead_code)]
pub static CONFIG_ROOT: Lazy<PathBuf> = Lazy::new(config_root);

/// User-facing settings. Defaults are written on first run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Display name of the application.
    pub app_name: String,
    /// Directory where calibration data is stored.
    pub calibration_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app_name: "HoloX".to_string(),
            calibration_dir: config_root().join("calibration"),
        }
    }
}

/// Loads the config from `<config_root>/config.toml`, writing defaults if absent.
pub fn load() -> Result<AppConfig> {
    let path = config_root().join("config.toml");
    if path.is_file() {
        let text = std::fs::read_to_string(&path).context("read holox config")?;
        let cfg: AppConfig = toml::from_str(&text).context("parse holox config")?;
        info!(path = %path.display(), "loaded holox config");
        Ok(cfg)
    } else {
        let cfg = AppConfig::default();
        save(&cfg)?;
        info!(path = %path.display(), "wrote default holox config");
        Ok(cfg)
    }
}

/// Persists the config to `<config_root>/config.toml`.
pub fn save(cfg: &AppConfig) -> Result<()> {
    let root = config_root();
    std::fs::create_dir_all(&root).context("create holox config dir")?;
    let path = root.join("config.toml");
    std::fs::write(&path, toml::to_string_pretty(cfg).context("serialize holox config")?)
        .context("write holox config")?;
    info!(path = %path.display(), "saved holox config");
    Ok(())
}

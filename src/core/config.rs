use anyhow::Result;
use std::path::PathBuf;

use super::types::KbConfig;

/// Return the path to the configuration file (~/.kb.conf).
pub fn config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?;
    Ok(home.join(".kb.conf"))
}

/// Load the configuration from disk, returning a default if the file
/// does not exist.
pub fn load_config() -> Result<KbConfig> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(KbConfig::default());
    }
    let content = std::fs::read_to_string(&path)?;
    let config: KbConfig = toml::from_str(&content)?;
    Ok(config)
}

/// Save the configuration to disk.
pub fn save_config(config: &KbConfig) -> Result<()> {
    let path = config_path()?;
    let content = toml::to_string_pretty(config)?;
    std::fs::write(&path, content)?;
    Ok(())
}

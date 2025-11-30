//! Sovereign Distribution Manifest (ENT-016)
//!
//! Provides distribution packaging for air-gapped deployment scenarios,
//! like old-school Linux .ISO hosting at universities.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Distribution tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DistributionTier {
    /// ~50MB: entrenar-core, trueno, aprender
    #[default]
    Core,
    /// ~200MB: + renacer, trueno-db, ruchy
    Standard,
    /// ~500MB: + GPU support, all tooling
    Full,
}

impl DistributionTier {
    /// Get the approximate size in megabytes
    pub fn approximate_size_mb(&self) -> u64 {
        match self {
            Self::Core => 50,
            Self::Standard => 200,
            Self::Full => 500,
        }
    }

    /// Get the core component names for this tier
    pub fn component_names(&self) -> Vec<&'static str> {
        match self {
            Self::Core => vec!["entrenar-core", "trueno", "aprender"],
            Self::Standard => vec![
                "entrenar-core",
                "trueno",
                "aprender",
                "renacer",
                "trueno-db",
                "ruchy",
            ],
            Self::Full => vec![
                "entrenar-core",
                "trueno",
                "aprender",
                "renacer",
                "trueno-db",
                "ruchy",
                "entrenar-gpu",
                "entrenar-bench",
                "entrenar-inspect",
                "entrenar-lora",
                "entrenar-shell",
            ],
        }
    }

    /// Check if this tier includes a specific component
    pub fn includes(&self, component: &str) -> bool {
        self.component_names().contains(&component)
    }
}

impl std::fmt::Display for DistributionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Standard => write!(f, "standard"),
            Self::Full => write!(f, "full"),
        }
    }
}

/// Distribution format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DistributionFormat {
    /// Bootable ISO with NixOS
    Iso,
    /// OCI container image
    Oci,
    /// Nix flake
    Nix,
    /// Flatpak bundle
    Flatpak,
    /// Simple tar.gz
    #[default]
    Tarball,
}

impl DistributionFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Iso => "iso",
            Self::Oci => "tar",
            Self::Nix => "nix",
            Self::Flatpak => "flatpak",
            Self::Tarball => "tar.gz",
        }
    }

    /// Get MIME type for the format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Iso => "application/x-iso9660-image",
            Self::Oci => "application/vnd.oci.image.layer.v1.tar",
            Self::Nix => "text/plain",
            Self::Flatpak => "application/vnd.flatpak",
            Self::Tarball => "application/gzip",
        }
    }
}

impl std::fmt::Display for DistributionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Iso => write!(f, "ISO"),
            Self::Oci => write!(f, "OCI"),
            Self::Nix => write!(f, "Nix"),
            Self::Flatpak => write!(f, "Flatpak"),
            Self::Tarball => write!(f, "Tarball"),
        }
    }
}

/// Individual component manifest
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentManifest {
    /// Component name
    pub name: String,
    /// Component version
    pub version: String,
    /// Crate name on crates.io
    pub crate_name: String,
    /// Enabled features
    pub features: Vec<String>,
}

impl ComponentManifest {
    /// Create a new component manifest
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        crate_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            crate_name: crate_name.into(),
            features: Vec::new(),
        }
    }

    /// Add features to the component
    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.features = features.into_iter().map(Into::into).collect();
        self
    }

    /// Create component for entrenar-core
    pub fn entrenar_core(version: &str) -> Self {
        Self::new("entrenar-core", version, "entrenar")
    }

    /// Create component for trueno
    pub fn trueno(version: &str) -> Self {
        Self::new("trueno", version, "trueno")
    }

    /// Create component for aprender
    pub fn aprender(version: &str) -> Self {
        Self::new("aprender", version, "aprender")
    }

    /// Create component for renacer
    pub fn renacer(version: &str) -> Self {
        Self::new("renacer", version, "renacer")
    }
}

/// Sovereign distribution manifest
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SovereignDistribution {
    /// Distribution name
    pub name: String,
    /// Distribution version
    pub version: String,
    /// Distribution tier
    pub tier: DistributionTier,
    /// Distribution format
    pub format: DistributionFormat,
    /// Component manifests
    pub components: Vec<ComponentManifest>,
    /// SHA-256 checksum of bundle
    pub checksum: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl SovereignDistribution {
    /// Create a new distribution manifest
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        tier: DistributionTier,
        format: DistributionFormat,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tier,
            format,
            components: Vec::new(),
            checksum: String::new(),
            created_at: Utc::now(),
        }
    }

    /// Create Core tier distribution (~50MB)
    pub fn core() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-core",
            version,
            DistributionTier::Core,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2"),
            ComponentManifest::aprender("0.1"),
        ];

        dist
    }

    /// Create Standard tier distribution (~200MB)
    pub fn standard() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-standard",
            version,
            DistributionTier::Standard,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2"),
            ComponentManifest::aprender("0.1"),
            ComponentManifest::renacer("0.1"),
            ComponentManifest::new("trueno-db", "0.1", "trueno-db"),
            ComponentManifest::new("ruchy", "0.1", "ruchy"),
        ];

        dist
    }

    /// Create Full tier distribution (~500MB)
    pub fn full() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-full",
            version,
            DistributionTier::Full,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2").with_features(["gpu", "cuda"]),
            ComponentManifest::aprender("0.1"),
            ComponentManifest::renacer("0.1"),
            ComponentManifest::new("trueno-db", "0.1", "trueno-db"),
            ComponentManifest::new("ruchy", "0.1", "ruchy"),
            ComponentManifest::new("entrenar-gpu", version, "entrenar-gpu")
                .with_features(["cuda", "rocm"]),
            ComponentManifest::new("entrenar-bench", version, "entrenar-bench"),
            ComponentManifest::new("entrenar-inspect", version, "entrenar-inspect"),
            ComponentManifest::new("entrenar-lora", version, "entrenar-lora"),
            ComponentManifest::new("entrenar-shell", version, "entrenar-shell"),
        ];

        dist
    }

    /// Set the distribution format
    pub fn with_format(mut self, format: DistributionFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the checksum
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = checksum.into();
        self
    }

    /// Calculate and set checksum from data
    pub fn compute_checksum(&mut self, data: &[u8]) {
        let mut hasher = Sha256::new();
        hasher.update(data);
        self.checksum = format!("{:x}", hasher.finalize());
    }

    /// Verify checksum against data
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        if self.checksum.is_empty() {
            return false;
        }

        let mut hasher = Sha256::new();
        hasher.update(data);
        let computed = format!("{:x}", hasher.finalize());

        computed == self.checksum
    }

    /// Serialize to JSON manifest
    pub fn to_manifest_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse from JSON manifest
    pub fn from_manifest_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the suggested filename for this distribution
    pub fn suggested_filename(&self) -> String {
        format!("{}-{}.{}", self.name, self.version, self.format.extension())
    }

    /// Get total component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Check if distribution includes a specific component
    pub fn has_component(&self, name: &str) -> bool {
        self.components.iter().any(|c| c.name == name)
    }
}

impl Default for SovereignDistribution {
    fn default() -> Self {
        Self::core()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_tier_core_components() {
        let core = DistributionTier::Core;
        let components = core.component_names();

        assert_eq!(components.len(), 3);
        assert!(components.contains(&"entrenar-core"));
        assert!(components.contains(&"trueno"));
        assert!(components.contains(&"aprender"));
    }

    #[test]
    fn test_distribution_tier_standard_components() {
        let standard = DistributionTier::Standard;
        let components = standard.component_names();

        assert_eq!(components.len(), 6);
        assert!(components.contains(&"entrenar-core"));
        assert!(components.contains(&"renacer"));
        assert!(components.contains(&"trueno-db"));
        assert!(components.contains(&"ruchy"));
    }

    #[test]
    fn test_distribution_tier_full_components() {
        let full = DistributionTier::Full;
        let components = full.component_names();

        assert_eq!(components.len(), 11);
        assert!(components.contains(&"entrenar-gpu"));
        assert!(components.contains(&"entrenar-bench"));
        assert!(components.contains(&"entrenar-lora"));
    }

    #[test]
    fn test_distribution_tier_sizes() {
        assert_eq!(DistributionTier::Core.approximate_size_mb(), 50);
        assert_eq!(DistributionTier::Standard.approximate_size_mb(), 200);
        assert_eq!(DistributionTier::Full.approximate_size_mb(), 500);
    }

    #[test]
    fn test_distribution_tier_includes() {
        let core = DistributionTier::Core;
        assert!(core.includes("trueno"));
        assert!(!core.includes("renacer"));

        let standard = DistributionTier::Standard;
        assert!(standard.includes("renacer"));
        assert!(!standard.includes("entrenar-gpu"));

        let full = DistributionTier::Full;
        assert!(full.includes("entrenar-gpu"));
    }

    #[test]
    fn test_distribution_format_extensions() {
        assert_eq!(DistributionFormat::Iso.extension(), "iso");
        assert_eq!(DistributionFormat::Oci.extension(), "tar");
        assert_eq!(DistributionFormat::Nix.extension(), "nix");
        assert_eq!(DistributionFormat::Flatpak.extension(), "flatpak");
        assert_eq!(DistributionFormat::Tarball.extension(), "tar.gz");
    }

    #[test]
    fn test_distribution_format_mime_types() {
        assert_eq!(
            DistributionFormat::Iso.mime_type(),
            "application/x-iso9660-image"
        );
        assert_eq!(
            DistributionFormat::Oci.mime_type(),
            "application/vnd.oci.image.layer.v1.tar"
        );
    }

    #[test]
    fn test_component_manifest_new() {
        let comp = ComponentManifest::new("test", "1.0.0", "test-crate");

        assert_eq!(comp.name, "test");
        assert_eq!(comp.version, "1.0.0");
        assert_eq!(comp.crate_name, "test-crate");
        assert!(comp.features.is_empty());
    }

    #[test]
    fn test_component_manifest_with_features() {
        let comp =
            ComponentManifest::new("test", "1.0.0", "test-crate").with_features(["gpu", "cuda"]);

        assert_eq!(comp.features.len(), 2);
        assert!(comp.features.contains(&"gpu".to_string()));
        assert!(comp.features.contains(&"cuda".to_string()));
    }

    #[test]
    fn test_sovereign_distribution_core() {
        let dist = SovereignDistribution::core();

        assert_eq!(dist.tier, DistributionTier::Core);
        assert_eq!(dist.components.len(), 3);
        assert!(dist.has_component("entrenar-core"));
        assert!(dist.has_component("trueno"));
        assert!(dist.has_component("aprender"));
    }

    #[test]
    fn test_sovereign_distribution_standard() {
        let dist = SovereignDistribution::standard();

        assert_eq!(dist.tier, DistributionTier::Standard);
        assert_eq!(dist.components.len(), 6);
        assert!(dist.has_component("renacer"));
    }

    #[test]
    fn test_sovereign_distribution_full() {
        let dist = SovereignDistribution::full();

        assert_eq!(dist.tier, DistributionTier::Full);
        assert_eq!(dist.components.len(), 11);
        assert!(dist.has_component("entrenar-gpu"));
    }

    #[test]
    fn test_sovereign_distribution_checksum() {
        let mut dist = SovereignDistribution::core();
        let data = b"test bundle content";

        dist.compute_checksum(data);
        assert!(!dist.checksum.is_empty());
        assert!(dist.verify_checksum(data));
        assert!(!dist.verify_checksum(b"different content"));
    }

    #[test]
    fn test_sovereign_distribution_empty_checksum_fails() {
        let dist = SovereignDistribution::core();
        assert!(!dist.verify_checksum(b"any content"));
    }

    #[test]
    fn test_sovereign_distribution_to_manifest_json() {
        let dist = SovereignDistribution::core();
        let json = dist.to_manifest_json();

        assert!(json.contains("entrenar-sovereign-core"));
        assert!(json.contains("\"tier\": \"Core\"") || json.contains("\"tier\":\"Core\""));

        // Verify round-trip
        let parsed = SovereignDistribution::from_manifest_json(&json).unwrap();
        assert_eq!(parsed.name, dist.name);
        assert_eq!(parsed.tier, dist.tier);
    }

    #[test]
    fn test_sovereign_distribution_suggested_filename() {
        let dist = SovereignDistribution::core();
        let filename = dist.suggested_filename();

        assert!(filename.starts_with("entrenar-sovereign-core-"));
        assert!(filename.ends_with(".tar.gz"));
    }

    #[test]
    fn test_sovereign_distribution_with_format() {
        let dist = SovereignDistribution::core().with_format(DistributionFormat::Iso);

        assert_eq!(dist.format, DistributionFormat::Iso);
        assert!(dist.suggested_filename().ends_with(".iso"));
    }

    #[test]
    fn test_distribution_tier_display() {
        assert_eq!(format!("{}", DistributionTier::Core), "core");
        assert_eq!(format!("{}", DistributionTier::Standard), "standard");
        assert_eq!(format!("{}", DistributionTier::Full), "full");
    }

    #[test]
    fn test_distribution_format_display() {
        assert_eq!(format!("{}", DistributionFormat::Iso), "ISO");
        assert_eq!(format!("{}", DistributionFormat::Oci), "OCI");
        assert_eq!(format!("{}", DistributionFormat::Nix), "Nix");
    }

    #[test]
    fn test_distribution_serialization() {
        let dist = SovereignDistribution::full();
        let json = serde_json::to_string(&dist).unwrap();
        let parsed: SovereignDistribution = serde_json::from_str(&json).unwrap();

        assert_eq!(dist.name, parsed.name);
        assert_eq!(dist.tier, parsed.tier);
        assert_eq!(dist.components.len(), parsed.components.len());
    }
}

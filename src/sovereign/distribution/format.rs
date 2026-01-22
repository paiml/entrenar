//! Distribution format options

use serde::{Deserialize, Serialize};

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

//! Software license types.

use serde::{Deserialize, Serialize};

/// Software license
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum License {
    /// MIT License
    Mit,
    /// Apache License 2.0
    Apache2,
    /// BSD 3-Clause
    Bsd3,
    /// GNU GPL v3
    Gpl3,
    /// Creative Commons Attribution 4.0
    CcBy4,
    /// Creative Commons Zero (public domain)
    Cc0,
    /// Custom license with SPDX identifier
    Custom(String),
}

impl std::fmt::Display for License {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mit => write!(f, "MIT"),
            Self::Apache2 => write!(f, "Apache-2.0"),
            Self::Bsd3 => write!(f, "BSD-3-Clause"),
            Self::Gpl3 => write!(f, "GPL-3.0"),
            Self::CcBy4 => write!(f, "CC-BY-4.0"),
            Self::Cc0 => write!(f, "CC0-1.0"),
            Self::Custom(spdx) => write!(f, "{spdx}"),
        }
    }
}

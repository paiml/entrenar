//! Contributor roles following the CRediT taxonomy.

use serde::{Deserialize, Serialize};

/// Contributor roles following the CRediT (Contributor Roles Taxonomy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContributorRole {
    /// Ideas; formulation or evolution of overarching research goals and aims
    Conceptualization,
    /// Management activities to annotate, scrub data and maintain research data
    DataCuration,
    /// Application of statistical, mathematical, computational techniques
    FormalAnalysis,
    /// Acquisition of financial support for the project
    FundingAcquisition,
    /// Conducting research and investigation process
    Investigation,
    /// Development or design of methodology
    Methodology,
    /// Management and coordination responsibility
    ProjectAdministration,
    /// Provision of study materials, reagents, materials, laboratory samples
    Resources,
    /// Programming, software development; designing computer programs
    Software,
    /// Oversight and leadership responsibility
    Supervision,
    /// Verification of results/experiments
    Validation,
    /// Preparation, creation and/or presentation of data visualization
    Visualization,
    /// Preparation and creation of the published work (original draft)
    WritingOriginal,
    /// Critical review, commentary or revision
    WritingReview,
}

impl std::fmt::Display for ContributorRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conceptualization => write!(f, "Conceptualization"),
            Self::DataCuration => write!(f, "Data curation"),
            Self::FormalAnalysis => write!(f, "Formal analysis"),
            Self::FundingAcquisition => write!(f, "Funding acquisition"),
            Self::Investigation => write!(f, "Investigation"),
            Self::Methodology => write!(f, "Methodology"),
            Self::ProjectAdministration => write!(f, "Project administration"),
            Self::Resources => write!(f, "Resources"),
            Self::Software => write!(f, "Software"),
            Self::Supervision => write!(f, "Supervision"),
            Self::Validation => write!(f, "Validation"),
            Self::Visualization => write!(f, "Visualization"),
            Self::WritingOriginal => write!(f, "Writing – original draft"),
            Self::WritingReview => write!(f, "Writing – review & editing"),
        }
    }
}

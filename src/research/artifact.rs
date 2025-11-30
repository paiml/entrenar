//! Research Artifact and Author structs (ENT-019)
//!
//! Provides core types for academic research artifacts with proper
//! attribution using CRediT taxonomy, ORCID, and ROR identifiers.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// ORCID validation pattern: 0000-0000-0000-000X
static ORCID_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4}-){3}\d{3}[\dX]$").expect("Invalid ORCID regex"));

/// ROR ID validation pattern: https://ror.org/xxxxxxxxx
static ROR_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^https://ror\.org/[a-z0-9]{9}$").expect("Invalid ROR regex"));

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

/// Institutional affiliation with optional ROR identifier
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Affiliation {
    /// Institution name
    pub name: String,
    /// Research Organization Registry ID (optional)
    pub ror_id: Option<String>,
    /// Country code (ISO 3166-1 alpha-2)
    pub country: Option<String>,
}

impl Affiliation {
    /// Create a new affiliation
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ror_id: None,
            country: None,
        }
    }

    /// Set the ROR ID (validates format)
    pub fn with_ror_id(mut self, ror_id: impl Into<String>) -> Result<Self, ValidationError> {
        let ror = ror_id.into();
        if !validate_ror_id(&ror) {
            return Err(ValidationError::InvalidRorId(ror));
        }
        self.ror_id = Some(ror);
        Ok(self)
    }

    /// Set the country code
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }
}

/// Author with ORCID and contributor roles
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Author {
    /// Full name
    pub name: String,
    /// ORCID identifier (optional)
    pub orcid: Option<String>,
    /// Institutional affiliations
    pub affiliations: Vec<Affiliation>,
    /// Contributor roles (CRediT taxonomy)
    pub roles: Vec<ContributorRole>,
}

impl Author {
    /// Create a new author
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            orcid: None,
            affiliations: Vec::new(),
            roles: Vec::new(),
        }
    }

    /// Set the ORCID (validates format)
    pub fn with_orcid(mut self, orcid: impl Into<String>) -> Result<Self, ValidationError> {
        let id = orcid.into();
        if !validate_orcid(&id) {
            return Err(ValidationError::InvalidOrcid(id));
        }
        self.orcid = Some(id);
        Ok(self)
    }

    /// Add an affiliation
    pub fn with_affiliation(mut self, affiliation: Affiliation) -> Self {
        self.affiliations.push(affiliation);
        self
    }

    /// Add a contributor role
    pub fn with_role(mut self, role: ContributorRole) -> Self {
        if !self.roles.contains(&role) {
            self.roles.push(role);
        }
        self
    }

    /// Add multiple contributor roles
    pub fn with_roles(mut self, roles: impl IntoIterator<Item = ContributorRole>) -> Self {
        for role in roles {
            if !self.roles.contains(&role) {
                self.roles.push(role);
            }
        }
        self
    }

    /// Get the author's last name (for citation keys)
    pub fn last_name(&self) -> &str {
        self.name
            .split_whitespace()
            .next_back()
            .unwrap_or(&self.name)
    }
}

/// Type of research artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Structured data collection
    Dataset,
    /// Trained model weights
    Model,
    /// Source code
    Code,
    /// Academic paper or preprint
    Paper,
    /// Jupyter or computational notebook
    Notebook,
    /// Computational workflow (e.g., Snakemake, CWL)
    Workflow,
}

impl std::fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "Dataset"),
            Self::Model => write!(f, "Model"),
            Self::Code => write!(f, "Code"),
            Self::Paper => write!(f, "Paper"),
            Self::Notebook => write!(f, "Notebook"),
            Self::Workflow => write!(f, "Workflow"),
        }
    }
}

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

/// Research artifact with full metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchArtifact {
    /// Unique identifier
    pub id: String,
    /// Artifact title
    pub title: String,
    /// Authors with affiliations and roles
    pub authors: Vec<Author>,
    /// Type of artifact
    pub artifact_type: ArtifactType,
    /// License
    pub license: License,
    /// Digital Object Identifier (optional)
    pub doi: Option<String>,
    /// Version string
    pub version: String,
    /// Abstract or description
    pub description: Option<String>,
    /// Keywords for discovery
    pub keywords: Vec<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ResearchArtifact {
    /// Create a new research artifact
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        artifact_type: ArtifactType,
        license: License,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            authors: Vec::new(),
            artifact_type,
            license,
            doi: None,
            version: "1.0.0".to_string(),
            description: None,
            keywords: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Add an author
    pub fn with_author(mut self, author: Author) -> Self {
        self.authors.push(author);
        self
    }

    /// Add multiple authors
    pub fn with_authors(mut self, authors: impl IntoIterator<Item = Author>) -> Self {
        self.authors.extend(authors);
        self
    }

    /// Set the DOI
    pub fn with_doi(mut self, doi: impl Into<String>) -> Self {
        self.doi = Some(doi.into());
        self
    }

    /// Set the version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add keywords
    pub fn with_keywords(mut self, keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.keywords.extend(keywords.into_iter().map(Into::into));
        self
    }

    /// Get the first author (for citation keys)
    pub fn first_author(&self) -> Option<&Author> {
        self.authors.first()
    }
}

/// Validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid ORCID format: {0}")]
    InvalidOrcid(String),
    #[error("Invalid ROR ID format: {0}")]
    InvalidRorId(String),
}

/// Validate ORCID format: 0000-0000-0000-000X
pub fn validate_orcid(orcid: &str) -> bool {
    ORCID_REGEX.is_match(orcid)
}

/// Validate ROR ID format: <https://ror.org/xxxxxxxxx>
pub fn validate_ror_id(ror_id: &str) -> bool {
    ROR_REGEX.is_match(ror_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orcid_validation_valid() {
        // Standard ORCID
        assert!(validate_orcid("0000-0002-1825-0097"));
        // ORCID with X checksum
        assert!(validate_orcid("0000-0001-5109-351X"));
        // All zeros
        assert!(validate_orcid("0000-0000-0000-0000"));
        // All nines
        assert!(validate_orcid("9999-9999-9999-9999"));
    }

    #[test]
    fn test_orcid_validation_invalid() {
        // Missing hyphen
        assert!(!validate_orcid("000000021825-0097"));
        // Too short
        assert!(!validate_orcid("0000-0002-1825"));
        // Invalid character
        assert!(!validate_orcid("0000-0002-1825-009A"));
        // Lowercase x
        assert!(!validate_orcid("0000-0001-5109-351x"));
        // Empty
        assert!(!validate_orcid(""));
        // URL format (should just be the ID)
        assert!(!validate_orcid("https://orcid.org/0000-0002-1825-0097"));
    }

    #[test]
    fn test_ror_id_validation() {
        // Valid ROR IDs
        assert!(validate_ror_id("https://ror.org/03yrm5c26"));
        assert!(validate_ror_id("https://ror.org/00hj8s172"));

        // Invalid ROR IDs
        assert!(!validate_ror_id("ror.org/03yrm5c26")); // Missing https://
        assert!(!validate_ror_id("https://ror.org/03yrm5c2")); // Too short
        assert!(!validate_ror_id("https://ror.org/03yrm5c267")); // Too long
        assert!(!validate_ror_id("https://ror.org/03YRM5C26")); // Uppercase
        assert!(!validate_ror_id("")); // Empty
    }

    #[test]
    fn test_contributor_role_display() {
        assert_eq!(
            format!("{}", ContributorRole::Conceptualization),
            "Conceptualization"
        );
        assert_eq!(
            format!("{}", ContributorRole::DataCuration),
            "Data curation"
        );
        assert_eq!(
            format!("{}", ContributorRole::FormalAnalysis),
            "Formal analysis"
        );
        assert_eq!(
            format!("{}", ContributorRole::WritingOriginal),
            "Writing – original draft"
        );
        assert_eq!(
            format!("{}", ContributorRole::WritingReview),
            "Writing – review & editing"
        );
    }

    #[test]
    fn test_artifact_with_multiple_authors() {
        let author1 = Author::new("Alice Smith")
            .with_orcid("0000-0002-1825-0097")
            .unwrap()
            .with_role(ContributorRole::Conceptualization)
            .with_role(ContributorRole::WritingOriginal);

        let author2 = Author::new("Bob Jones")
            .with_role(ContributorRole::Software)
            .with_role(ContributorRole::DataCuration);

        let artifact = ResearchArtifact::new(
            "artifact-001",
            "Novel Deep Learning Architecture",
            ArtifactType::Paper,
            License::CcBy4,
        )
        .with_authors([author1, author2])
        .with_doi("10.1000/xyz123")
        .with_keywords(["deep learning", "neural networks"]);

        assert_eq!(artifact.authors.len(), 2);
        assert_eq!(artifact.authors[0].name, "Alice Smith");
        assert_eq!(
            artifact.authors[0].orcid,
            Some("0000-0002-1825-0097".to_string())
        );
        assert_eq!(artifact.authors[0].roles.len(), 2);
        assert_eq!(artifact.authors[1].name, "Bob Jones");
        assert_eq!(artifact.authors[1].orcid, None);
        assert_eq!(artifact.doi, Some("10.1000/xyz123".to_string()));
        assert_eq!(artifact.keywords.len(), 2);
    }

    #[test]
    fn test_author_last_name() {
        let author1 = Author::new("Alice Marie Smith");
        assert_eq!(author1.last_name(), "Smith");

        let author2 = Author::new("Madonna");
        assert_eq!(author2.last_name(), "Madonna");

        let author3 = Author::new("Ludwig van Beethoven");
        assert_eq!(author3.last_name(), "Beethoven");
    }

    #[test]
    fn test_affiliation_with_ror() {
        let affiliation = Affiliation::new("MIT")
            .with_ror_id("https://ror.org/03yrm5c26")
            .unwrap()
            .with_country("US");

        assert_eq!(affiliation.name, "MIT");
        assert_eq!(
            affiliation.ror_id,
            Some("https://ror.org/03yrm5c26".to_string())
        );
        assert_eq!(affiliation.country, Some("US".to_string()));
    }

    #[test]
    fn test_affiliation_invalid_ror() {
        let result = Affiliation::new("MIT").with_ror_id("invalid-ror");
        assert!(result.is_err());
    }

    #[test]
    fn test_author_invalid_orcid() {
        let result = Author::new("Alice Smith").with_orcid("invalid-orcid");
        assert!(result.is_err());
    }

    #[test]
    fn test_artifact_type_display() {
        assert_eq!(format!("{}", ArtifactType::Dataset), "Dataset");
        assert_eq!(format!("{}", ArtifactType::Model), "Model");
        assert_eq!(format!("{}", ArtifactType::Code), "Code");
        assert_eq!(format!("{}", ArtifactType::Paper), "Paper");
        assert_eq!(format!("{}", ArtifactType::Notebook), "Notebook");
        assert_eq!(format!("{}", ArtifactType::Workflow), "Workflow");
    }

    #[test]
    fn test_license_display() {
        assert_eq!(format!("{}", License::Mit), "MIT");
        assert_eq!(format!("{}", License::Apache2), "Apache-2.0");
        assert_eq!(format!("{}", License::CcBy4), "CC-BY-4.0");
        assert_eq!(
            format!("{}", License::Custom("LGPL-2.1".to_string())),
            "LGPL-2.1"
        );
    }

    #[test]
    fn test_author_with_affiliation() {
        let affiliation = Affiliation::new("Stanford University").with_country("US");

        let author = Author::new("Jane Doe").with_affiliation(affiliation);

        assert_eq!(author.affiliations.len(), 1);
        assert_eq!(author.affiliations[0].name, "Stanford University");
    }

    #[test]
    fn test_role_deduplication() {
        let author = Author::new("Test Author")
            .with_role(ContributorRole::Software)
            .with_role(ContributorRole::Software)
            .with_roles([ContributorRole::Software, ContributorRole::Validation]);

        assert_eq!(author.roles.len(), 2);
        assert!(author.roles.contains(&ContributorRole::Software));
        assert!(author.roles.contains(&ContributorRole::Validation));
    }
}

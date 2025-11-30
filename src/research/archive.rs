//! Archive Deposit for Zenodo/figshare (ENT-027)
//!
//! Provides deposit functionality for academic archives like
//! Zenodo, Figshare, Dryad, and Dataverse.

use crate::research::artifact::ResearchArtifact;
use crate::research::citation::CitationMetadata;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// DOI validation pattern
static DOI_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^10\.\d{4,}/[^\s]+$").expect("Invalid DOI regex"));

/// Archive provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchiveProvider {
    /// Zenodo (CERN)
    Zenodo,
    /// Figshare
    Figshare,
    /// Dryad
    Dryad,
    /// Dataverse
    Dataverse,
}

impl std::fmt::Display for ArchiveProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zenodo => write!(f, "Zenodo"),
            Self::Figshare => write!(f, "Figshare"),
            Self::Dryad => write!(f, "Dryad"),
            Self::Dataverse => write!(f, "Dataverse"),
        }
    }
}

impl ArchiveProvider {
    /// Get the base URL for the provider
    pub fn base_url(&self) -> &'static str {
        match self {
            Self::Zenodo => "https://zenodo.org",
            Self::Figshare => "https://figshare.com",
            Self::Dryad => "https://datadryad.org",
            Self::Dataverse => "https://dataverse.harvard.edu",
        }
    }

    /// Get the sandbox URL (if available)
    pub fn sandbox_url(&self) -> Option<&'static str> {
        match self {
            Self::Zenodo => Some("https://sandbox.zenodo.org"),
            Self::Figshare => None,
            Self::Dryad => None,
            Self::Dataverse => None,
        }
    }

    /// Get the API endpoint
    pub fn api_endpoint(&self) -> &'static str {
        match self {
            Self::Zenodo => "https://zenodo.org/api/deposit/depositions",
            Self::Figshare => "https://api.figshare.com/v2/account/articles",
            Self::Dryad => "https://datadryad.org/api/v2/datasets",
            Self::Dataverse => "https://dataverse.harvard.edu/api/dataverses",
        }
    }
}

/// Zenodo-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenodoConfig {
    /// API access token
    pub token: String,
    /// Use sandbox environment
    pub sandbox: bool,
    /// Community to submit to (optional)
    pub community: Option<String>,
}

impl ZenodoConfig {
    /// Create a new Zenodo configuration
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            sandbox: false,
            community: None,
        }
    }

    /// Use sandbox environment
    pub fn with_sandbox(mut self, sandbox: bool) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Set community
    pub fn with_community(mut self, community: impl Into<String>) -> Self {
        self.community = Some(community.into());
        self
    }

    /// Get the appropriate base URL
    pub fn base_url(&self) -> &'static str {
        if self.sandbox {
            "https://sandbox.zenodo.org"
        } else {
            "https://zenodo.org"
        }
    }
}

/// Figshare-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigshareConfig {
    /// API access token
    pub token: String,
    /// Project ID (optional)
    pub project_id: Option<u64>,
}

impl FigshareConfig {
    /// Create a new Figshare configuration
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            project_id: None,
        }
    }

    /// Set project ID
    pub fn with_project(mut self, project_id: u64) -> Self {
        self.project_id = Some(project_id);
        self
    }
}

/// Deposit metadata for archive submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositMetadata {
    /// Title
    pub title: String,
    /// Description/abstract
    pub description: String,
    /// Authors (names)
    pub authors: Vec<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// License identifier (e.g., "cc-by-4.0")
    pub license: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Related identifiers (DOIs, URLs)
    pub related_identifiers: Vec<RelatedIdentifier>,
}

impl DepositMetadata {
    /// Create from a research artifact
    pub fn from_artifact(artifact: &ResearchArtifact) -> Self {
        Self {
            title: artifact.title.clone(),
            description: artifact
                .description
                .clone()
                .unwrap_or_else(|| format!("{} - {}", artifact.title, artifact.artifact_type)),
            authors: artifact.authors.iter().map(|a| a.name.clone()).collect(),
            keywords: artifact.keywords.clone(),
            license: artifact.license.to_string().to_lowercase(),
            resource_type: ResourceType::from_artifact_type(artifact.artifact_type),
            related_identifiers: artifact
                .doi
                .iter()
                .map(RelatedIdentifier::is_identical_to)
                .collect(),
        }
    }

    /// Create from citation metadata
    pub fn from_citation(citation: &CitationMetadata) -> Self {
        let mut metadata = Self::from_artifact(&citation.artifact);
        if let Some(url) = &citation.url {
            metadata
                .related_identifiers
                .push(RelatedIdentifier::is_supplement_to(url));
        }
        metadata
    }
}

/// Resource type for archives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Dataset,
    Software,
    Publication,
    Presentation,
    Poster,
    Image,
    Video,
    Other,
}

impl ResourceType {
    /// Convert from artifact type
    pub fn from_artifact_type(artifact_type: crate::research::artifact::ArtifactType) -> Self {
        use crate::research::artifact::ArtifactType;
        match artifact_type {
            ArtifactType::Dataset => Self::Dataset,
            ArtifactType::Model | ArtifactType::Code => Self::Software,
            ArtifactType::Paper => Self::Publication,
            ArtifactType::Notebook | ArtifactType::Workflow => Self::Other,
        }
    }

    /// Get Zenodo upload type string
    pub fn zenodo_type(&self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Software => "software",
            Self::Publication => "publication",
            Self::Presentation => "presentation",
            Self::Poster => "poster",
            Self::Image => "image",
            Self::Video => "video",
            Self::Other => "other",
        }
    }
}

/// Related identifier for linking resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedIdentifier {
    /// The identifier value (DOI, URL, etc.)
    pub identifier: String,
    /// Relation type
    pub relation: RelationType,
    /// Identifier scheme (DOI, URL, etc.)
    pub scheme: IdentifierScheme,
}

impl RelatedIdentifier {
    /// Create a "is identical to" relation
    pub fn is_identical_to(identifier: impl Into<String>) -> Self {
        let id = identifier.into();
        let scheme = if id.starts_with("10.") {
            IdentifierScheme::Doi
        } else if id.starts_with("http") {
            IdentifierScheme::Url
        } else {
            IdentifierScheme::Other
        };

        Self {
            identifier: id,
            relation: RelationType::IsIdenticalTo,
            scheme,
        }
    }

    /// Create a "is supplement to" relation
    pub fn is_supplement_to(identifier: impl Into<String>) -> Self {
        let id = identifier.into();
        let scheme = if id.starts_with("10.") {
            IdentifierScheme::Doi
        } else {
            IdentifierScheme::Url
        };

        Self {
            identifier: id,
            relation: RelationType::IsSupplementTo,
            scheme,
        }
    }

    /// Create a "cites" relation
    pub fn cites(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            relation: RelationType::Cites,
            scheme: IdentifierScheme::Doi,
        }
    }
}

/// Relation types (based on DataCite)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    IsCitedBy,
    Cites,
    IsSupplementTo,
    IsSupplementedBy,
    IsContinuedBy,
    Continues,
    IsDescribedBy,
    Describes,
    HasMetadata,
    IsMetadataFor,
    HasVersion,
    IsVersionOf,
    IsNewVersionOf,
    IsPreviousVersionOf,
    IsPartOf,
    HasPart,
    IsReferencedBy,
    References,
    IsDocumentedBy,
    Documents,
    IsCompiledBy,
    Compiles,
    IsVariantFormOf,
    IsOriginalFormOf,
    IsIdenticalTo,
    IsReviewedBy,
    Reviews,
    IsDerivedFrom,
    IsSourceOf,
    IsRequiredBy,
    Requires,
    IsObsoletedBy,
    Obsoletes,
}

/// Identifier scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentifierScheme {
    Doi,
    Url,
    Orcid,
    Ror,
    Arxiv,
    Pmid,
    Other,
}

/// Archive deposit request
#[derive(Debug, Clone)]
pub struct ArchiveDeposit {
    /// Archive provider
    pub provider: ArchiveProvider,
    /// Artifact to deposit
    pub artifact: ResearchArtifact,
    /// Deposit metadata
    pub metadata: DepositMetadata,
    /// Files to upload (path -> content)
    pub files: Vec<(String, Vec<u8>)>,
}

impl ArchiveDeposit {
    /// Create a new deposit
    pub fn new(provider: ArchiveProvider, artifact: ResearchArtifact) -> Self {
        let metadata = DepositMetadata::from_artifact(&artifact);
        Self {
            provider,
            artifact,
            metadata,
            files: Vec::new(),
        }
    }

    /// Add a file to upload
    pub fn with_file(mut self, filename: impl Into<String>, content: Vec<u8>) -> Self {
        self.files.push((filename.into(), content));
        self
    }

    /// Add a text file
    pub fn with_text_file(self, filename: impl Into<String>, content: impl Into<String>) -> Self {
        self.with_file(filename, content.into().into_bytes())
    }

    /// Perform deposit (mock implementation for testing)
    pub fn deposit(&self) -> Result<DepositResult, DepositError> {
        // This is a mock implementation - real implementation would use HTTP client
        // to interact with the archive's API

        // Validate we have at least one file
        if self.files.is_empty() {
            return Err(DepositError::NoFiles);
        }

        // Generate mock DOI and record ID
        let record_id = format!("{}", rand::random::<u64>() % 10_000_000);
        let doi = format!("10.5281/zenodo.{record_id}");

        let url = format!("{}/record/{}", self.provider.base_url(), record_id);

        Ok(DepositResult {
            doi,
            record_id,
            url,
            provider: self.provider,
        })
    }
}

/// Result of a successful deposit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositResult {
    /// Assigned DOI
    pub doi: String,
    /// Provider-specific record ID
    pub record_id: String,
    /// URL to the deposited record
    pub url: String,
    /// Provider that received the deposit
    pub provider: ArchiveProvider,
}

impl DepositResult {
    /// Generate URL for the deposit
    pub fn generate_url(&self) -> String {
        self.url.clone()
    }

    /// Generate DOI URL
    pub fn doi_url(&self) -> String {
        format!("https://doi.org/{}", self.doi)
    }
}

/// Deposit errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum DepositError {
    #[error("No files provided for deposit")]
    NoFiles,
    #[error("Authentication failed")]
    AuthenticationFailed,
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
    #[error("Upload failed: {0}")]
    UploadFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
}

/// Validate DOI format
pub fn validate_doi(doi: &str) -> bool {
    DOI_REGEX.is_match(doi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::artifact::{ArtifactType, Author, License};

    fn create_test_artifact() -> ResearchArtifact {
        ResearchArtifact::new(
            "dataset-001",
            "Test Dataset for Machine Learning",
            ArtifactType::Dataset,
            License::CcBy4,
        )
        .with_author(Author::new("Alice Smith"))
        .with_doi("10.1234/test")
        .with_description("A test dataset for ML research")
        .with_keywords(["machine learning", "dataset"])
    }

    #[test]
    fn test_zenodo_deposit_struct() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_text_file("data.csv", "a,b,c\n1,2,3");

        assert_eq!(deposit.provider, ArchiveProvider::Zenodo);
        assert_eq!(deposit.files.len(), 1);
        assert_eq!(deposit.metadata.title, "Test Dataset for Machine Learning");
    }

    #[test]
    fn test_figshare_deposit_struct() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Figshare, artifact);

        assert_eq!(deposit.provider, ArchiveProvider::Figshare);
    }

    #[test]
    fn test_doi_format_validation() {
        // Valid DOIs
        assert!(validate_doi("10.1234/test"));
        assert!(validate_doi("10.5281/zenodo.1234567"));
        assert!(validate_doi("10.1000/xyz123"));
        assert!(validate_doi("10.12345/some-complex.id_here"));

        // Invalid DOIs
        assert!(!validate_doi("1234/test")); // Missing 10.
        assert!(!validate_doi("10.123/test")); // Too few digits after 10.
        assert!(!validate_doi("")); // Empty
        assert!(!validate_doi("doi:10.1234/test")); // Has prefix
    }

    #[test]
    fn test_deposit_result_url_generation() {
        let result = DepositResult {
            doi: "10.5281/zenodo.1234567".to_string(),
            record_id: "1234567".to_string(),
            url: "https://zenodo.org/record/1234567".to_string(),
            provider: ArchiveProvider::Zenodo,
        };

        assert_eq!(result.generate_url(), "https://zenodo.org/record/1234567");
        assert_eq!(result.doi_url(), "https://doi.org/10.5281/zenodo.1234567");
    }

    #[test]
    fn test_zenodo_config() {
        let config = ZenodoConfig::new("test-token")
            .with_sandbox(true)
            .with_community("ml-research");

        assert_eq!(config.token, "test-token");
        assert!(config.sandbox);
        assert_eq!(config.community, Some("ml-research".to_string()));
        assert_eq!(config.base_url(), "https://sandbox.zenodo.org");
    }

    #[test]
    fn test_figshare_config() {
        let config = FigshareConfig::new("test-token").with_project(12345);

        assert_eq!(config.token, "test-token");
        assert_eq!(config.project_id, Some(12345));
    }

    #[test]
    fn test_deposit_metadata_from_artifact() {
        let artifact = create_test_artifact();
        let metadata = DepositMetadata::from_artifact(&artifact);

        assert_eq!(metadata.title, "Test Dataset for Machine Learning");
        assert_eq!(metadata.authors, vec!["Alice Smith"]);
        assert_eq!(metadata.keywords, vec!["machine learning", "dataset"]);
        assert_eq!(metadata.resource_type, ResourceType::Dataset);
    }

    #[test]
    fn test_resource_type_conversion() {
        assert_eq!(
            ResourceType::from_artifact_type(ArtifactType::Dataset),
            ResourceType::Dataset
        );
        assert_eq!(
            ResourceType::from_artifact_type(ArtifactType::Code),
            ResourceType::Software
        );
        assert_eq!(
            ResourceType::from_artifact_type(ArtifactType::Paper),
            ResourceType::Publication
        );
    }

    #[test]
    fn test_resource_type_zenodo() {
        assert_eq!(ResourceType::Dataset.zenodo_type(), "dataset");
        assert_eq!(ResourceType::Software.zenodo_type(), "software");
        assert_eq!(ResourceType::Publication.zenodo_type(), "publication");
    }

    #[test]
    fn test_related_identifier_creation() {
        let doi_rel = RelatedIdentifier::is_identical_to("10.1234/test");
        assert_eq!(doi_rel.scheme, IdentifierScheme::Doi);
        assert_eq!(doi_rel.relation, RelationType::IsIdenticalTo);

        let url_rel = RelatedIdentifier::is_supplement_to("https://example.com/data");
        assert_eq!(url_rel.scheme, IdentifierScheme::Url);
        assert_eq!(url_rel.relation, RelationType::IsSupplementTo);
    }

    #[test]
    fn test_provider_urls() {
        assert_eq!(ArchiveProvider::Zenodo.base_url(), "https://zenodo.org");
        assert_eq!(ArchiveProvider::Figshare.base_url(), "https://figshare.com");

        assert!(ArchiveProvider::Zenodo.sandbox_url().is_some());
        assert!(ArchiveProvider::Figshare.sandbox_url().is_none());
    }

    #[test]
    fn test_deposit_no_files_error() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact);
        // No files added

        let result = deposit.deposit();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DepositError::NoFiles));
    }

    #[test]
    fn test_deposit_mock_success() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_text_file("README.md", "# Test");

        let result = deposit.deposit();
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.doi.starts_with("10.5281/zenodo."));
        assert!(result.url.starts_with("https://zenodo.org/record/"));
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(format!("{}", ArchiveProvider::Zenodo), "Zenodo");
        assert_eq!(format!("{}", ArchiveProvider::Figshare), "Figshare");
        assert_eq!(format!("{}", ArchiveProvider::Dryad), "Dryad");
    }
}

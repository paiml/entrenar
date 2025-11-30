//! RO-Crate bundling (ENT-026)
//!
//! Provides Research Object Crate (RO-Crate) packaging for
//! FAIR-compliant research data packaging.

use crate::research::artifact::ResearchArtifact;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// JSON-LD context for RO-Crate 1.1
pub const RO_CRATE_CONTEXT: &str = "https://w3id.org/ro/crate/1.1/context";

/// RO-Crate entity types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// Root dataset
    Dataset,
    /// File entity
    File,
    /// Person
    Person,
    /// Organization
    Organization,
    /// Software application
    SoftwareApplication,
    /// Creative work (paper, etc.)
    CreativeWork,
    /// Action (workflow execution, etc.)
    CreateAction,
    /// Custom type
    Custom(String),
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "Dataset"),
            Self::File => write!(f, "File"),
            Self::Person => write!(f, "Person"),
            Self::Organization => write!(f, "Organization"),
            Self::SoftwareApplication => write!(f, "SoftwareApplication"),
            Self::CreativeWork => write!(f, "CreativeWork"),
            Self::CreateAction => write!(f, "CreateAction"),
            Self::Custom(t) => write!(f, "{t}"),
        }
    }
}

/// An entity in the RO-Crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoCrateEntity {
    /// Entity ID (path or URL)
    #[serde(rename = "@id")]
    pub id: String,
    /// Entity type
    #[serde(rename = "@type")]
    pub type_field: String,
    /// Additional properties
    #[serde(flatten)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl RoCrateEntity {
    /// Create a new entity
    pub fn new(id: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            id: id.into(),
            type_field: entity_type.to_string(),
            properties: HashMap::new(),
        }
    }

    /// Add a property
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add a name property
    pub fn with_name(self, name: impl Into<String>) -> Self {
        self.with_property("name", name.into())
    }

    /// Add a description
    pub fn with_description(self, description: impl Into<String>) -> Self {
        self.with_property("description", description.into())
    }

    /// Add a reference to another entity
    pub fn with_reference(self, key: impl Into<String>, ref_id: impl Into<String>) -> Self {
        self.with_property(key, json!({ "@id": ref_id.into() }))
    }

    /// Add multiple references
    pub fn with_references(
        self,
        key: impl Into<String>,
        ref_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let refs: Vec<serde_json::Value> = ref_ids
            .into_iter()
            .map(|id| json!({ "@id": id.into() }))
            .collect();
        self.with_property(key, refs)
    }

    /// Create the root dataset entity
    pub fn root_dataset() -> Self {
        Self::new("./", EntityType::Dataset)
    }

    /// Create a file entity
    pub fn file(path: impl Into<String>) -> Self {
        Self::new(path, EntityType::File)
    }

    /// Create a person entity
    pub fn person(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, EntityType::Person).with_name(name)
    }
}

/// RO-Crate metadata descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoCrateDescriptor {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: String,
    /// Graph of entities
    #[serde(rename = "@graph")]
    pub graph: Vec<RoCrateEntity>,
}

impl Default for RoCrateDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl RoCrateDescriptor {
    /// Create a new RO-Crate descriptor
    pub fn new() -> Self {
        // Start with the metadata file entity pointing to root
        let metadata_entity =
            RoCrateEntity::new("ro-crate-metadata.json", EntityType::CreativeWork)
                .with_property(
                    "conformsTo",
                    json!({ "@id": "https://w3id.org/ro/crate/1.1" }),
                )
                .with_reference("about", "./");

        Self {
            context: RO_CRATE_CONTEXT.to_string(),
            graph: vec![metadata_entity],
        }
    }

    /// Add an entity to the graph
    pub fn add_entity(&mut self, entity: RoCrateEntity) {
        self.graph.push(entity);
    }

    /// Get the root dataset entity
    pub fn root_dataset(&self) -> Option<&RoCrateEntity> {
        self.graph.iter().find(|e| e.id == "./")
    }

    /// Get the root dataset entity mutably
    pub fn root_dataset_mut(&mut self) -> Option<&mut RoCrateEntity> {
        self.graph.iter_mut().find(|e| e.id == "./")
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// RO-Crate package
#[derive(Debug, Clone)]
pub struct RoCrate {
    /// Root directory path
    pub root: PathBuf,
    /// Metadata descriptor
    pub descriptor: RoCrateDescriptor,
    /// Data files to include (relative path -> content)
    pub data_files: HashMap<String, Vec<u8>>,
}

impl RoCrate {
    /// Create a new RO-Crate
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let mut descriptor = RoCrateDescriptor::new();

        // Add root dataset
        let root_entity = RoCrateEntity::root_dataset().with_property(
            "datePublished",
            chrono::Utc::now().format("%Y-%m-%d").to_string(),
        );
        descriptor.add_entity(root_entity);

        Self {
            root: root.into(),
            descriptor,
            data_files: HashMap::new(),
        }
    }

    /// Create from a research artifact
    pub fn from_artifact(artifact: &ResearchArtifact, root: impl Into<PathBuf>) -> Self {
        let mut crate_pkg = Self::new(root);

        // Update root dataset with artifact metadata
        if let Some(root_entity) = crate_pkg.descriptor.root_dataset_mut() {
            root_entity
                .properties
                .insert("name".to_string(), json!(artifact.title));
            if let Some(desc) = &artifact.description {
                root_entity
                    .properties
                    .insert("description".to_string(), json!(desc));
            }
            root_entity
                .properties
                .insert("version".to_string(), json!(artifact.version));
            root_entity
                .properties
                .insert("license".to_string(), json!(artifact.license.to_string()));

            if let Some(doi) = &artifact.doi {
                root_entity
                    .properties
                    .insert("identifier".to_string(), json!(doi));
            }

            if !artifact.keywords.is_empty() {
                root_entity
                    .properties
                    .insert("keywords".to_string(), json!(artifact.keywords.join(", ")));
            }
        }

        // Add author entities
        let mut author_ids = Vec::new();
        for (i, author) in artifact.authors.iter().enumerate() {
            let author_id = format!("#author-{}", i + 1);
            author_ids.push(author_id.clone());

            let mut person_entity = RoCrateEntity::person(&author_id, &author.name);

            if let Some(orcid) = &author.orcid {
                person_entity =
                    person_entity.with_property("identifier", format!("https://orcid.org/{orcid}"));
            }

            if let Some(affiliation) = author.affiliations.first() {
                let org_id = format!("#org-{}", i + 1);
                let org_entity = RoCrateEntity::new(&org_id, EntityType::Organization)
                    .with_name(&affiliation.name);
                crate_pkg.descriptor.add_entity(org_entity);
                person_entity = person_entity.with_reference("affiliation", &org_id);
            }

            crate_pkg.descriptor.add_entity(person_entity);
        }

        // Link authors to root dataset
        if !author_ids.is_empty() {
            if let Some(root_entity) = crate_pkg.descriptor.root_dataset_mut() {
                let author_refs: Vec<serde_json::Value> =
                    author_ids.iter().map(|id| json!({ "@id": id })).collect();
                root_entity
                    .properties
                    .insert("author".to_string(), json!(author_refs));
            }
        }

        crate_pkg
    }

    /// Add a data file
    pub fn add_file(&mut self, path: impl Into<String>, content: Vec<u8>) {
        let path_str = path.into();

        // Add file entity to descriptor
        let file_entity = RoCrateEntity::file(&path_str)
            .with_property("contentSize", content.len().to_string())
            .with_property("encodingFormat", guess_mime_type(&path_str));

        self.descriptor.add_entity(file_entity);
        self.data_files.insert(path_str, content);
    }

    /// Add a text file
    pub fn add_text_file(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.add_file(path, content.into().into_bytes());
    }

    /// Write to a directory
    pub fn to_directory(&self) -> std::io::Result<PathBuf> {
        // Create root directory
        std::fs::create_dir_all(&self.root)?;

        // Write ro-crate-metadata.json
        let metadata_path = self.root.join("ro-crate-metadata.json");
        std::fs::write(&metadata_path, self.descriptor.to_json())?;

        // Write data files
        for (path, content) in &self.data_files {
            let file_path = self.root.join(path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, content)?;
        }

        Ok(self.root.clone())
    }

    /// Create a ZIP archive
    pub fn to_zip(&self) -> std::io::Result<Vec<u8>> {
        let mut buffer = std::io::Cursor::new(Vec::new());

        {
            let mut zip = zip::ZipWriter::new(&mut buffer);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            // Write ro-crate-metadata.json
            zip.start_file("ro-crate-metadata.json", options)?;
            zip.write_all(self.descriptor.to_json().as_bytes())?;

            // Write data files
            for (path, content) in &self.data_files {
                zip.start_file(path, options)?;
                zip.write_all(content)?;
            }

            zip.finish()?;
        }

        Ok(buffer.into_inner())
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.descriptor.graph.len()
    }

    /// Get file count
    pub fn file_count(&self) -> usize {
        self.data_files.len()
    }
}

/// Guess MIME type from file extension
fn guess_mime_type(path: &str) -> &'static str {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext.to_lowercase().as_str() {
        "json" => "application/json",
        "yaml" | "yml" => "application/x-yaml",
        "csv" => "text/csv",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "py" => "text/x-python",
        "rs" => "text/x-rust",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "parquet" => "application/vnd.apache.parquet",
        "safetensors" => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::artifact::{Affiliation, ArtifactType, Author, License};
    use tempfile::TempDir;

    fn create_test_artifact() -> ResearchArtifact {
        let author = Author::new("Alice Smith")
            .with_orcid("0000-0002-1825-0097")
            .unwrap()
            .with_affiliation(Affiliation::new("MIT"));

        ResearchArtifact::new(
            "dataset-001",
            "Test Dataset",
            ArtifactType::Dataset,
            License::CcBy4,
        )
        .with_author(author)
        .with_doi("10.1234/test")
        .with_description("A test dataset")
        .with_keywords(["test", "dataset"])
    }

    #[test]
    fn test_ro_crate_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().join("test-crate");

        let crate_pkg = RoCrate::new(&crate_path);
        let result = crate_pkg.to_directory();

        assert!(result.is_ok());
        assert!(crate_path.exists());
        assert!(crate_path.join("ro-crate-metadata.json").exists());
    }

    #[test]
    fn test_ro_crate_metadata_json() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().join("test-crate");

        let crate_pkg = RoCrate::new(&crate_path);
        crate_pkg.to_directory().unwrap();

        let metadata_content =
            std::fs::read_to_string(crate_path.join("ro-crate-metadata.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&metadata_content).unwrap();

        assert_eq!(parsed["@context"], RO_CRATE_CONTEXT);
        assert!(parsed["@graph"].is_array());
    }

    #[test]
    fn test_ro_crate_zip_creation() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().join("test-crate");

        let mut crate_pkg = RoCrate::new(&crate_path);
        crate_pkg.add_text_file("data.csv", "a,b,c\n1,2,3");

        let zip_data = crate_pkg.to_zip().unwrap();

        assert!(!zip_data.is_empty());

        // Verify ZIP structure
        let reader = std::io::Cursor::new(zip_data);
        let mut archive = zip::ZipArchive::new(reader).unwrap();

        assert!(archive.by_name("ro-crate-metadata.json").is_ok());
        assert!(archive.by_name("data.csv").is_ok());
    }

    #[test]
    fn test_ro_crate_entities_linked() {
        let artifact = create_test_artifact();
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().join("test-crate");

        let crate_pkg = RoCrate::from_artifact(&artifact, &crate_path);

        // Should have: metadata descriptor, root dataset, author, organization
        assert!(crate_pkg.entity_count() >= 4);

        // Check root dataset has author reference
        let root = crate_pkg.descriptor.root_dataset().unwrap();
        assert!(root.properties.contains_key("author"));
        assert!(root.properties.contains_key("name"));
    }

    #[test]
    fn test_ro_crate_includes_data_files() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().join("test-crate");

        let mut crate_pkg = RoCrate::new(&crate_path);
        crate_pkg.add_text_file("data/train.csv", "x,y\n1,2");
        crate_pkg.add_text_file("data/test.csv", "x,y\n3,4");
        crate_pkg.add_file("model.safetensors", vec![0u8; 100]);

        crate_pkg.to_directory().unwrap();

        assert!(crate_path.join("data/train.csv").exists());
        assert!(crate_path.join("data/test.csv").exists());
        assert!(crate_path.join("model.safetensors").exists());

        // Check file entities in descriptor
        assert_eq!(crate_pkg.file_count(), 3);
    }

    #[test]
    fn test_entity_creation() {
        let entity = RoCrateEntity::new("#test", EntityType::Dataset)
            .with_name("Test Dataset")
            .with_description("A test")
            .with_reference("author", "#person-1");

        assert_eq!(entity.id, "#test");
        assert_eq!(entity.type_field, "Dataset");
        assert!(entity.properties.contains_key("name"));
        assert!(entity.properties.contains_key("description"));
        assert!(entity.properties.contains_key("author"));
    }

    #[test]
    fn test_person_entity() {
        let person = RoCrateEntity::person("#alice", "Alice Smith");

        assert_eq!(person.id, "#alice");
        assert_eq!(person.type_field, "Person");
        assert_eq!(person.properties.get("name"), Some(&json!("Alice Smith")));
    }

    #[test]
    fn test_mime_type_guessing() {
        assert_eq!(guess_mime_type("data.json"), "application/json");
        assert_eq!(guess_mime_type("data.csv"), "text/csv");
        assert_eq!(guess_mime_type("script.py"), "text/x-python");
        assert_eq!(guess_mime_type("lib.rs"), "text/x-rust");
        assert_eq!(guess_mime_type("unknown.xyz"), "application/octet-stream");
    }

    #[test]
    fn test_descriptor_serialization() {
        let descriptor = RoCrateDescriptor::new();
        let json = descriptor.to_json();

        assert!(json.contains("@context"));
        assert!(json.contains("@graph"));
        assert!(json.contains("ro-crate-metadata.json"));
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(format!("{}", EntityType::Dataset), "Dataset");
        assert_eq!(format!("{}", EntityType::Person), "Person");
        assert_eq!(
            format!("{}", EntityType::Custom("MyType".to_string())),
            "MyType"
        );
    }

    #[test]
    fn test_artifact_metadata_in_crate() {
        let artifact = create_test_artifact();
        let temp_dir = TempDir::new().unwrap();

        let crate_pkg = RoCrate::from_artifact(&artifact, temp_dir.path().join("crate"));

        let root = crate_pkg.descriptor.root_dataset().unwrap();

        assert_eq!(root.properties.get("name"), Some(&json!("Test Dataset")));
        assert_eq!(root.properties.get("version"), Some(&json!("1.0.0")));
        assert_eq!(root.properties.get("license"), Some(&json!("CC-BY-4.0")));
        assert_eq!(
            root.properties.get("identifier"),
            Some(&json!("10.1234/test"))
        );
    }
}

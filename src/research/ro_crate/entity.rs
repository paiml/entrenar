//! RO-Crate entity types and structures.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

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

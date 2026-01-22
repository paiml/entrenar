//! Research init subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{ArtifactTypeArg, LicenseArg, ResearchInitArgs};
use crate::research::{Affiliation, ArtifactType, Author, License, ResearchArtifact};

pub fn run_research_init(args: ResearchInitArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Initializing research artifact: {}", args.id),
    );

    // Convert CLI types to research types
    let artifact_type = match args.artifact_type {
        ArtifactTypeArg::Dataset => ArtifactType::Dataset,
        ArtifactTypeArg::Paper => ArtifactType::Paper,
        ArtifactTypeArg::Model => ArtifactType::Model,
        ArtifactTypeArg::Code => ArtifactType::Code,
        ArtifactTypeArg::Notebook => ArtifactType::Notebook,
        ArtifactTypeArg::Workflow => ArtifactType::Workflow,
    };

    let license = match args.license {
        LicenseArg::CcBy4 => License::CcBy4,
        LicenseArg::CcBySa4 => License::Custom("CC-BY-SA-4.0".to_string()),
        LicenseArg::Cc0 => License::Cc0,
        LicenseArg::Mit => License::Mit,
        LicenseArg::Apache2 => License::Apache2,
        LicenseArg::Gpl3 => License::Gpl3,
        LicenseArg::Bsd3 => License::Bsd3,
    };

    let mut artifact = ResearchArtifact::new(&args.id, &args.title, artifact_type, license);

    // Add author if provided
    if let Some(author_name) = &args.author {
        let mut author = Author::new(author_name);

        if let Some(orcid) = &args.orcid {
            author = author
                .with_orcid(orcid)
                .map_err(|e| format!("Invalid ORCID: {e}"))?;
        }

        if let Some(affiliation) = &args.affiliation {
            author = author.with_affiliation(Affiliation::new(affiliation));
        }

        artifact = artifact.with_author(author);
    }

    // Add optional fields
    if let Some(description) = &args.description {
        artifact = artifact.with_description(description);
    }

    if let Some(keywords) = &args.keywords {
        let kw: Vec<&str> = keywords.split(',').map(str::trim).collect();
        artifact = artifact.with_keywords(kw);
    }

    if let Some(doi) = &args.doi {
        artifact = artifact.with_doi(doi);
    }

    // Serialize to YAML
    let yaml = serde_yaml::to_string(&artifact).map_err(|e| format!("Serialization error: {e}"))?;

    std::fs::write(&args.output, &yaml).map_err(|e| format!("Failed to write file: {e}"))?;

    log(
        level,
        LogLevel::Normal,
        &format!("Artifact saved to: {}", args.output.display()),
    );

    Ok(())
}

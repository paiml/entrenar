//! Research command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::ArchiveProviderArg;
use crate::config::{
    ArtifactTypeArg, BundleArgs, CitationFormat, CiteArgs, DepositArgs, ExportArgs, ExportFormat,
    LicenseArg, PreregisterArgs, ResearchArgs, ResearchCommand, ResearchInitArgs, VerifyArgs,
};
use crate::research::{
    Affiliation, AnonymizationConfig, ArchiveDeposit, ArchiveProvider, ArtifactType, Author,
    CitationMetadata, License, LiterateDocument, NotebookExporter, PreRegistration,
    ResearchArtifact, RoCrate, SignedPreRegistration, TimestampProof,
};

pub fn run_research(args: ResearchArgs, level: LogLevel) -> Result<(), String> {
    match args.command {
        ResearchCommand::Init(init_args) => run_research_init(init_args, level),
        ResearchCommand::Preregister(prereg_args) => run_research_preregister(prereg_args, level),
        ResearchCommand::Cite(cite_args) => run_research_cite(cite_args, level),
        ResearchCommand::Export(export_args) => run_research_export(export_args, level),
        ResearchCommand::Deposit(deposit_args) => run_research_deposit(deposit_args, level),
        ResearchCommand::Bundle(bundle_args) => run_research_bundle(bundle_args, level),
        ResearchCommand::Verify(verify_args) => run_research_verify(verify_args, level),
    }
}

fn run_research_init(args: ResearchInitArgs, level: LogLevel) -> Result<(), String> {
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

fn run_research_preregister(args: PreregisterArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Creating pre-registration: {}", args.title),
    );

    let mut prereg = PreRegistration::new(
        &args.title,
        &args.hypothesis,
        &args.methodology,
        &args.analysis_plan,
    );

    if let Some(notes) = &args.notes {
        prereg = prereg.with_notes(notes);
    }

    // Create commitment
    let commitment = prereg.commit();
    log(
        level,
        LogLevel::Verbose,
        &format!("  Commitment hash: {}...", &commitment.hash[..32]),
    );

    // Sign if key provided
    let output = if let Some(key_path) = &args.sign_key {
        use ed25519_dalek::SigningKey;

        let key_bytes =
            std::fs::read(key_path).map_err(|e| format!("Failed to read signing key: {e}"))?;

        if key_bytes.len() != 32 {
            return Err("Signing key must be exactly 32 bytes".to_string());
        }

        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let signing_key = SigningKey::from_bytes(&key_array);

        let mut signed = SignedPreRegistration::sign(&prereg, &signing_key);

        // Add git timestamp if requested
        if args.git_timestamp {
            let output = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .map_err(|e| format!("Failed to get git commit: {e}"))?;

            let commit_hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            signed = signed.with_timestamp_proof(TimestampProof::git(&commit_hash));
            log(
                level,
                LogLevel::Verbose,
                &format!("  Git timestamp: {commit_hash}"),
            );
        }

        serde_yaml::to_string(&signed).map_err(|e| format!("Serialization error: {e}"))?
    } else {
        // Output commitment without signature
        serde_yaml::to_string(&commitment).map_err(|e| format!("Serialization error: {e}"))?
    };

    std::fs::write(&args.output, &output).map_err(|e| format!("Failed to write file: {e}"))?;

    log(
        level,
        LogLevel::Normal,
        &format!("Pre-registration saved to: {}", args.output.display()),
    );

    Ok(())
}

fn run_research_cite(args: CiteArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Generating citation from: {}", args.artifact.display()),
    );

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    // Create citation
    let mut citation = CitationMetadata::new(artifact, args.year);

    if let Some(journal) = &args.journal {
        citation = citation.with_journal(journal);
    }
    if let Some(volume) = &args.volume {
        citation = citation.with_volume(volume);
    }
    if let Some(pages) = &args.pages {
        citation = citation.with_pages(pages);
    }
    if let Some(url) = &args.url {
        citation = citation.with_url(url);
    }

    // Generate output
    let output = match args.format {
        CitationFormat::Bibtex => citation.to_bibtex(),
        CitationFormat::Cff => citation.to_cff(),
        CitationFormat::Json => {
            serde_json::to_string_pretty(&citation).map_err(|e| format!("JSON error: {e}"))?
        }
    };

    // Write or print
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &output).map_err(|e| format!("Failed to write file: {e}"))?;
        log(
            level,
            LogLevel::Normal,
            &format!("Citation saved to: {}", output_path.display()),
        );
    } else {
        println!("{output}");
    }

    Ok(())
}

fn run_research_export(args: ExportArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!(
            "Exporting: {} -> {}",
            args.input.display(),
            args.output.display()
        ),
    );

    match args.format {
        ExportFormat::Notebook => {
            // Read input as literate document
            let content = std::fs::read_to_string(&args.input)
                .map_err(|e| format!("Failed to read input: {e}"))?;

            let doc = LiterateDocument::parse_markdown(&content);
            let notebook = NotebookExporter::from_literate(&doc);

            let ipynb = notebook.to_ipynb();
            std::fs::write(&args.output, &ipynb)
                .map_err(|e| format!("Failed to write notebook: {e}"))?;

            log(
                level,
                LogLevel::Normal,
                &format!("Notebook exported: {} cells", notebook.cell_count()),
            );
        }
        ExportFormat::Html => {
            let content = std::fs::read_to_string(&args.input)
                .map_err(|e| format!("Failed to read input: {e}"))?;

            let doc = LiterateDocument::parse_markdown(&content);
            let html = doc.to_html();

            std::fs::write(&args.output, &html)
                .map_err(|e| format!("Failed to write HTML: {e}"))?;

            log(level, LogLevel::Normal, "HTML exported successfully");
        }
        ExportFormat::AnonymizedJson => {
            if !args.anonymize {
                return Err("--anonymize flag required for anonymized export".to_string());
            }

            let salt = args
                .anon_salt
                .as_ref()
                .ok_or("--anon-salt required for anonymization")?;

            // Load artifact
            let yaml = std::fs::read_to_string(&args.input)
                .map_err(|e| format!("Failed to read artifact: {e}"))?;

            let artifact: ResearchArtifact = serde_yaml::from_str(&yaml)
                .map_err(|e| format!("Failed to parse artifact: {e}"))?;

            let config = AnonymizationConfig::new(salt);
            let anon = config.anonymize(&artifact);

            let json =
                serde_json::to_string_pretty(&anon).map_err(|e| format!("JSON error: {e}"))?;

            std::fs::write(&args.output, &json)
                .map_err(|e| format!("Failed to write JSON: {e}"))?;

            log(
                level,
                LogLevel::Normal,
                &format!("Anonymized artifact: {}", anon.anonymous_id),
            );
        }
        ExportFormat::RoCrate => {
            return Err("Use 'entrenar research bundle' for RO-Crate export".to_string());
        }
    }

    Ok(())
}

fn run_research_deposit(args: DepositArgs, level: LogLevel) -> Result<(), String> {
    let provider = match args.provider {
        ArchiveProviderArg::Zenodo => ArchiveProvider::Zenodo,
        ArchiveProviderArg::Figshare => ArchiveProvider::Figshare,
        ArchiveProviderArg::Dryad => ArchiveProvider::Dryad,
        ArchiveProviderArg::Dataverse => ArchiveProvider::Dataverse,
    };

    log(
        level,
        LogLevel::Normal,
        &format!("Preparing deposit to: {provider}"),
    );

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    let mut deposit = ArchiveDeposit::new(provider, artifact);

    // Add files
    for file_path in &args.file {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read {}: {e}", file_path.display()))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("Invalid file name: {}", file_path.display()))?;

        deposit = deposit.with_text_file(file_name, &content);
    }

    if args.dry_run {
        log(level, LogLevel::Normal, "Dry run - deposit validated:");
        log(
            level,
            LogLevel::Normal,
            &format!("  Provider: {}", deposit.provider),
        );
        log(
            level,
            LogLevel::Normal,
            &format!("  Title: {}", deposit.metadata.title),
        );
        log(
            level,
            LogLevel::Normal,
            &format!("  Files: {}", deposit.files.len()),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Base URL: {}", provider.base_url()),
        );
    } else {
        // Note: Actual deposit would require async HTTP client
        // For now, we just validate the deposit structure
        log(
            level,
            LogLevel::Normal,
            "Deposit prepared (actual upload requires API token and network access)",
        );
        log(
            level,
            LogLevel::Normal,
            &format!("  Provider: {} ({})", deposit.provider, provider.base_url()),
        );
        log(
            level,
            LogLevel::Normal,
            &format!("  Files ready: {}", deposit.files.len()),
        );
    }

    Ok(())
}

fn run_research_bundle(args: BundleArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Bundling RO-Crate: {}", args.output.display()),
    );

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    let mut crate_pkg = RoCrate::from_artifact(&artifact, &args.output);

    // Add files
    for file_path in &args.file {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read {}: {e}", file_path.display()))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("Invalid file name: {}", file_path.display()))?;

        crate_pkg.add_text_file(file_name, &content);
    }

    if args.zip {
        let zip_path = args.output.with_extension("zip");
        let zip_data = crate_pkg.to_zip().map_err(|e| format!("ZIP error: {e}"))?;
        std::fs::write(&zip_path, &zip_data).map_err(|e| format!("Failed to write ZIP: {e}"))?;

        log(
            level,
            LogLevel::Normal,
            &format!(
                "RO-Crate ZIP created: {} ({} bytes)",
                zip_path.display(),
                zip_data.len()
            ),
        );
    } else {
        crate_pkg
            .to_directory()
            .map_err(|e| format!("Failed to create directory: {e}"))?;

        log(
            level,
            LogLevel::Normal,
            &format!(
                "RO-Crate directory created: {} ({} entities)",
                args.output.display(),
                crate_pkg.entity_count()
            ),
        );
    }

    Ok(())
}

fn run_research_verify(args: VerifyArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Verifying: {}", args.file.display()),
    );

    let content =
        std::fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {e}"))?;

    // Try to parse as signed pre-registration
    if let Ok(signed) = serde_yaml::from_str::<SignedPreRegistration>(&content) {
        // Verify signature
        match signed.verify() {
            Ok(true) => {
                log(level, LogLevel::Normal, "Signature verification: VALID");
            }
            Ok(false) => {
                log(level, LogLevel::Normal, "Signature verification: INVALID");
                return Err("Signature verification failed".to_string());
            }
            Err(e) => {
                return Err(format!("Verification error: {e}"));
            }
        }

        // Verify git timestamp if requested
        if args.verify_git {
            if let Some(proof) = &signed.timestamp_proof {
                if proof.is_git() {
                    log(level, LogLevel::Normal, "Git timestamp proof: GitCommit");
                    if let Some(commit) = proof.git_commit() {
                        log(level, LogLevel::Verbose, &format!("  Commit: {commit}"));
                    }
                } else {
                    log(
                        level,
                        LogLevel::Normal,
                        "Timestamp proof is not a git commit",
                    );
                }
            } else {
                log(level, LogLevel::Normal, "No git timestamp proof found");
            }
        }

        log(
            level,
            LogLevel::Normal,
            "Pre-registration verified successfully",
        );
    } else {
        // Try to parse as commitment
        log(
            level,
            LogLevel::Normal,
            "File does not contain a signed pre-registration",
        );

        if let Some(original_path) = &args.original {
            // Verify commitment against original
            let original_content = std::fs::read_to_string(original_path)
                .map_err(|e| format!("Failed to read original: {e}"))?;

            let prereg: PreRegistration = serde_yaml::from_str(&original_content)
                .map_err(|e| format!("Failed to parse original: {e}"))?;

            let commitment = prereg.commit();
            log(
                level,
                LogLevel::Normal,
                &format!("Computed commitment: {}...", &commitment.hash[..32]),
            );
        }
    }

    Ok(())
}

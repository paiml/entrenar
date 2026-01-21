//! Research command types for academic workflows

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use super::types::{ArchiveProviderArg, ArtifactTypeArg, CitationFormat, ExportFormat, LicenseArg};

/// Arguments for the research command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchArgs {
    /// Research subcommand to execute
    #[command(subcommand)]
    pub command: ResearchCommand,
}

/// Research subcommands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum ResearchCommand {
    /// Initialize a new research artifact
    Init(ResearchInitArgs),

    /// Create a pre-registration with cryptographic commitment
    Preregister(PreregisterArgs),

    /// Generate citations in various formats
    Cite(CiteArgs),

    /// Export artifacts to various formats
    Export(ExportArgs),

    /// Deposit to academic archives
    Deposit(DepositArgs),

    /// Bundle artifacts into RO-Crate package
    Bundle(BundleArgs),

    /// Verify pre-registration commitments or signatures
    Verify(VerifyArgs),
}

/// Arguments for research init command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchInitArgs {
    /// Artifact ID (unique identifier)
    #[arg(long)]
    pub id: String,

    /// Artifact title
    #[arg(long)]
    pub title: String,

    /// Artifact type
    #[arg(long, default_value = "dataset")]
    pub artifact_type: ArtifactTypeArg,

    /// License (e.g., CC-BY-4.0, MIT, Apache-2.0)
    #[arg(long, default_value = "cc-by-4.0")]
    pub license: LicenseArg,

    /// Output path for artifact YAML
    #[arg(short, long, default_value = "artifact.yaml")]
    pub output: PathBuf,

    /// Author name
    #[arg(long)]
    pub author: Option<String>,

    /// Author ORCID (format: 0000-0002-1825-0097)
    #[arg(long)]
    pub orcid: Option<String>,

    /// Author affiliation
    #[arg(long)]
    pub affiliation: Option<String>,

    /// Description of the artifact
    #[arg(long)]
    pub description: Option<String>,

    /// Keywords (comma-separated)
    #[arg(long)]
    pub keywords: Option<String>,

    /// DOI (if already assigned)
    #[arg(long)]
    pub doi: Option<String>,
}

/// Arguments for preregister command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct PreregisterArgs {
    /// Research question or title
    #[arg(long)]
    pub title: String,

    /// Hypothesis being tested
    #[arg(long)]
    pub hypothesis: String,

    /// Methodology description
    #[arg(long)]
    pub methodology: String,

    /// Statistical analysis plan
    #[arg(long)]
    pub analysis_plan: String,

    /// Additional notes
    #[arg(long)]
    pub notes: Option<String>,

    /// Output path for pre-registration
    #[arg(short, long, default_value = "preregistration.yaml")]
    pub output: PathBuf,

    /// Path to Ed25519 private key for signing
    #[arg(long)]
    pub sign_key: Option<PathBuf>,

    /// Add git commit hash as timestamp proof
    #[arg(long)]
    pub git_timestamp: bool,
}

/// Arguments for cite command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct CiteArgs {
    /// Path to artifact YAML file
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Publication year
    #[arg(long)]
    pub year: u16,

    /// Output format
    #[arg(short, long, default_value = "bibtex")]
    pub format: CitationFormat,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Journal name
    #[arg(long)]
    pub journal: Option<String>,

    /// Volume number
    #[arg(long)]
    pub volume: Option<String>,

    /// Page range
    #[arg(long)]
    pub pages: Option<String>,

    /// URL
    #[arg(long)]
    pub url: Option<String>,
}

/// Arguments for export command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ExportArgs {
    /// Path to artifact or document
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Export format
    #[arg(short, long)]
    pub format: ExportFormat,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Anonymize for double-blind review
    #[arg(long)]
    pub anonymize: bool,

    /// Salt for anonymization (required with --anonymize)
    #[arg(long)]
    pub anon_salt: Option<String>,

    /// Jupyter kernel (for notebook export)
    #[arg(long, default_value = "python3")]
    pub kernel: String,
}

/// Arguments for deposit command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct DepositArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Archive provider
    #[arg(short, long)]
    pub provider: ArchiveProviderArg,

    /// API token (or use env var ZENODO_TOKEN, etc.)
    #[arg(long)]
    pub token: Option<String>,

    /// Use sandbox/test environment
    #[arg(long)]
    pub sandbox: bool,

    /// Community to submit to
    #[arg(long)]
    pub community: Option<String>,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Dry run (validate without uploading)
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for bundle command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct BundleArgs {
    /// Path to artifact YAML
    #[arg(value_name = "ARTIFACT")]
    pub artifact: PathBuf,

    /// Output directory for RO-Crate
    #[arg(short, long)]
    pub output: PathBuf,

    /// Files to include (can be repeated)
    #[arg(short, long)]
    pub file: Vec<PathBuf>,

    /// Create ZIP archive instead of directory
    #[arg(long)]
    pub zip: bool,

    /// Include citation graph
    #[arg(long)]
    pub include_citations: bool,
}

/// Arguments for verify command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct VerifyArgs {
    /// Path to pre-registration or signed artifact
    #[arg(value_name = "FILE")]
    pub file: PathBuf,

    /// Path to Ed25519 public key for signature verification
    #[arg(long)]
    pub public_key: Option<PathBuf>,

    /// Original content to verify against commitment
    #[arg(long)]
    pub original: Option<PathBuf>,

    /// Verify git timestamp proof
    #[arg(long)]
    pub verify_git: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::cli::parse_args;

    #[test]
    fn test_parse_research_init() {
        let cli = parse_args([
            "entrenar",
            "research",
            "init",
            "--id",
            "dataset-2024",
            "--title",
            "My Dataset",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Init(init_args) => {
                    assert_eq!(init_args.id, "dataset-2024");
                    assert_eq!(init_args.title, "My Dataset");
                    assert_eq!(init_args.artifact_type, ArtifactTypeArg::Dataset);
                    assert_eq!(init_args.license, LicenseArg::CcBy4);
                }
                _ => panic!("Expected Init subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_preregister() {
        let cli = parse_args([
            "entrenar",
            "research",
            "preregister",
            "--title",
            "Effect of X on Y",
            "--hypothesis",
            "X improves Y by 20%",
            "--methodology",
            "RCT, n=100",
            "--analysis-plan",
            "t-test, alpha=0.05",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Preregister(prereg_args) => {
                    assert_eq!(prereg_args.title, "Effect of X on Y");
                    assert_eq!(prereg_args.hypothesis, "X improves Y by 20%");
                }
                _ => panic!("Expected Preregister subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_cite() {
        let cli = parse_args([
            "entrenar",
            "research",
            "cite",
            "artifact.yaml",
            "--year",
            "2024",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Cite(cite_args) => {
                    assert_eq!(cite_args.artifact, PathBuf::from("artifact.yaml"));
                    assert_eq!(cite_args.year, 2024);
                    assert_eq!(cite_args.format, CitationFormat::Bibtex);
                }
                _ => panic!("Expected Cite subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_export() {
        let cli = parse_args([
            "entrenar",
            "research",
            "export",
            "document.md",
            "--format",
            "notebook",
            "--output",
            "analysis.ipynb",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Export(export_args) => {
                    assert_eq!(export_args.input, PathBuf::from("document.md"));
                    assert_eq!(export_args.format, ExportFormat::Notebook);
                }
                _ => panic!("Expected Export subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_deposit() {
        let cli = parse_args([
            "entrenar",
            "research",
            "deposit",
            "artifact.yaml",
            "--provider",
            "zenodo",
            "--sandbox",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Deposit(deposit_args) => {
                    assert_eq!(deposit_args.provider, ArchiveProviderArg::Zenodo);
                    assert!(deposit_args.sandbox);
                }
                _ => panic!("Expected Deposit subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_bundle() {
        let cli = parse_args([
            "entrenar",
            "research",
            "bundle",
            "artifact.yaml",
            "--output",
            "./ro-crate",
            "--zip",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Bundle(bundle_args) => {
                    assert_eq!(bundle_args.output, PathBuf::from("./ro-crate"));
                    assert!(bundle_args.zip);
                }
                _ => panic!("Expected Bundle subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_verify() {
        let cli = parse_args([
            "entrenar",
            "research",
            "verify",
            "preregistration.yaml",
            "--verify-git",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Verify(verify_args) => {
                    assert_eq!(verify_args.file, PathBuf::from("preregistration.yaml"));
                    assert!(verify_args.verify_git);
                }
                _ => panic!("Expected Verify subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    // Additional coverage tests for optional fields

    #[test]
    fn test_parse_research_init_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "init",
            "--id",
            "model-2024",
            "--title",
            "My Model",
            "--artifact-type",
            "model",
            "--license",
            "mit",
            "--output",
            "model.yaml",
            "--author",
            "John Doe",
            "--orcid",
            "0000-0002-1825-0097",
            "--affiliation",
            "University of Test",
            "--description",
            "A test model",
            "--keywords",
            "ml,ai,test",
            "--doi",
            "10.5281/zenodo.12345",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Init(init_args) => {
                    assert_eq!(init_args.artifact_type, ArtifactTypeArg::Model);
                    assert_eq!(init_args.license, LicenseArg::Mit);
                    assert_eq!(init_args.author, Some("John Doe".to_string()));
                    assert_eq!(init_args.orcid, Some("0000-0002-1825-0097".to_string()));
                    assert_eq!(
                        init_args.affiliation,
                        Some("University of Test".to_string())
                    );
                    assert_eq!(init_args.description, Some("A test model".to_string()));
                    assert_eq!(init_args.keywords, Some("ml,ai,test".to_string()));
                    assert_eq!(init_args.doi, Some("10.5281/zenodo.12345".to_string()));
                }
                _ => panic!("Expected Init subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_preregister_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "preregister",
            "--title",
            "Test Study",
            "--hypothesis",
            "H1: X > Y",
            "--methodology",
            "Survey study",
            "--analysis-plan",
            "ANOVA",
            "--notes",
            "Additional notes here",
            "--output",
            "prereg.yaml",
            "--git-timestamp",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Preregister(prereg_args) => {
                    assert_eq!(prereg_args.notes, Some("Additional notes here".to_string()));
                    assert_eq!(prereg_args.output, PathBuf::from("prereg.yaml"));
                    assert!(prereg_args.git_timestamp);
                }
                _ => panic!("Expected Preregister subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_cite_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "cite",
            "artifact.yaml",
            "--year",
            "2024",
            "--format",
            "cff",
            "--output",
            "citation.txt",
            "--journal",
            "Nature ML",
            "--volume",
            "12",
            "--pages",
            "100-120",
            "--url",
            "https://example.com",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Cite(cite_args) => {
                    assert_eq!(cite_args.format, CitationFormat::Cff);
                    assert_eq!(cite_args.output, Some(PathBuf::from("citation.txt")));
                    assert_eq!(cite_args.journal, Some("Nature ML".to_string()));
                    assert_eq!(cite_args.volume, Some("12".to_string()));
                    assert_eq!(cite_args.pages, Some("100-120".to_string()));
                    assert_eq!(cite_args.url, Some("https://example.com".to_string()));
                }
                _ => panic!("Expected Cite subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_export_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "export",
            "doc.md",
            "--format",
            "html",
            "--output",
            "doc.html",
            "--anonymize",
            "--anon-salt",
            "secret123",
            "--kernel",
            "julia",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Export(export_args) => {
                    assert_eq!(export_args.format, ExportFormat::Html);
                    assert!(export_args.anonymize);
                    assert_eq!(export_args.anon_salt, Some("secret123".to_string()));
                    assert_eq!(export_args.kernel, "julia");
                }
                _ => panic!("Expected Export subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_deposit_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "deposit",
            "artifact.yaml",
            "--provider",
            "figshare",
            "--token",
            "my-api-token",
            "--community",
            "ml-research",
            "--file",
            "data.csv",
            "--file",
            "model.bin",
            "--dry-run",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Deposit(deposit_args) => {
                    assert_eq!(deposit_args.provider, ArchiveProviderArg::Figshare);
                    assert_eq!(deposit_args.token, Some("my-api-token".to_string()));
                    assert_eq!(deposit_args.community, Some("ml-research".to_string()));
                    assert_eq!(deposit_args.file.len(), 2);
                    assert!(deposit_args.dry_run);
                }
                _ => panic!("Expected Deposit subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_bundle_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "bundle",
            "artifact.yaml",
            "--output",
            "./bundle",
            "--file",
            "data.csv",
            "--include-citations",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Bundle(bundle_args) => {
                    assert_eq!(bundle_args.file.len(), 1);
                    assert!(bundle_args.include_citations);
                    assert!(!bundle_args.zip);
                }
                _ => panic!("Expected Bundle subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_parse_research_verify_all_options() {
        let cli = parse_args([
            "entrenar",
            "research",
            "verify",
            "prereg.yaml",
            "--public-key",
            "key.pub",
            "--original",
            "original.yaml",
        ])
        .unwrap();

        match cli.command {
            crate::config::cli::Command::Research(args) => match args.command {
                ResearchCommand::Verify(verify_args) => {
                    assert_eq!(verify_args.public_key, Some(PathBuf::from("key.pub")));
                    assert_eq!(verify_args.original, Some(PathBuf::from("original.yaml")));
                    assert!(!verify_args.verify_git);
                }
                _ => panic!("Expected Verify subcommand"),
            },
            _ => panic!("Expected Research command"),
        }
    }

    #[test]
    fn test_research_args_debug() {
        let args = ResearchInitArgs {
            id: "test".to_string(),
            title: "Test".to_string(),
            artifact_type: ArtifactTypeArg::Dataset,
            license: LicenseArg::CcBy4,
            output: PathBuf::from("test.yaml"),
            author: None,
            orcid: None,
            affiliation: None,
            description: None,
            keywords: None,
            doi: None,
        };
        let debug = format!("{args:?}");
        assert!(debug.contains("ResearchInitArgs"));
    }

    #[test]
    fn test_research_command_clone() {
        let init = ResearchCommand::Init(ResearchInitArgs {
            id: "test".to_string(),
            title: "Test".to_string(),
            artifact_type: ArtifactTypeArg::Dataset,
            license: LicenseArg::CcBy4,
            output: PathBuf::from("test.yaml"),
            author: None,
            orcid: None,
            affiliation: None,
            description: None,
            keywords: None,
            doi: None,
        });
        let cloned = init.clone();
        assert_eq!(init, cloned);
    }
}

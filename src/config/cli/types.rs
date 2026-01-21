//! CLI type enums for formats, licenses, artifact types, etc.

/// Output format for info command
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Yaml,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            _ => Err(format!(
                "Unknown output format: {s}. Valid formats: text, json, yaml"
            )),
        }
    }
}

/// Artifact type for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArtifactTypeArg {
    #[default]
    Dataset,
    Paper,
    Model,
    Code,
    Notebook,
    Workflow,
}

impl std::str::FromStr for ArtifactTypeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dataset" => Ok(ArtifactTypeArg::Dataset),
            "paper" => Ok(ArtifactTypeArg::Paper),
            "model" => Ok(ArtifactTypeArg::Model),
            "code" => Ok(ArtifactTypeArg::Code),
            "notebook" => Ok(ArtifactTypeArg::Notebook),
            "workflow" => Ok(ArtifactTypeArg::Workflow),
            _ => Err(format!(
                "Unknown artifact type: {s}. Valid types: dataset, paper, model, code, notebook, workflow"
            )),
        }
    }
}

impl std::fmt::Display for ArtifactTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactTypeArg::Dataset => write!(f, "dataset"),
            ArtifactTypeArg::Paper => write!(f, "paper"),
            ArtifactTypeArg::Model => write!(f, "model"),
            ArtifactTypeArg::Code => write!(f, "code"),
            ArtifactTypeArg::Notebook => write!(f, "notebook"),
            ArtifactTypeArg::Workflow => write!(f, "workflow"),
        }
    }
}

/// License for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LicenseArg {
    #[default]
    CcBy4,
    CcBySa4,
    Cc0,
    Mit,
    Apache2,
    Gpl3,
    Bsd3,
}

impl std::str::FromStr for LicenseArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace(['-', '.'], "").as_str() {
            "ccby4" | "ccby40" => Ok(LicenseArg::CcBy4),
            "ccbysa4" | "ccbysa40" => Ok(LicenseArg::CcBySa4),
            "cc0" => Ok(LicenseArg::Cc0),
            "mit" => Ok(LicenseArg::Mit),
            "apache2" | "apache20" => Ok(LicenseArg::Apache2),
            "gpl3" | "gplv3" => Ok(LicenseArg::Gpl3),
            "bsd3" | "bsd3clause" => Ok(LicenseArg::Bsd3),
            _ => Err(format!(
                "Unknown license: {s}. Valid licenses: CC-BY-4.0, CC-BY-SA-4.0, CC0, MIT, Apache-2.0, GPL-3.0, BSD-3-Clause"
            )),
        }
    }
}

impl std::fmt::Display for LicenseArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseArg::CcBy4 => write!(f, "CC-BY-4.0"),
            LicenseArg::CcBySa4 => write!(f, "CC-BY-SA-4.0"),
            LicenseArg::Cc0 => write!(f, "CC0"),
            LicenseArg::Mit => write!(f, "MIT"),
            LicenseArg::Apache2 => write!(f, "Apache-2.0"),
            LicenseArg::Gpl3 => write!(f, "GPL-3.0"),
            LicenseArg::Bsd3 => write!(f, "BSD-3-Clause"),
        }
    }
}

/// Citation format for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CitationFormat {
    #[default]
    Bibtex,
    Cff,
    Json,
}

impl std::str::FromStr for CitationFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bibtex" | "bib" => Ok(CitationFormat::Bibtex),
            "cff" | "citation.cff" => Ok(CitationFormat::Cff),
            "json" => Ok(CitationFormat::Json),
            _ => Err(format!(
                "Unknown citation format: {s}. Valid formats: bibtex, cff, json"
            )),
        }
    }
}

impl std::fmt::Display for CitationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CitationFormat::Bibtex => write!(f, "bibtex"),
            CitationFormat::Cff => write!(f, "cff"),
            CitationFormat::Json => write!(f, "json"),
        }
    }
}

/// Export format for CLI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Notebook,
    Html,
    AnonymizedJson,
    RoCrate,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "notebook" | "ipynb" | "jupyter" => Ok(ExportFormat::Notebook),
            "html" => Ok(ExportFormat::Html),
            "anonymized" | "anon" | "anonymized-json" => Ok(ExportFormat::AnonymizedJson),
            "ro-crate" | "rocrate" => Ok(ExportFormat::RoCrate),
            _ => Err(format!(
                "Unknown export format: {s}. Valid formats: notebook, html, anonymized, ro-crate"
            )),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Notebook => write!(f, "notebook"),
            ExportFormat::Html => write!(f, "html"),
            ExportFormat::AnonymizedJson => write!(f, "anonymized-json"),
            ExportFormat::RoCrate => write!(f, "ro-crate"),
        }
    }
}

/// Archive provider for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArchiveProviderArg {
    #[default]
    Zenodo,
    Figshare,
    Dryad,
    Dataverse,
}

impl std::str::FromStr for ArchiveProviderArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "zenodo" => Ok(ArchiveProviderArg::Zenodo),
            "figshare" => Ok(ArchiveProviderArg::Figshare),
            "dryad" => Ok(ArchiveProviderArg::Dryad),
            "dataverse" => Ok(ArchiveProviderArg::Dataverse),
            _ => Err(format!(
                "Unknown archive provider: {s}. Valid providers: zenodo, figshare, dryad, dataverse"
            )),
        }
    }
}

impl std::fmt::Display for ArchiveProviderArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveProviderArg::Zenodo => write!(f, "zenodo"),
            ArchiveProviderArg::Figshare => write!(f, "figshare"),
            ArchiveProviderArg::Dryad => write!(f, "dryad"),
            ArchiveProviderArg::Dataverse => write!(f, "dataverse"),
        }
    }
}

/// Shell type for completions
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ShellType {
    #[default]
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

impl std::str::FromStr for ShellType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bash" => Ok(ShellType::Bash),
            "zsh" => Ok(ShellType::Zsh),
            "fish" => Ok(ShellType::Fish),
            "powershell" | "ps" => Ok(ShellType::PowerShell),
            _ => Err(format!(
                "Unknown shell: {s}. Valid shells: bash, zsh, fish, powershell"
            )),
        }
    }
}

impl std::fmt::Display for ShellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellType::Bash => write!(f, "bash"),
            ShellType::Zsh => write!(f, "zsh"),
            ShellType::Fish => write!(f, "fish"),
            ShellType::PowerShell => write!(f, "powershell"),
        }
    }
}

/// Inspection mode
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InspectMode {
    #[default]
    Summary,
    Outliers,
    Distribution,
    Schema,
}

impl std::str::FromStr for InspectMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "summary" => Ok(InspectMode::Summary),
            "outliers" => Ok(InspectMode::Outliers),
            "distribution" | "dist" => Ok(InspectMode::Distribution),
            "schema" => Ok(InspectMode::Schema),
            _ => Err(format!(
                "Unknown inspect mode: {s}. Valid modes: summary, outliers, distribution, schema"
            )),
        }
    }
}

impl std::fmt::Display for InspectMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectMode::Summary => write!(f, "summary"),
            InspectMode::Outliers => write!(f, "outliers"),
            InspectMode::Distribution => write!(f, "distribution"),
            InspectMode::Schema => write!(f, "schema"),
        }
    }
}

/// Audit type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AuditType {
    #[default]
    Bias,
    Fairness,
    Privacy,
    Security,
}

impl std::str::FromStr for AuditType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bias" => Ok(AuditType::Bias),
            "fairness" => Ok(AuditType::Fairness),
            "privacy" => Ok(AuditType::Privacy),
            "security" => Ok(AuditType::Security),
            _ => Err(format!(
                "Unknown audit type: {s}. Valid types: bias, fairness, privacy, security"
            )),
        }
    }
}

impl std::fmt::Display for AuditType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditType::Bias => write!(f, "bias"),
            AuditType::Fairness => write!(f, "fairness"),
            AuditType::Privacy => write!(f, "privacy"),
            AuditType::Security => write!(f, "security"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_from_str() {
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Text);
    }

    #[test]
    fn test_artifact_type_from_str() {
        assert_eq!(
            "dataset".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Dataset
        );
        assert_eq!(
            "paper".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Paper
        );
        assert_eq!(
            "model".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Model
        );
        assert_eq!(
            "code".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Code
        );
        assert_eq!(
            "notebook".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Notebook
        );
        assert_eq!(
            "workflow".parse::<ArtifactTypeArg>().unwrap(),
            ArtifactTypeArg::Workflow
        );
        assert!("invalid".parse::<ArtifactTypeArg>().is_err());
    }

    #[test]
    fn test_artifact_type_display() {
        assert_eq!(format!("{}", ArtifactTypeArg::Dataset), "dataset");
        assert_eq!(format!("{}", ArtifactTypeArg::Paper), "paper");
        assert_eq!(format!("{}", ArtifactTypeArg::Model), "model");
        assert_eq!(format!("{}", ArtifactTypeArg::Code), "code");
        assert_eq!(format!("{}", ArtifactTypeArg::Notebook), "notebook");
        assert_eq!(format!("{}", ArtifactTypeArg::Workflow), "workflow");
    }

    #[test]
    fn test_artifact_type_default() {
        assert_eq!(ArtifactTypeArg::default(), ArtifactTypeArg::Dataset);
    }

    #[test]
    fn test_license_from_str() {
        assert_eq!(
            "cc-by-4.0".parse::<LicenseArg>().unwrap(),
            LicenseArg::CcBy4
        );
        assert_eq!(
            "cc-by-sa-4.0".parse::<LicenseArg>().unwrap(),
            LicenseArg::CcBySa4
        );
        assert_eq!("cc0".parse::<LicenseArg>().unwrap(), LicenseArg::Cc0);
        assert_eq!("mit".parse::<LicenseArg>().unwrap(), LicenseArg::Mit);
        assert_eq!(
            "apache-2.0".parse::<LicenseArg>().unwrap(),
            LicenseArg::Apache2
        );
        assert_eq!("gpl3".parse::<LicenseArg>().unwrap(), LicenseArg::Gpl3);
        assert_eq!("gplv3".parse::<LicenseArg>().unwrap(), LicenseArg::Gpl3);
        assert_eq!("bsd3".parse::<LicenseArg>().unwrap(), LicenseArg::Bsd3);
        assert!("invalid".parse::<LicenseArg>().is_err());
    }

    #[test]
    fn test_license_display() {
        assert_eq!(format!("{}", LicenseArg::CcBy4), "CC-BY-4.0");
        assert_eq!(format!("{}", LicenseArg::CcBySa4), "CC-BY-SA-4.0");
        assert_eq!(format!("{}", LicenseArg::Cc0), "CC0");
        assert_eq!(format!("{}", LicenseArg::Mit), "MIT");
        assert_eq!(format!("{}", LicenseArg::Apache2), "Apache-2.0");
        assert_eq!(format!("{}", LicenseArg::Gpl3), "GPL-3.0");
        assert_eq!(format!("{}", LicenseArg::Bsd3), "BSD-3-Clause");
    }

    #[test]
    fn test_license_default() {
        assert_eq!(LicenseArg::default(), LicenseArg::CcBy4);
    }

    #[test]
    fn test_citation_format_from_str() {
        assert_eq!(
            "bibtex".parse::<CitationFormat>().unwrap(),
            CitationFormat::Bibtex
        );
        assert_eq!(
            "bib".parse::<CitationFormat>().unwrap(),
            CitationFormat::Bibtex
        );
        assert_eq!(
            "cff".parse::<CitationFormat>().unwrap(),
            CitationFormat::Cff
        );
        assert_eq!(
            "json".parse::<CitationFormat>().unwrap(),
            CitationFormat::Json
        );
        assert!("invalid".parse::<CitationFormat>().is_err());
    }

    #[test]
    fn test_citation_format_display() {
        assert_eq!(format!("{}", CitationFormat::Bibtex), "bibtex");
        assert_eq!(format!("{}", CitationFormat::Cff), "cff");
        assert_eq!(format!("{}", CitationFormat::Json), "json");
    }

    #[test]
    fn test_citation_format_default() {
        assert_eq!(CitationFormat::default(), CitationFormat::Bibtex);
    }

    #[test]
    fn test_export_format_from_str() {
        assert_eq!(
            "notebook".parse::<ExportFormat>().unwrap(),
            ExportFormat::Notebook
        );
        assert_eq!(
            "ipynb".parse::<ExportFormat>().unwrap(),
            ExportFormat::Notebook
        );
        assert_eq!(
            "jupyter".parse::<ExportFormat>().unwrap(),
            ExportFormat::Notebook
        );
        assert_eq!("html".parse::<ExportFormat>().unwrap(), ExportFormat::Html);
        assert_eq!(
            "anonymized".parse::<ExportFormat>().unwrap(),
            ExportFormat::AnonymizedJson
        );
        assert_eq!(
            "anon".parse::<ExportFormat>().unwrap(),
            ExportFormat::AnonymizedJson
        );
        assert_eq!(
            "ro-crate".parse::<ExportFormat>().unwrap(),
            ExportFormat::RoCrate
        );
        assert_eq!(
            "rocrate".parse::<ExportFormat>().unwrap(),
            ExportFormat::RoCrate
        );
        assert!("invalid".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(format!("{}", ExportFormat::Notebook), "notebook");
        assert_eq!(format!("{}", ExportFormat::Html), "html");
        assert_eq!(
            format!("{}", ExportFormat::AnonymizedJson),
            "anonymized-json"
        );
        assert_eq!(format!("{}", ExportFormat::RoCrate), "ro-crate");
    }

    #[test]
    fn test_archive_provider_from_str() {
        assert_eq!(
            "zenodo".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Zenodo
        );
        assert_eq!(
            "figshare".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Figshare
        );
        assert_eq!(
            "dryad".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Dryad
        );
        assert_eq!(
            "dataverse".parse::<ArchiveProviderArg>().unwrap(),
            ArchiveProviderArg::Dataverse
        );
        assert!("invalid".parse::<ArchiveProviderArg>().is_err());
    }

    #[test]
    fn test_archive_provider_display() {
        assert_eq!(format!("{}", ArchiveProviderArg::Zenodo), "zenodo");
        assert_eq!(format!("{}", ArchiveProviderArg::Figshare), "figshare");
        assert_eq!(format!("{}", ArchiveProviderArg::Dryad), "dryad");
        assert_eq!(format!("{}", ArchiveProviderArg::Dataverse), "dataverse");
    }

    #[test]
    fn test_archive_provider_default() {
        assert_eq!(ArchiveProviderArg::default(), ArchiveProviderArg::Zenodo);
    }

    #[test]
    fn test_shell_type_from_str() {
        assert_eq!("bash".parse::<ShellType>().unwrap(), ShellType::Bash);
        assert_eq!("zsh".parse::<ShellType>().unwrap(), ShellType::Zsh);
        assert_eq!("fish".parse::<ShellType>().unwrap(), ShellType::Fish);
        assert_eq!(
            "powershell".parse::<ShellType>().unwrap(),
            ShellType::PowerShell
        );
        assert_eq!("ps".parse::<ShellType>().unwrap(), ShellType::PowerShell);
        assert!("invalid".parse::<ShellType>().is_err());
    }

    #[test]
    fn test_shell_type_display() {
        assert_eq!(format!("{}", ShellType::Bash), "bash");
        assert_eq!(format!("{}", ShellType::Zsh), "zsh");
        assert_eq!(format!("{}", ShellType::Fish), "fish");
        assert_eq!(format!("{}", ShellType::PowerShell), "powershell");
    }

    #[test]
    fn test_shell_type_default() {
        assert_eq!(ShellType::default(), ShellType::Bash);
    }

    #[test]
    fn test_inspect_mode_from_str() {
        assert_eq!(
            "summary".parse::<InspectMode>().unwrap(),
            InspectMode::Summary
        );
        assert_eq!(
            "outliers".parse::<InspectMode>().unwrap(),
            InspectMode::Outliers
        );
        assert_eq!(
            "distribution".parse::<InspectMode>().unwrap(),
            InspectMode::Distribution
        );
        assert_eq!(
            "dist".parse::<InspectMode>().unwrap(),
            InspectMode::Distribution
        );
        assert_eq!(
            "schema".parse::<InspectMode>().unwrap(),
            InspectMode::Schema
        );
        assert!("invalid".parse::<InspectMode>().is_err());
    }

    #[test]
    fn test_inspect_mode_display() {
        assert_eq!(format!("{}", InspectMode::Summary), "summary");
        assert_eq!(format!("{}", InspectMode::Outliers), "outliers");
        assert_eq!(format!("{}", InspectMode::Distribution), "distribution");
        assert_eq!(format!("{}", InspectMode::Schema), "schema");
    }

    #[test]
    fn test_inspect_mode_default() {
        assert_eq!(InspectMode::default(), InspectMode::Summary);
    }

    #[test]
    fn test_audit_type_from_str() {
        assert_eq!("bias".parse::<AuditType>().unwrap(), AuditType::Bias);
        assert_eq!(
            "fairness".parse::<AuditType>().unwrap(),
            AuditType::Fairness
        );
        assert_eq!("privacy".parse::<AuditType>().unwrap(), AuditType::Privacy);
        assert_eq!(
            "security".parse::<AuditType>().unwrap(),
            AuditType::Security
        );
        assert!("invalid".parse::<AuditType>().is_err());
    }

    #[test]
    fn test_audit_type_display() {
        assert_eq!(format!("{}", AuditType::Bias), "bias");
        assert_eq!(format!("{}", AuditType::Fairness), "fairness");
        assert_eq!(format!("{}", AuditType::Privacy), "privacy");
        assert_eq!(format!("{}", AuditType::Security), "security");
    }

    #[test]
    fn test_audit_type_default() {
        assert_eq!(AuditType::default(), AuditType::Bias);
    }
}

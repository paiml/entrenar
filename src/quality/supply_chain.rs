//! Supply Chain Auditing (ENT-006)
//!
//! Provides cargo-deny integration for dependency vulnerability scanning
//! and license compliance checking.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for supply chain auditing
#[derive(Debug, Error)]
pub enum SupplyChainError {
    #[error("Failed to parse cargo-deny output: {0}")]
    ParseError(String),

    #[error("Vulnerable dependency found: {0}")]
    VulnerabilityFound(String),
}

/// Result type for supply chain operations
pub type Result<T> = std::result::Result<T, SupplyChainError>;

/// Severity level for security advisories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Severity {
    /// Informational only
    None,
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

impl Severity {
    /// Parse severity from string (infallible, returns None for unknown)
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            _ => Self::None,
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Audit status for a dependency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditStatus {
    /// No issues found
    Clean,
    /// Warnings present (license issues, yanked versions)
    Warning,
    /// Vulnerabilities found
    Vulnerable,
}

impl AuditStatus {
    /// Returns true if the status indicates a failure condition
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Vulnerable)
    }

    /// Returns true if the status indicates any issues
    pub fn has_issues(&self) -> bool {
        !matches!(self, Self::Clean)
    }
}

/// Security advisory information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Advisory {
    /// Advisory ID (e.g., RUSTSEC-2021-0001)
    pub id: String,

    /// Severity level
    pub severity: Severity,

    /// Short title/description
    pub title: String,
}

impl Advisory {
    /// Create a new advisory
    pub fn new(id: impl Into<String>, severity: Severity, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            severity,
            title: title.into(),
        }
    }
}

/// Dependency audit result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyAudit {
    /// Crate name
    pub crate_name: String,

    /// Version string
    pub version: String,

    /// Security advisories affecting this crate
    pub advisories: Vec<Advisory>,

    /// License identifier (e.g., "MIT", "Apache-2.0")
    pub license: String,

    /// Overall audit status
    pub audit_status: AuditStatus,
}

impl DependencyAudit {
    /// Create a new clean dependency audit
    pub fn clean(
        crate_name: impl Into<String>,
        version: impl Into<String>,
        license: impl Into<String>,
    ) -> Self {
        Self {
            crate_name: crate_name.into(),
            version: version.into(),
            advisories: Vec::new(),
            license: license.into(),
            audit_status: AuditStatus::Clean,
        }
    }

    /// Create a vulnerable dependency audit
    pub fn vulnerable(
        crate_name: impl Into<String>,
        version: impl Into<String>,
        license: impl Into<String>,
        advisories: Vec<Advisory>,
    ) -> Self {
        Self {
            crate_name: crate_name.into(),
            version: version.into(),
            advisories,
            license: license.into(),
            audit_status: AuditStatus::Vulnerable,
        }
    }

    /// Parse dependency audits from cargo-deny JSON output
    ///
    /// # Arguments
    ///
    /// * `json` - JSON output from `cargo deny check --format json`
    ///
    /// # Example JSON format
    ///
    /// ```json
    /// {
    ///   "type": "diagnostic",
    ///   "fields": {
    ///     "graphs": [...],
    ///     "severity": "error",
    ///     "code": "A001",
    ///     "message": "Detected security vulnerability",
    ///     "labels": [{"span": {"crate": {"name": "foo", "version": "1.0.0"}}}]
    ///   }
    /// }
    /// ```
    pub fn from_cargo_deny_output(json: &str) -> Result<Vec<Self>> {
        let mut audits = Vec::new();

        // cargo deny outputs newline-delimited JSON
        for line in json.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| SupplyChainError::ParseError(e.to_string()))?;

            // Skip non-diagnostic messages
            if value.get("type").and_then(|t| t.as_str()) != Some("diagnostic") {
                continue;
            }

            let fields = match value.get("fields") {
                Some(f) => f,
                None => continue,
            };

            // Extract severity
            let severity_str = fields
                .get("severity")
                .and_then(|s| s.as_str())
                .unwrap_or("none");

            let is_vulnerability = severity_str == "error"
                && fields
                    .get("code")
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| c.starts_with('A'));

            if !is_vulnerability {
                continue;
            }

            // Extract crate info from labels
            if let Some(labels) = fields.get("labels").and_then(|l| l.as_array()) {
                for label in labels {
                    if let Some(span) = label.get("span") {
                        if let Some(krate) = span.get("crate") {
                            let crate_name = krate
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let version = krate
                                .get("version")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let message = fields
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown vulnerability")
                                .to_string();

                            let code = fields
                                .get("code")
                                .and_then(|c| c.as_str())
                                .unwrap_or("UNKNOWN")
                                .to_string();

                            let advisory = Advisory::new(code, Severity::High, message);

                            audits.push(Self::vulnerable(
                                crate_name,
                                version,
                                "unknown",
                                vec![advisory],
                            ));
                        }
                    }
                }
            }
        }

        Ok(audits)
    }

    /// Returns true if this dependency has vulnerabilities
    pub fn is_vulnerable(&self) -> bool {
        self.audit_status == AuditStatus::Vulnerable
    }

    /// Returns the highest severity advisory
    pub fn max_severity(&self) -> Severity {
        self.advisories
            .iter()
            .map(|a| a.severity)
            .max()
            .unwrap_or(Severity::None)
    }
}

/// Summary of a full dependency audit
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditSummary {
    /// Total dependencies scanned
    pub total_dependencies: u32,

    /// Clean dependencies
    pub clean_count: u32,

    /// Dependencies with warnings
    pub warning_count: u32,

    /// Vulnerable dependencies
    pub vulnerable_count: u32,

    /// Individual audit results
    pub audits: Vec<DependencyAudit>,
}

impl AuditSummary {
    /// Create a summary from a list of audits
    pub fn from_audits(audits: Vec<DependencyAudit>) -> Self {
        let total_dependencies = audits.len() as u32;
        let clean_count = audits
            .iter()
            .filter(|a| a.audit_status == AuditStatus::Clean)
            .count() as u32;
        let warning_count = audits
            .iter()
            .filter(|a| a.audit_status == AuditStatus::Warning)
            .count() as u32;
        let vulnerable_count = audits
            .iter()
            .filter(|a| a.audit_status == AuditStatus::Vulnerable)
            .count() as u32;

        Self {
            total_dependencies,
            clean_count,
            warning_count,
            vulnerable_count,
            audits,
        }
    }

    /// Returns true if any vulnerabilities were found
    pub fn has_vulnerabilities(&self) -> bool {
        self.vulnerable_count > 0
    }

    /// Returns true if any issues (warnings or vulnerabilities) were found
    pub fn has_issues(&self) -> bool {
        self.warning_count > 0 || self.vulnerable_count > 0
    }

    /// Get all vulnerable dependencies
    pub fn vulnerable_deps(&self) -> Vec<&DependencyAudit> {
        self.audits.iter().filter(|a| a.is_vulnerable()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_from_str() {
        assert_eq!(Severity::parse("critical"), Severity::Critical);
        assert_eq!(Severity::parse("CRITICAL"), Severity::Critical);
        assert_eq!(Severity::parse("high"), Severity::High);
        assert_eq!(Severity::parse("medium"), Severity::Medium);
        assert_eq!(Severity::parse("low"), Severity::Low);
        assert_eq!(Severity::parse("unknown"), Severity::None);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
        assert!(Severity::Low > Severity::None);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Critical), "critical");
        assert_eq!(format!("{}", Severity::None), "none");
    }

    #[test]
    fn test_audit_status_is_failure() {
        assert!(AuditStatus::Vulnerable.is_failure());
        assert!(!AuditStatus::Warning.is_failure());
        assert!(!AuditStatus::Clean.is_failure());
    }

    #[test]
    fn test_audit_status_has_issues() {
        assert!(AuditStatus::Vulnerable.has_issues());
        assert!(AuditStatus::Warning.has_issues());
        assert!(!AuditStatus::Clean.has_issues());
    }

    #[test]
    fn test_advisory_new() {
        let advisory = Advisory::new("RUSTSEC-2021-0001", Severity::High, "Test vulnerability");

        assert_eq!(advisory.id, "RUSTSEC-2021-0001");
        assert_eq!(advisory.severity, Severity::High);
        assert_eq!(advisory.title, "Test vulnerability");
    }

    #[test]
    fn test_dependency_audit_clean() {
        let audit = DependencyAudit::clean("serde", "1.0.0", "MIT OR Apache-2.0");

        assert_eq!(audit.crate_name, "serde");
        assert_eq!(audit.version, "1.0.0");
        assert_eq!(audit.license, "MIT OR Apache-2.0");
        assert_eq!(audit.audit_status, AuditStatus::Clean);
        assert!(audit.advisories.is_empty());
        assert!(!audit.is_vulnerable());
    }

    #[test]
    fn test_dependency_audit_vulnerable() {
        let advisory = Advisory::new("RUSTSEC-2021-0001", Severity::Critical, "RCE vulnerability");
        let audit = DependencyAudit::vulnerable("unsafe-crate", "0.1.0", "MIT", vec![advisory]);

        assert_eq!(audit.crate_name, "unsafe-crate");
        assert_eq!(audit.audit_status, AuditStatus::Vulnerable);
        assert!(audit.is_vulnerable());
        assert_eq!(audit.advisories.len(), 1);
        assert_eq!(audit.max_severity(), Severity::Critical);
    }

    #[test]
    fn test_dependency_audit_max_severity() {
        let audit = DependencyAudit::vulnerable(
            "multi-vuln",
            "1.0.0",
            "MIT",
            vec![
                Advisory::new("A001", Severity::Low, "Low issue"),
                Advisory::new("A002", Severity::Critical, "Critical issue"),
                Advisory::new("A003", Severity::Medium, "Medium issue"),
            ],
        );

        assert_eq!(audit.max_severity(), Severity::Critical);
    }

    #[test]
    fn test_dependency_audit_max_severity_empty() {
        let audit = DependencyAudit::clean("safe-crate", "1.0.0", "MIT");
        assert_eq!(audit.max_severity(), Severity::None);
    }

    #[test]
    fn test_from_cargo_deny_output_empty() {
        let json = "";
        let audits = DependencyAudit::from_cargo_deny_output(json).unwrap();
        assert!(audits.is_empty());
    }

    #[test]
    fn test_from_cargo_deny_output_no_vulnerabilities() {
        let json = r#"{"type": "summary", "fields": {"total": 100}}"#;
        let audits = DependencyAudit::from_cargo_deny_output(json).unwrap();
        assert!(audits.is_empty());
    }

    #[test]
    fn test_from_cargo_deny_output_with_vulnerability() {
        let json = r#"{"type":"diagnostic","fields":{"graphs":[],"severity":"error","code":"A001","message":"Detected security vulnerability","labels":[{"span":{"crate":{"name":"vulnerable-crate","version":"0.1.0"}}}]}}"#;

        let audits = DependencyAudit::from_cargo_deny_output(json).unwrap();

        assert_eq!(audits.len(), 1);
        assert_eq!(audits[0].crate_name, "vulnerable-crate");
        assert_eq!(audits[0].version, "0.1.0");
        assert!(audits[0].is_vulnerable());
    }

    #[test]
    fn test_from_cargo_deny_output_invalid_json() {
        let json = "not valid json";
        let result = DependencyAudit::from_cargo_deny_output(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_audit_summary_from_audits() {
        let audits = vec![
            DependencyAudit::clean("serde", "1.0.0", "MIT"),
            DependencyAudit::clean("tokio", "1.0.0", "MIT"),
            DependencyAudit::vulnerable(
                "vuln-crate",
                "0.1.0",
                "MIT",
                vec![Advisory::new("A001", Severity::High, "Vuln")],
            ),
        ];

        let summary = AuditSummary::from_audits(audits);

        assert_eq!(summary.total_dependencies, 3);
        assert_eq!(summary.clean_count, 2);
        assert_eq!(summary.warning_count, 0);
        assert_eq!(summary.vulnerable_count, 1);
        assert!(summary.has_vulnerabilities());
        assert!(summary.has_issues());
    }

    #[test]
    fn test_audit_summary_all_clean() {
        let audits = vec![
            DependencyAudit::clean("serde", "1.0.0", "MIT"),
            DependencyAudit::clean("tokio", "1.0.0", "MIT"),
        ];

        let summary = AuditSummary::from_audits(audits);

        assert!(!summary.has_vulnerabilities());
        assert!(!summary.has_issues());
        assert!(summary.vulnerable_deps().is_empty());
    }

    #[test]
    fn test_audit_summary_vulnerable_deps() {
        let audits = vec![
            DependencyAudit::clean("serde", "1.0.0", "MIT"),
            DependencyAudit::vulnerable(
                "vuln1",
                "0.1.0",
                "MIT",
                vec![Advisory::new("A001", Severity::High, "Vuln")],
            ),
            DependencyAudit::vulnerable(
                "vuln2",
                "0.2.0",
                "MIT",
                vec![Advisory::new("A002", Severity::Critical, "Vuln")],
            ),
        ];

        let summary = AuditSummary::from_audits(audits);
        let vulnerable = summary.vulnerable_deps();

        assert_eq!(vulnerable.len(), 2);
        assert_eq!(vulnerable[0].crate_name, "vuln1");
        assert_eq!(vulnerable[1].crate_name, "vuln2");
    }

    #[test]
    fn test_dependency_audit_serialization() {
        let audit = DependencyAudit::vulnerable(
            "test-crate",
            "1.0.0",
            "MIT",
            vec![Advisory::new("A001", Severity::High, "Test")],
        );

        let json = serde_json::to_string(&audit).unwrap();
        let parsed: DependencyAudit = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.crate_name, audit.crate_name);
        assert_eq!(parsed.audit_status, audit.audit_status);
    }
}

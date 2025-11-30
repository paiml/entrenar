//! Nix Flake Configuration (ENT-018)
//!
//! Generates Nix flake configurations for reproducible sovereign deployments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Crate specification for Nix packaging
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrateSpec {
    /// Crate name
    pub name: String,
    /// Local path (if building from source)
    pub path: Option<PathBuf>,
    /// Crates.io version (if using published version)
    pub version: Option<String>,
    /// Git repository URL (if using git source)
    pub git: Option<String>,
    /// Git revision/tag
    pub rev: Option<String>,
}

impl CrateSpec {
    /// Create a crate spec for a local path
    pub fn local(name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            path: Some(path.into()),
            version: None,
            git: None,
            rev: None,
        }
    }

    /// Create a crate spec for crates.io version
    pub fn crates_io(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: None,
            version: Some(version.into()),
            git: None,
            rev: None,
        }
    }

    /// Create a crate spec for git source
    pub fn git(name: impl Into<String>, url: impl Into<String>, rev: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: None,
            version: None,
            git: Some(url.into()),
            rev: Some(rev.into()),
        }
    }

    /// Check if this is a local source
    pub fn is_local(&self) -> bool {
        self.path.is_some()
    }

    /// Check if this is a crates.io source
    pub fn is_crates_io(&self) -> bool {
        self.version.is_some() && self.git.is_none()
    }

    /// Check if this is a git source
    pub fn is_git(&self) -> bool {
        self.git.is_some()
    }

    /// Get source string for Nix
    pub fn nix_source(&self) -> String {
        if let Some(path) = &self.path {
            format!("./{}", path.display())
        } else if let Some(version) = &self.version {
            format!("crates.io:{version}")
        } else if let (Some(git), Some(rev)) = (&self.git, &self.rev) {
            format!("git:{git}?rev={rev}")
        } else {
            "unknown".to_string()
        }
    }
}

/// Target system for Nix builds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixSystem {
    /// x86_64 Linux
    X86_64Linux,
    /// aarch64 Linux
    Aarch64Linux,
    /// x86_64 macOS
    X86_64Darwin,
    /// aarch64 macOS (Apple Silicon)
    Aarch64Darwin,
}

impl NixSystem {
    /// Get the Nix system string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86_64Linux => "x86_64-linux",
            Self::Aarch64Linux => "aarch64-linux",
            Self::X86_64Darwin => "x86_64-darwin",
            Self::Aarch64Darwin => "aarch64-darwin",
        }
    }

    /// Get all supported systems
    pub fn all() -> Vec<Self> {
        vec![
            Self::X86_64Linux,
            Self::Aarch64Linux,
            Self::X86_64Darwin,
            Self::Aarch64Darwin,
        ]
    }

    /// Get Linux systems only
    pub fn linux_only() -> Vec<Self> {
        vec![Self::X86_64Linux, Self::Aarch64Linux]
    }
}

impl std::fmt::Display for NixSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Nix flake configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixFlakeConfig {
    /// Crate specifications
    pub crates: Vec<CrateSpec>,
    /// Rust toolchain version
    pub rust_version: String,
    /// Features to enable per crate
    pub features: HashMap<String, Vec<String>>,
    /// Target systems
    pub systems: Vec<NixSystem>,
    /// Flake description
    pub description: String,
    /// Enable GPU support
    pub gpu_support: bool,
    /// Include dev shell
    pub include_dev_shell: bool,
    /// Include CI checks
    pub include_checks: bool,
}

impl NixFlakeConfig {
    /// Create a new flake config
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            crates: Vec::new(),
            rust_version: "1.75.0".to_string(),
            features: HashMap::new(),
            systems: NixSystem::all(),
            description: description.into(),
            gpu_support: false,
            include_dev_shell: true,
            include_checks: true,
        }
    }

    /// Create the sovereign stack configuration (all PAIML crates)
    pub fn sovereign_stack() -> Self {
        let mut config = Self::new("PAIML Sovereign ML Stack - Air-gapped deployment ready");

        // Add all PAIML crates
        config.crates = vec![
            CrateSpec::crates_io("trueno", "0.2"),
            CrateSpec::crates_io("aprender", "0.1"),
            CrateSpec::crates_io("renacer", "0.1"),
            CrateSpec::crates_io("entrenar", "0.2"),
            CrateSpec::crates_io("realizar", "0.1"),
        ];

        // Set features
        config
            .features
            .insert("trueno".to_string(), vec!["simd".to_string()]);
        config
            .features
            .insert("entrenar".to_string(), vec!["full".to_string()]);

        config.rust_version = "1.75.0".to_string();
        config.include_dev_shell = true;
        config.include_checks = true;

        config
    }

    /// Add a crate to the configuration
    pub fn add_crate(mut self, spec: CrateSpec) -> Self {
        self.crates.push(spec);
        self
    }

    /// Set features for a crate
    pub fn with_features(
        mut self,
        crate_name: impl Into<String>,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.features.insert(
            crate_name.into(),
            features.into_iter().map(Into::into).collect(),
        );
        self
    }

    /// Set Rust version
    pub fn with_rust_version(mut self, version: impl Into<String>) -> Self {
        self.rust_version = version.into();
        self
    }

    /// Set target systems
    pub fn with_systems(mut self, systems: Vec<NixSystem>) -> Self {
        self.systems = systems;
        self
    }

    /// Enable GPU support
    pub fn with_gpu_support(mut self, enabled: bool) -> Self {
        self.gpu_support = enabled;
        self
    }

    /// Generate the flake.nix content
    pub fn generate_flake_nix(&self) -> String {
        let systems_list: Vec<&str> = self.systems.iter().map(NixSystem::as_str).collect();
        let systems_str = systems_list
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(" ");

        let crate_names: Vec<&str> = self.crates.iter().map(|c| c.name.as_str()).collect();

        let mut flake = String::new();

        // Header
        flake.push_str(&format!(
            r#"# Nix Flake for PAIML Sovereign Stack
# {}
# Generated by entrenar sovereign deployment tooling
{{
  description = "{}";

  inputs = {{
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {{
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    }};
    crane = {{
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    }};
    flake-utils.url = "github:numtide/flake-utils";
  }};

  outputs = {{ self, nixpkgs, rust-overlay, crane, flake-utils, ... }}:
    flake-utils.lib.eachSystem [ {} ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {{
          inherit system overlays;
        }};

        rustToolchain = pkgs.rust-bin.stable."{}" .default.override {{
          extensions = [ "rust-src" "rust-analyzer" ];
        }};

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

"#,
            chrono::Utc::now().format("%Y-%m-%d"),
            self.description,
            systems_str,
            self.rust_version,
        ));

        // Build inputs
        flake.push_str("        buildInputs = with pkgs; [\n");
        flake.push_str("          openssl\n");
        flake.push_str("          pkg-config\n");
        if self.gpu_support {
            flake.push_str("          # GPU support\n");
            flake.push_str("          cudatoolkit\n");
            flake.push_str("          cudnn\n");
        }
        flake.push_str("        ];\n\n");

        // Common args
        flake.push_str(&format!(
            r"        commonArgs = {{
          src = craneLib.cleanCargoSource ./.;
          inherit buildInputs;
          nativeBuildInputs = with pkgs; [ pkg-config ];
        }};

        # Build dependencies first (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Main packages
{}",
            self.generate_package_definitions(&crate_names),
        ));

        // Outputs
        flake.push_str(&format!(
            r"
      in {{
        packages = {{
{}          default = {};
        }};
",
            self.generate_packages_attr(&crate_names),
            crate_names.first().unwrap_or(&"entrenar"),
        ));

        // Dev shell
        if self.include_dev_shell {
            flake.push_str(&format!(
                r"
        devShells.default = pkgs.mkShell {{
          inputsFrom = [ {} ];
          buildInputs = with pkgs; [
            rustToolchain
            rust-analyzer
            cargo-watch
            cargo-edit
            cargo-expand
          ];
        }};
",
                crate_names.first().unwrap_or(&"entrenar"),
            ));
        }

        // Checks
        if self.include_checks {
            flake.push_str(
                r#"
        checks = {
          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- -D warnings";
          });

          test = craneLib.cargoNextest (commonArgs // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
          });

          fmt = craneLib.cargoFmt {
            src = craneLib.cleanCargoSource ./.;
          };
        };
"#,
            );
        }

        // Close outputs
        flake.push_str("      });\n}\n");

        flake
    }

    /// Generate package definitions
    fn generate_package_definitions(&self, crate_names: &[&str]) -> String {
        use std::fmt::Write;
        let mut result = String::new();
        for name in crate_names {
            let features = self
                .features
                .get(*name)
                .map(|f| {
                    let feature_list = f.join(",");
                    format!(r#"cargoExtraArgs = "--features {feature_list}";"#)
                })
                .unwrap_or_default();

            let _ = writeln!(
                &mut result,
                "        {name} = craneLib.buildPackage (commonArgs // {{\n\
          inherit cargoArtifacts;\n\
          pname = \"{name}\";\n\
          {features}\n\
        }});\n"
            );
        }
        result
    }

    /// Generate packages attribute
    fn generate_packages_attr(&self, crate_names: &[&str]) -> String {
        use std::fmt::Write;
        let mut result = String::new();
        for name in crate_names {
            let _ = writeln!(&mut result, "          {name} = {name};");
        }
        result
    }

    /// Generate Cachix configuration
    pub fn generate_cachix_config(&self) -> String {
        format!(
            r#"# Cachix configuration for PAIML Sovereign Stack
# Push: cachix push paiml $(nix-build)
# Use:  cachix use paiml

{{
  "name": "paiml",
  "signing_key_path": "$HOME/.config/cachix/cachix.dhall",
  "binary_caches": [
    {{
      "url": "https://paiml.cachix.org",
      "public_signing_keys": ["paiml.cachix.org-1:..."]
    }}
  ],
  "crates": {:?},
  "rust_version": "{}",
  "generated": "{}"
}}
"#,
            self.crates.iter().map(|c| &c.name).collect::<Vec<_>>(),
            self.rust_version,
            chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
        )
    }

    /// Generate a minimal flake for a single crate
    pub fn minimal_flake(crate_name: &str, crate_path: &str) -> String {
        format!(
            r#"{{
  description = "{crate_name} - PAIML component";

  inputs = {{
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  }};

  outputs = {{ self, nixpkgs, rust-overlay, crane, flake-utils, ... }}:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {{
          inherit system;
          overlays = [ (import rust-overlay) ];
        }};
        craneLib = crane.mkLib pkgs;
      in {{
        packages.default = craneLib.buildPackage {{
          src = ./{crate_path};
          pname = "{crate_name}";
        }};

        devShells.default = pkgs.mkShell {{
          inputsFrom = [ self.packages.${{system}}.default ];
          buildInputs = with pkgs; [ rust-analyzer ];
        }};
      }});
}}
"#
        )
    }
}

impl Default for NixFlakeConfig {
    fn default() -> Self {
        Self::sovereign_stack()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_spec_local() {
        let spec = CrateSpec::local("test", "crates/test");

        assert_eq!(spec.name, "test");
        assert!(spec.is_local());
        assert!(!spec.is_crates_io());
        assert!(!spec.is_git());
    }

    #[test]
    fn test_crate_spec_crates_io() {
        let spec = CrateSpec::crates_io("test", "1.0.0");

        assert_eq!(spec.name, "test");
        assert!(!spec.is_local());
        assert!(spec.is_crates_io());
        assert!(!spec.is_git());
    }

    #[test]
    fn test_crate_spec_git() {
        let spec = CrateSpec::git("test", "https://github.com/test/repo", "abc123");

        assert_eq!(spec.name, "test");
        assert!(!spec.is_local());
        assert!(!spec.is_crates_io());
        assert!(spec.is_git());
    }

    #[test]
    fn test_crate_spec_nix_source() {
        assert!(CrateSpec::local("a", "path").nix_source().contains("path"));
        assert!(CrateSpec::crates_io("b", "1.0")
            .nix_source()
            .contains("1.0"));
        assert!(CrateSpec::git("c", "url", "rev")
            .nix_source()
            .contains("rev"));
    }

    #[test]
    fn test_nix_system_as_str() {
        assert_eq!(NixSystem::X86_64Linux.as_str(), "x86_64-linux");
        assert_eq!(NixSystem::Aarch64Linux.as_str(), "aarch64-linux");
        assert_eq!(NixSystem::X86_64Darwin.as_str(), "x86_64-darwin");
        assert_eq!(NixSystem::Aarch64Darwin.as_str(), "aarch64-darwin");
    }

    #[test]
    fn test_nix_system_all() {
        let all = NixSystem::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_nix_system_linux_only() {
        let linux = NixSystem::linux_only();
        assert_eq!(linux.len(), 2);
        assert!(linux.contains(&NixSystem::X86_64Linux));
        assert!(linux.contains(&NixSystem::Aarch64Linux));
    }

    #[test]
    fn test_nix_flake_config_new() {
        let config = NixFlakeConfig::new("Test flake");

        assert_eq!(config.description, "Test flake");
        assert!(config.crates.is_empty());
        assert_eq!(config.rust_version, "1.75.0");
    }

    #[test]
    fn test_nix_flake_config_sovereign_stack() {
        let config = NixFlakeConfig::sovereign_stack();

        assert!(!config.crates.is_empty());
        assert!(config.crates.iter().any(|c| c.name == "trueno"));
        assert!(config.crates.iter().any(|c| c.name == "entrenar"));
        assert!(config.features.contains_key("trueno"));
    }

    #[test]
    fn test_nix_flake_config_add_crate() {
        let config = NixFlakeConfig::new("Test").add_crate(CrateSpec::crates_io("test", "1.0"));

        assert_eq!(config.crates.len(), 1);
        assert_eq!(config.crates[0].name, "test");
    }

    #[test]
    fn test_nix_flake_config_with_features() {
        let config = NixFlakeConfig::new("Test").with_features("test", ["feat1", "feat2"]);

        let features = config.features.get("test").unwrap();
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_nix_flake_config_with_systems() {
        let config = NixFlakeConfig::new("Test").with_systems(NixSystem::linux_only());

        assert_eq!(config.systems.len(), 2);
    }

    #[test]
    fn test_nix_flake_config_generate_flake_nix() {
        let config = NixFlakeConfig::sovereign_stack();
        let flake = config.generate_flake_nix();

        assert!(flake.contains("description ="));
        assert!(flake.contains("nixpkgs.url"));
        assert!(flake.contains("rust-overlay"));
        assert!(flake.contains("crane"));
        assert!(flake.contains("packages"));
        assert!(flake.contains("devShells"));
    }

    #[test]
    fn test_nix_flake_config_generate_flake_nix_with_gpu() {
        let config = NixFlakeConfig::new("Test")
            .add_crate(CrateSpec::crates_io("test", "1.0"))
            .with_gpu_support(true);

        let flake = config.generate_flake_nix();
        assert!(flake.contains("cudatoolkit"));
    }

    #[test]
    fn test_nix_flake_config_generate_cachix_config() {
        let config = NixFlakeConfig::sovereign_stack();
        let cachix = config.generate_cachix_config();

        assert!(cachix.contains("paiml"));
        assert!(cachix.contains("binary_caches"));
    }

    #[test]
    fn test_nix_flake_config_minimal_flake() {
        let flake = NixFlakeConfig::minimal_flake("mylib", "crates/mylib");

        assert!(flake.contains("mylib"));
        assert!(flake.contains("crates/mylib"));
        assert!(flake.contains("packages.default"));
    }

    #[test]
    fn test_nix_flake_config_default() {
        let config = NixFlakeConfig::default();

        // Default should be sovereign_stack
        assert!(!config.crates.is_empty());
        assert!(config.description.contains("Sovereign"));
    }

    #[test]
    fn test_crate_spec_serialization() {
        let spec = CrateSpec::crates_io("test", "1.0.0");
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: CrateSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(spec.name, parsed.name);
        assert_eq!(spec.version, parsed.version);
    }

    #[test]
    fn test_nix_system_display() {
        assert_eq!(format!("{}", NixSystem::X86_64Linux), "x86_64-linux");
    }
}

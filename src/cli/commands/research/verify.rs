//! Research verify subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::VerifyArgs;
use crate::research::{PreRegistration, SignedPreRegistration, TimestampProof};

/// Result of signature verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureStatus {
    /// Signature is valid
    Valid,
    /// Signature is invalid
    Invalid,
    /// Verification error occurred
    Error(String),
}

/// Verify signature on a signed pre-registration
pub fn verify_signature(signed: &SignedPreRegistration) -> SignatureStatus {
    match signed.verify() {
        Ok(true) => SignatureStatus::Valid,
        Ok(false) => SignatureStatus::Invalid,
        Err(e) => SignatureStatus::Error(e.to_string()),
    }
}

/// Log git timestamp proof information
pub fn log_git_proof(level: LogLevel, proof: &TimestampProof) {
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
}

/// Verify signed pre-registration content
pub fn verify_signed_content(
    signed: &SignedPreRegistration,
    verify_git: bool,
    level: LogLevel,
) -> Result<(), String> {
    match verify_signature(signed) {
        SignatureStatus::Valid => {
            log(level, LogLevel::Normal, "Signature verification: VALID");
        }
        SignatureStatus::Invalid => {
            log(level, LogLevel::Normal, "Signature verification: INVALID");
            return Err("Signature verification failed".to_string());
        }
        SignatureStatus::Error(e) => {
            return Err(format!("Verification error: {e}"));
        }
    }

    if verify_git {
        if let Some(proof) = &signed.timestamp_proof {
            log_git_proof(level, proof);
        } else {
            log(level, LogLevel::Normal, "No git timestamp proof found");
        }
    }

    log(
        level,
        LogLevel::Normal,
        "Pre-registration verified successfully",
    );
    Ok(())
}

/// Compute and log commitment from pre-registration
pub fn compute_commitment(prereg: &PreRegistration, level: LogLevel) {
    let commitment = prereg.commit();
    log(
        level,
        LogLevel::Normal,
        &format!("Computed commitment: {}...", &commitment.hash[..32]),
    );
}

pub fn run_research_verify(args: VerifyArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Verifying: {}", args.file.display()),
    );

    let content =
        std::fs::read_to_string(&args.file).map_err(|e| format!("Failed to read file: {e}"))?;

    if let Ok(signed) = serde_yaml::from_str::<SignedPreRegistration>(&content) {
        return verify_signed_content(&signed, args.verify_git, level);
    }

    log(
        level,
        LogLevel::Normal,
        "File does not contain a signed pre-registration",
    );

    if let Some(original_path) = &args.original {
        let original_content = std::fs::read_to_string(original_path)
            .map_err(|e| format!("Failed to read original: {e}"))?;

        let prereg: PreRegistration = serde_yaml::from_str(&original_content)
            .map_err(|e| format!("Failed to parse original: {e}"))?;

        compute_commitment(&prereg, level);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_status_eq() {
        assert_eq!(SignatureStatus::Valid, SignatureStatus::Valid);
        assert_eq!(SignatureStatus::Invalid, SignatureStatus::Invalid);
        assert_ne!(SignatureStatus::Valid, SignatureStatus::Invalid);
    }

    #[test]
    fn test_signature_status_error() {
        let err = SignatureStatus::Error("test error".to_string());
        assert_eq!(err, SignatureStatus::Error("test error".to_string()));
        assert_ne!(err, SignatureStatus::Valid);
    }

    #[test]
    fn test_signature_status_debug() {
        let valid = SignatureStatus::Valid;
        assert!(format!("{:?}", valid).contains("Valid"));
    }

    #[test]
    fn test_signature_status_clone() {
        let orig = SignatureStatus::Error("test".to_string());
        let cloned = orig.clone();
        assert_eq!(orig, cloned);
    }
}

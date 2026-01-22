//! Research verify subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::VerifyArgs;
use crate::research::{PreRegistration, SignedPreRegistration};

pub fn run_research_verify(args: VerifyArgs, level: LogLevel) -> Result<(), String> {
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

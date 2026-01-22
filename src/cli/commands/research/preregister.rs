//! Research preregister subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::PreregisterArgs;
use crate::research::{PreRegistration, SignedPreRegistration, TimestampProof};

pub fn run_research_preregister(args: PreregisterArgs, level: LogLevel) -> Result<(), String> {
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

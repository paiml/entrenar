//! Init command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::InitArgs;
use crate::config::InitTemplate;
use crate::yaml_mode::{generate_yaml, Template};

pub fn run_init(args: InitArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Generating {} template for: {}", args.template, args.name),
    );

    // Convert CLI InitTemplate to yaml_mode Template
    let template = match args.template {
        InitTemplate::Minimal => Template::Minimal,
        InitTemplate::Lora => Template::Lora,
        InitTemplate::Qlora => Template::Qlora,
        InitTemplate::Full => Template::Full,
    };

    // Generate YAML manifest
    let yaml = generate_yaml(
        template,
        &args.name,
        args.model.as_deref(),
        args.data.as_deref(),
    );

    // Output to file or stdout
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &yaml).map_err(|e| format!("Failed to write file: {e}"))?;
        log(
            level,
            LogLevel::Normal,
            &format!("Manifest saved to: {}", output_path.display()),
        );
    } else {
        println!("{yaml}");
    }

    Ok(())
}

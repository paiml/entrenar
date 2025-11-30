//! Entrenar CLI
//!
//! Single-command training entry point for the entrenar library.
//!
//! # Usage
//!
//! ```bash
//! # Train from config
//! entrenar train config.yaml
//!
//! # Train with overrides
//! entrenar train config.yaml --epochs 10 --lr 0.001
//!
//! # Validate config
//! entrenar validate config.yaml
//!
//! # Show config info
//! entrenar info config.yaml
//!
//! # Quantize model
//! entrenar quantize model.gguf --output model_q4.gguf
//!
//! # Merge models
//! entrenar merge model1.gguf model2.gguf --output merged.gguf
//! ```

use clap::Parser;
use entrenar::config::{
    apply_overrides, load_config, train_from_yaml, validate_config, Cli, Command, OutputFormat,
    ResearchArgs, ResearchCommand,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Configure output based on verbose/quiet flags
    let log_level = if cli.quiet {
        LogLevel::Quiet
    } else if cli.verbose {
        LogLevel::Verbose
    } else {
        LogLevel::Normal
    };

    let result = match cli.command {
        Command::Train(args) => run_train(args, log_level),
        Command::Validate(args) => run_validate(args, log_level),
        Command::Info(args) => run_info(args, log_level),
        Command::Init(args) => run_init(args, log_level),
        Command::Quantize(args) => run_quantize(args, log_level),
        Command::Merge(args) => run_merge(args, log_level),
        Command::Research(args) => run_research(args, log_level),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum LogLevel {
    Quiet,
    Normal,
    Verbose,
}

fn log(level: LogLevel, required: LogLevel, msg: &str) {
    if level != LogLevel::Quiet && (level == required || required == LogLevel::Normal) {
        println!("{msg}");
    }
}

fn run_train(args: entrenar::config::TrainArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Entrenar: Training from {}", args.config.display()),
    );

    // Load and validate config
    let mut spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    // Apply command-line overrides
    apply_overrides(&mut spec, &args);

    if args.dry_run {
        log(
            level,
            LogLevel::Normal,
            "Dry run - config validated successfully",
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Model: {}", spec.model.path.display()),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Optimizer: {} (lr={})",
                spec.optimizer.name, spec.optimizer.lr
            ),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Epochs: {}", spec.training.epochs),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Batch size: {}", spec.data.batch_size),
        );
        return Ok(());
    }

    // Run training
    train_from_yaml(&args.config).map_err(|e| format!("Training error: {e}"))?;

    log(level, LogLevel::Normal, "Training complete!");
    Ok(())
}

fn run_validate(args: entrenar::config::ValidateArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Validating config: {}", args.config.display()),
    );

    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    validate_config(&spec).map_err(|e| format!("Validation failed: {e}"))?;

    log(level, LogLevel::Normal, "Configuration is valid");

    if args.detailed {
        println!();
        println!("Configuration Summary:");
        println!("  Model path: {}", spec.model.path.display());
        println!("  Target layers: {:?}", spec.model.layers);
        println!();
        println!("  Training data: {}", spec.data.train.display());
        if let Some(val) = &spec.data.val {
            println!("  Validation data: {}", val.display());
        }
        println!("  Batch size: {}", spec.data.batch_size);
        println!();
        println!("  Optimizer: {}", spec.optimizer.name);
        println!("  Learning rate: {}", spec.optimizer.lr);
        if let Some(wd) = spec.optimizer.params.get("weight_decay") {
            println!("  Weight decay: {wd}");
        }
        println!();
        println!("  Epochs: {}", spec.training.epochs);
        if let Some(clip) = spec.training.grad_clip {
            println!("  Gradient clipping: {clip}");
        }
        println!("  Output dir: {}", spec.training.output_dir.display());

        if let Some(lora) = &spec.lora {
            println!();
            println!("  LoRA:");
            println!("    Rank: {}", lora.rank);
            println!("    Alpha: {}", lora.alpha);
            if lora.dropout > 0.0 {
                println!("    Dropout: {}", lora.dropout);
            }
        }

        if let Some(quant) = &spec.quantize {
            println!();
            println!("  Quantization:");
            println!("    Bits: {}", quant.bits);
            println!("    Symmetric: {}", quant.symmetric);
        }

        if let Some(merge) = &spec.merge {
            println!();
            println!("  Merge:");
            println!("    Method: {}", merge.method);
            if let Some(weight) = merge.params.get("weight") {
                println!("    Weight: {weight}");
            }
        }
    }

    Ok(())
}

fn run_info(args: entrenar::config::InfoArgs, level: LogLevel) -> Result<(), String> {
    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    match args.format {
        OutputFormat::Text => {
            log(level, LogLevel::Normal, "Configuration Info:");
            println!();
            println!("Model: {}", spec.model.path.display());
            println!(
                "Optimizer: {} (lr={})",
                spec.optimizer.name, spec.optimizer.lr
            );
            println!("Epochs: {}", spec.training.epochs);
            println!("Batch size: {}", spec.data.batch_size);

            if spec.lora.is_some() {
                println!("LoRA: enabled");
            }
            if spec.quantize.is_some() {
                println!("Quantization: enabled");
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&spec)
                .map_err(|e| format!("JSON serialization error: {e}"))?;
            println!("{json}");
        }
        OutputFormat::Yaml => {
            let yaml = serde_yaml::to_string(&spec)
                .map_err(|e| format!("YAML serialization error: {e}"))?;
            println!("{yaml}");
        }
    }

    Ok(())
}

fn run_init(args: entrenar::config::InitArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::config::InitTemplate;
    use entrenar::yaml_mode::{generate_yaml, Template};

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

fn run_quantize(args: entrenar::config::QuantizeArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::config::QuantMethod;
    use entrenar::quant::{quantize_tensor, QuantGranularity, QuantMode, QuantizedTensor};
    use safetensors::SafeTensors;
    use std::collections::HashMap;

    log(
        level,
        LogLevel::Normal,
        &format!("Quantizing {} to {}-bit", args.model.display(), args.bits),
    );

    log(
        level,
        LogLevel::Verbose,
        &format!("  Method: {:?}", args.method),
    );
    log(
        level,
        LogLevel::Verbose,
        &format!("  Per-channel: {}", args.per_channel),
    );
    log(
        level,
        LogLevel::Verbose,
        &format!("  Output: {}", args.output.display()),
    );

    // Validate bit width
    if args.bits != 4 && args.bits != 8 {
        return Err(format!("Unsupported bit width: {}. Use 4 or 8.", args.bits));
    }

    // Load safetensors model
    let data = std::fs::read(&args.model).map_err(|e| format!("Failed to read model file: {e}"))?;

    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    // Convert CLI args to quant module types
    let mode = match args.method {
        QuantMethod::Symmetric => QuantMode::Symmetric,
        QuantMethod::Asymmetric => QuantMode::Asymmetric,
    };

    let granularity = if args.per_channel {
        QuantGranularity::PerChannel
    } else {
        QuantGranularity::PerTensor
    };

    // Quantize each tensor
    let mut quantized_tensors: HashMap<String, QuantizedTensor> = HashMap::new();
    let mut total_original_bytes = 0usize;
    let mut total_quantized_bytes = 0usize;

    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

        // Only quantize float tensors
        if tensor.dtype() != safetensors::tensor::Dtype::F32 {
            log(
                level,
                LogLevel::Verbose,
                &format!("  Skipping {name} (not F32)"),
            );
            continue;
        }

        let shape: Vec<usize> = tensor.shape().to_vec();
        let num_elements: usize = shape.iter().product();
        total_original_bytes += num_elements * 4; // 4 bytes per f32

        // Convert bytes to f32 values
        let bytes = tensor.data();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Quantize
        let quantized = quantize_tensor(&values, &shape, granularity, mode, args.bits);
        total_quantized_bytes += quantized.memory_bytes();

        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Quantized {}: {:?} -> {} bytes",
                name,
                shape,
                quantized.memory_bytes()
            ),
        );

        quantized_tensors.insert((*name).to_string(), quantized);
    }

    // Save quantized model as JSON
    // Note: Quantized tensors use custom block formats (Q4_0, Q8_0) that are not
    // directly compatible with SafeTensors. For SafeTensors output, use GGUF export
    // or dequantize first. JSON format preserves the quantization parameters.
    let output_data = serde_json::to_vec_pretty(&quantized_tensors)
        .map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &output_data)
        .map_err(|e| format!("Failed to write output: {e}"))?;

    let compression_ratio = if total_quantized_bytes > 0 {
        total_original_bytes as f64 / total_quantized_bytes as f64
    } else {
        1.0
    };

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Quantization complete: {} tensors, {:.1}x compression",
            quantized_tensors.len(),
            compression_ratio
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Output: {}", args.output.display()),
    );

    Ok(())
}

fn run_merge(args: entrenar::config::MergeArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::autograd::Tensor;
    use entrenar::config::MergeMethod;
    use entrenar::merge::{
        dare_merge, ensemble_merge, slerp_merge, ties_merge, DareConfig, EnsembleConfig, Model,
        SlerpConfig, TiesConfig,
    };
    use safetensors::SafeTensors;
    use std::collections::HashMap;

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Merging {} models using {:?}",
            args.models.len(),
            args.method
        ),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(
            level,
            LogLevel::Verbose,
            &format!("  Model {}: {}", i + 1, model.display()),
        );
    }
    log(
        level,
        LogLevel::Verbose,
        &format!("  Output: {}", args.output.display()),
    );

    // Validate we have enough models
    if args.models.len() < 2 {
        return Err("Need at least 2 models to merge".to_string());
    }

    // Load all models
    let mut models: Vec<Model> = Vec::new();
    for path in &args.models {
        let data =
            std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;

        let mut model: Model = HashMap::new();
        for name in tensors.names() {
            let tensor = tensors
                .tensor(name)
                .map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

            // Only process F32 tensors
            if tensor.dtype() != safetensors::tensor::Dtype::F32 {
                continue;
            }

            let bytes = tensor.data();
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            model.insert((*name).to_string(), Tensor::from_vec(values, false));
        }
        models.push(model);

        let tensor_count = models.last().map_or(0, HashMap::len);
        log(
            level,
            LogLevel::Verbose,
            &format!("  Loaded {} tensors from {}", tensor_count, path.display()),
        );
    }

    // Perform merge
    let merged = match args.method {
        MergeMethod::Ties => {
            let config = TiesConfig {
                density: args.density.unwrap_or(0.2),
            };
            // First model is base, rest are task-specific
            let base = &models[0];
            ties_merge(&models[1..], base, &config)
                .map_err(|e| format!("TIES merge failed: {e}"))?
        }
        MergeMethod::Dare => {
            let config = DareConfig {
                drop_prob: 1.0 - args.density.unwrap_or(0.5), // density -> drop_prob
                seed: None,
            };
            let base = &models[0];
            dare_merge(&models[1..], base, &config)
                .map_err(|e| format!("DARE merge failed: {e}"))?
        }
        MergeMethod::Slerp => {
            if models.len() != 2 {
                return Err("SLERP requires exactly 2 models".to_string());
            }
            let config = SlerpConfig {
                t: args.weight.unwrap_or(0.5),
            };
            slerp_merge(&models[0], &models[1], &config)
                .map_err(|e| format!("SLERP merge failed: {e}"))?
        }
        MergeMethod::Average => {
            // Parse weights if provided
            let config = if let Some(w_str) = &args.weights {
                let weights: Vec<f32> = w_str
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("Invalid weights: {e}"))?;
                EnsembleConfig::weighted_average(weights)
            } else {
                EnsembleConfig::uniform_average()
            };

            ensemble_merge(&models, &config).map_err(|e| format!("Average merge failed: {e}"))?
        }
    };

    // Determine output format from file extension
    let output_ext = args
        .output
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("json");

    if output_ext == "safetensors" {
        // Export to SafeTensors format (HuggingFace compatible)
        use safetensors::tensor::{Dtype, TensorView};

        // Collect tensor data with proper lifetime management
        let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = merged
            .iter()
            .map(|(name, tensor)| {
                let data = tensor.data();
                let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap()).to_vec();
                let shape = vec![tensor.len()];
                (name.clone(), bytes, shape)
            })
            .collect();

        // Create TensorViews
        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
                (name.as_str(), view)
            })
            .collect();

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), "merged-model".to_string());
        metadata.insert("merge_method".to_string(), format!("{:?}", args.method));
        metadata.insert("tensor_count".to_string(), merged.len().to_string());

        // Serialize
        let safetensor_bytes = safetensors::serialize(views, Some(metadata))
            .map_err(|e| format!("Failed to serialize SafeTensors: {e}"))?;

        std::fs::write(&args.output, safetensor_bytes)
            .map_err(|e| format!("Failed to write output: {e}"))?;
    } else {
        // Fall back to JSON for other formats
        let output_data: HashMap<String, Vec<f32>> = merged
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.data().to_vec()))
            .collect();

        let json_data = serde_json::to_vec_pretty(&output_data)
            .map_err(|e| format!("Failed to serialize: {e}"))?;

        std::fs::write(&args.output, &json_data)
            .map_err(|e| format!("Failed to write output: {e}"))?;
    }

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Merge complete: {} tensors written to {}",
            merged.len(),
            args.output.display()
        ),
    );

    Ok(())
}

fn run_research(args: ResearchArgs, level: LogLevel) -> Result<(), String> {
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

fn run_research_init(
    args: entrenar::config::ResearchInitArgs,
    level: LogLevel,
) -> Result<(), String> {
    use entrenar::config::{ArtifactTypeArg, LicenseArg};
    use entrenar::research::{Affiliation, ArtifactType, Author, License, ResearchArtifact};

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

fn run_research_preregister(
    args: entrenar::config::PreregisterArgs,
    level: LogLevel,
) -> Result<(), String> {
    use entrenar::research::{PreRegistration, SignedPreRegistration, TimestampProof};

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

fn run_research_cite(args: entrenar::config::CiteArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::config::CitationFormat;
    use entrenar::research::{CitationMetadata, ResearchArtifact};

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

fn run_research_export(args: entrenar::config::ExportArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::config::ExportFormat;
    use entrenar::research::{
        AnonymizationConfig, LiterateDocument, NotebookExporter, ResearchArtifact,
    };

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

fn run_research_deposit(
    args: entrenar::config::DepositArgs,
    level: LogLevel,
) -> Result<(), String> {
    use entrenar::config::ArchiveProviderArg;
    use entrenar::research::{ArchiveDeposit, ArchiveProvider, ResearchArtifact};

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

fn run_research_bundle(args: entrenar::config::BundleArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::research::{ResearchArtifact, RoCrate};

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

fn run_research_verify(args: entrenar::config::VerifyArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::research::{PreRegistration, SignedPreRegistration};

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

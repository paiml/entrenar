//! Command parsing and execution for the REPL.

use crate::state::{HistoryEntry, LoadedModel, ModelRole, SessionState};
use entrenar_common::{EntrenarError, Result};

/// A parsed command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Fetch a model from HuggingFace
    Fetch { model_id: String, role: ModelRole },
    /// Inspect a loaded model
    Inspect { target: InspectTarget },
    /// Estimate memory requirements
    Memory {
        batch_size: Option<u32>,
        seq_len: Option<usize>,
    },
    /// Set configuration values
    Set { key: String, value: String },
    /// Run distillation
    Distill { dry_run: bool },
    /// Export model to file
    Export { format: String, path: String },
    /// Show command history
    History,
    /// Show help
    Help { topic: Option<String> },
    /// Clear screen
    Clear,
    /// Quit the shell
    Quit,
    /// Unknown command
    Unknown { input: String },
}

/// Target for inspect command.
#[derive(Debug, Clone, PartialEq)]
pub enum InspectTarget {
    /// Inspect layer structure
    Layers,
    /// Inspect memory usage
    Memory,
    /// Inspect all info
    All,
    /// Inspect specific model by name
    Model(String),
}

/// Parse a command string into a Command.
pub fn parse(input: &str) -> Result<Command> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(Command::Unknown {
            input: String::new(),
        });
    }

    let parts: Vec<&str> = input.split_whitespace().collect();
    let cmd = parts[0].to_lowercase();
    let args = &parts[1..];

    match cmd.as_str() {
        "fetch" | "download" => parse_fetch(args),
        "inspect" | "show" => parse_inspect(args),
        "memory" | "mem" => parse_memory(args),
        "set" => parse_set(args),
        "distill" | "train" => parse_distill(args),
        "export" | "save" => parse_export(args),
        "history" | "hist" => Ok(Command::History),
        "help" | "?" => parse_help(args),
        "clear" | "cls" => Ok(Command::Clear),
        "quit" | "exit" | "q" => Ok(Command::Quit),
        _ => Ok(Command::Unknown {
            input: input.to_string(),
        }),
    }
}

fn parse_fetch(args: &[&str]) -> Result<Command> {
    if args.is_empty() {
        return Err(EntrenarError::ConfigValue {
            field: "model_id".into(),
            message: "No model ID provided".into(),
            suggestion: "Usage: fetch <model_id> [--teacher|--student]".into(),
        });
    }

    let model_id = args[0].to_string();
    let role = if args.contains(&"--teacher") {
        ModelRole::Teacher
    } else if args.contains(&"--student") {
        ModelRole::Student
    } else {
        ModelRole::None
    };

    Ok(Command::Fetch { model_id, role })
}

fn parse_inspect(args: &[&str]) -> Result<Command> {
    let target = if args.is_empty() {
        InspectTarget::All
    } else {
        match args[0].to_lowercase().as_str() {
            "layers" | "layer" => InspectTarget::Layers,
            "memory" | "mem" => InspectTarget::Memory,
            "all" => InspectTarget::All,
            name => InspectTarget::Model(name.to_string()),
        }
    };

    Ok(Command::Inspect { target })
}

fn parse_memory(args: &[&str]) -> Result<Command> {
    let mut batch_size = None;
    let mut seq_len = None;

    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--batch" | "-b" if i + 1 < args.len() => {
                batch_size = args[i + 1].parse().ok();
                i += 2;
            }
            "--seq" | "-s" if i + 1 < args.len() => {
                seq_len = args[i + 1].parse().ok();
                i += 2;
            }
            _ => i += 1,
        }
    }

    Ok(Command::Memory {
        batch_size,
        seq_len,
    })
}

fn parse_set(args: &[&str]) -> Result<Command> {
    if args.len() < 2 {
        return Err(EntrenarError::ConfigValue {
            field: "set".into(),
            message: "Not enough arguments".into(),
            suggestion: "Usage: set <key> <value>".into(),
        });
    }

    Ok(Command::Set {
        key: args[0].to_string(),
        value: args[1..].join(" "),
    })
}

fn parse_distill(args: &[&str]) -> Result<Command> {
    let dry_run = args.contains(&"--dry-run") || args.contains(&"-n");
    Ok(Command::Distill { dry_run })
}

fn parse_export(args: &[&str]) -> Result<Command> {
    if args.len() < 2 {
        return Err(EntrenarError::ConfigValue {
            field: "export".into(),
            message: "Not enough arguments".into(),
            suggestion: "Usage: export <format> <path>".into(),
        });
    }

    Ok(Command::Export {
        format: args[0].to_string(),
        path: args[1].to_string(),
    })
}

fn parse_help(args: &[&str]) -> Result<Command> {
    let topic = args.first().map(|s| s.to_string());
    Ok(Command::Help { topic })
}

/// Execute a command and update state.
pub fn execute(cmd: &Command, state: &mut SessionState) -> Result<String> {
    let start = std::time::Instant::now();

    let result = match cmd {
        Command::Fetch { model_id, role } => execute_fetch(model_id, *role, state),
        Command::Inspect { target } => execute_inspect(target, state),
        Command::Memory {
            batch_size,
            seq_len,
        } => execute_memory(*batch_size, *seq_len, state),
        Command::Set { key, value } => execute_set(key, value, state),
        Command::Distill { dry_run } => execute_distill(*dry_run, state),
        Command::Export { format, path } => execute_export(format, path, state),
        Command::History => execute_history(state),
        Command::Help { topic } => execute_help(topic.as_deref()),
        Command::Clear => Ok("".to_string()),
        Command::Quit => Ok("Goodbye!".to_string()),
        Command::Unknown { input } => {
            if input.is_empty() {
                Ok(String::new())
            } else {
                Err(EntrenarError::ConfigValue {
                    field: "command".into(),
                    message: format!("Unknown command: {}", input),
                    suggestion: "Type 'help' for available commands".into(),
                })
            }
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    let success = result.is_ok();

    // Record in history (except for help/history/clear/quit)
    if !matches!(
        cmd,
        Command::Help { .. }
            | Command::History
            | Command::Clear
            | Command::Quit
            | Command::Unknown { .. }
    ) {
        let cmd_str = format!("{:?}", cmd);
        state.add_to_history(HistoryEntry::new(cmd_str, duration_ms, success));
        state.record_command(duration_ms, success);
    }

    result
}

fn execute_fetch(model_id: &str, role: ModelRole, state: &mut SessionState) -> Result<String> {
    // Simulate model fetching
    let model = LoadedModel {
        id: model_id.to_string(),
        path: std::path::PathBuf::from(format!("/tmp/models/{}", model_id.replace('/', "_"))),
        architecture: detect_architecture(model_id),
        parameters: estimate_params(model_id),
        layers: estimate_layers(model_id),
        hidden_dim: 4096,
        role,
    };

    let name = if role == ModelRole::Teacher {
        "teacher"
    } else if role == ModelRole::Student {
        "student"
    } else {
        model_id.split('/').last().unwrap_or(model_id)
    };

    state.add_model(name.to_string(), model.clone());

    Ok(format!(
        "✓ Fetched {}\n  Architecture: {}\n  Parameters: {:.1}B\n  Layers: {}",
        model_id,
        model.architecture,
        model.parameters as f64 / 1e9,
        model.layers
    ))
}

fn execute_inspect(target: &InspectTarget, state: &SessionState) -> Result<String> {
    match target {
        InspectTarget::All => {
            if state.loaded_models().is_empty() {
                return Ok("No models loaded. Use 'fetch <model_id>' to load a model.".to_string());
            }

            let mut output = String::from("Loaded Models:\n");
            for (name, model) in state.loaded_models() {
                output.push_str(&format!(
                    "  {} ({}): {:.1}B params, {} layers\n",
                    name,
                    model.id,
                    model.parameters as f64 / 1e9,
                    model.layers
                ));
            }
            Ok(output)
        }
        InspectTarget::Layers => {
            let mut output = String::from("Layer Analysis:\n");
            for (name, model) in state.loaded_models() {
                output.push_str(&format!(
                    "  {}: {} layers, hidden_dim={}\n",
                    name, model.layers, model.hidden_dim
                ));
            }
            Ok(output)
        }
        InspectTarget::Memory => execute_memory(None, None, state),
        InspectTarget::Model(name) => {
            if let Some(model) = state.get_model(name) {
                Ok(format!(
                    "Model: {}\n  ID: {}\n  Path: {}\n  Architecture: {}\n  Parameters: {:.1}B\n  Layers: {}\n  Hidden Dim: {}",
                    name, model.id, model.path.display(), model.architecture,
                    model.parameters as f64 / 1e9, model.layers, model.hidden_dim
                ))
            } else {
                Err(EntrenarError::ModelNotFound { path: name.into() })
            }
        }
    }
}

fn execute_memory(
    batch_size: Option<u32>,
    seq_len: Option<usize>,
    state: &SessionState,
) -> Result<String> {
    let batch = batch_size.unwrap_or(state.preferences().default_batch_size);
    let seq = seq_len.unwrap_or(state.preferences().default_seq_len);

    let total_params: u64 = state.loaded_models().values().map(|m| m.parameters).sum();
    let model_mem = total_params * 2; // FP16
    let activation_mem = (batch as u64) * (seq as u64) * 4096 * 32 * 2;
    let total = model_mem + activation_mem;

    Ok(format!(
        "Memory Estimate (batch={}, seq={}):\n  Model: {:.1} GB\n  Activations: {:.1} GB\n  Total: {:.1} GB",
        batch, seq,
        model_mem as f64 / 1e9,
        activation_mem as f64 / 1e9,
        total as f64 / 1e9
    ))
}

fn execute_set(key: &str, value: &str, state: &mut SessionState) -> Result<String> {
    match key {
        "batch_size" | "batch" => {
            let v: u32 = value.parse().map_err(|_| EntrenarError::ConfigValue {
                field: "batch_size".into(),
                message: "Invalid number".into(),
                suggestion: "Use a positive integer".into(),
            })?;
            state.preferences_mut().default_batch_size = v;
            Ok(format!("Set batch_size = {}", v))
        }
        "seq_len" | "seq" => {
            let v: usize = value.parse().map_err(|_| EntrenarError::ConfigValue {
                field: "seq_len".into(),
                message: "Invalid number".into(),
                suggestion: "Use a positive integer".into(),
            })?;
            state.preferences_mut().default_seq_len = v;
            Ok(format!("Set seq_len = {}", v))
        }
        _ => Err(EntrenarError::ConfigValue {
            field: key.into(),
            message: "Unknown setting".into(),
            suggestion: "Available settings: batch_size, seq_len".into(),
        }),
    }
}

fn execute_distill(dry_run: bool, state: &SessionState) -> Result<String> {
    let teacher = state
        .loaded_models()
        .values()
        .find(|m| m.role == ModelRole::Teacher);
    let student = state
        .loaded_models()
        .values()
        .find(|m| m.role == ModelRole::Student);

    if teacher.is_none() {
        return Err(EntrenarError::ConfigValue {
            field: "teacher".into(),
            message: "No teacher model loaded".into(),
            suggestion: "Use 'fetch <model_id> --teacher' to load a teacher model".into(),
        });
    }

    if student.is_none() {
        return Err(EntrenarError::ConfigValue {
            field: "student".into(),
            message: "No student model loaded".into(),
            suggestion: "Use 'fetch <model_id> --student' to load a student model".into(),
        });
    }

    if dry_run {
        Ok(format!(
            "Dry run configuration:\n  Teacher: {} ({:.1}B)\n  Student: {} ({:.1}B)\n  Ready to train",
            teacher.unwrap().id, teacher.unwrap().parameters as f64 / 1e9,
            student.unwrap().id, student.unwrap().parameters as f64 / 1e9
        ))
    } else {
        Ok("Training started... (simulated)".to_string())
    }
}

fn execute_export(format: &str, path: &str, _state: &SessionState) -> Result<String> {
    Ok(format!("Exported to {} in {} format", path, format))
}

fn execute_history(state: &SessionState) -> Result<String> {
    if state.history().is_empty() {
        return Ok("No command history.".to_string());
    }

    let mut output = String::from("Command History:\n");
    for (i, entry) in state.history().iter().enumerate() {
        let status = if entry.success { "✓" } else { "✗" };
        output.push_str(&format!(
            "  {}. {} {} ({}ms)\n",
            i + 1,
            status,
            entry.command,
            entry.duration_ms
        ));
    }
    Ok(output)
}

fn execute_help(topic: Option<&str>) -> Result<String> {
    match topic {
        Some("fetch") => Ok(
            "fetch <model_id> [--teacher|--student]\n  Download a model from HuggingFace"
                .to_string(),
        ),
        Some("inspect") => {
            Ok("inspect [layers|memory|all|<model>]\n  Inspect loaded models".to_string())
        }
        Some("memory") => {
            Ok("memory [--batch <n>] [--seq <n>]\n  Estimate memory requirements".to_string())
        }
        Some("distill") => Ok("distill [--dry-run]\n  Run distillation training".to_string()),
        _ => Ok("Available commands:
  fetch <model>      Download model from HuggingFace
  inspect [target]   Inspect loaded models
  memory             Estimate memory requirements
  set <key> <value>  Configure settings
  distill            Run distillation
  export <fmt> <path> Export model
  history            Show command history
  help [topic]       Show help
  quit               Exit shell"
            .to_string()),
    }
}

fn detect_architecture(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("llama") {
        "llama".to_string()
    } else if lower.contains("bert") {
        "bert".to_string()
    } else if lower.contains("gpt") {
        "gpt".to_string()
    } else if lower.contains("mistral") {
        "mistral".to_string()
    } else {
        "unknown".to_string()
    }
}

fn estimate_params(model_id: &str) -> u64 {
    let lower = model_id.to_lowercase();
    if lower.contains("70b") {
        70_000_000_000
    } else if lower.contains("13b") {
        13_000_000_000
    } else if lower.contains("7b") {
        7_000_000_000
    } else if lower.contains("1.1b") || lower.contains("1b") {
        1_100_000_000
    } else if lower.contains("base") {
        350_000_000
    } else {
        1_000_000_000
    }
}

fn estimate_layers(model_id: &str) -> u32 {
    let lower = model_id.to_lowercase();
    if lower.contains("70b") {
        80
    } else if lower.contains("13b") {
        40
    } else if lower.contains("7b") {
        32
    } else if lower.contains("base") {
        12
    } else {
        24
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fetch() {
        let cmd = parse("fetch meta-llama/Llama-2-7b --teacher").unwrap();
        assert!(matches!(
            cmd,
            Command::Fetch {
                role: ModelRole::Teacher,
                ..
            }
        ));

        let cmd = parse("fetch TinyLlama/TinyLlama-1.1B --student").unwrap();
        assert!(matches!(
            cmd,
            Command::Fetch {
                role: ModelRole::Student,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_inspect() {
        assert!(matches!(
            parse("inspect").unwrap(),
            Command::Inspect {
                target: InspectTarget::All
            }
        ));
        assert!(matches!(
            parse("inspect layers").unwrap(),
            Command::Inspect {
                target: InspectTarget::Layers
            }
        ));
    }

    #[test]
    fn test_parse_memory() {
        let cmd = parse("memory --batch 64 --seq 1024").unwrap();
        if let Command::Memory {
            batch_size,
            seq_len,
        } = cmd
        {
            assert_eq!(batch_size, Some(64));
            assert_eq!(seq_len, Some(1024));
        } else {
            panic!("Expected Memory command");
        }
    }

    #[test]
    fn test_parse_quit_variants() {
        assert!(matches!(parse("quit").unwrap(), Command::Quit));
        assert!(matches!(parse("exit").unwrap(), Command::Quit));
        assert!(matches!(parse("q").unwrap(), Command::Quit));
    }

    #[test]
    fn test_execute_fetch() {
        let mut state = SessionState::new();
        let result = execute_fetch("meta-llama/Llama-2-7b", ModelRole::Teacher, &mut state);

        assert!(result.is_ok());
        assert!(state.get_model("teacher").is_some());
    }

    #[test]
    fn test_execute_set() {
        let mut state = SessionState::new();

        execute_set("batch_size", "64", &mut state).unwrap();
        assert_eq!(state.preferences().default_batch_size, 64);

        execute_set("seq_len", "1024", &mut state).unwrap();
        assert_eq!(state.preferences().default_seq_len, 1024);
    }

    #[test]
    fn test_unknown_command() {
        let cmd = parse("foobar").unwrap();
        assert!(matches!(cmd, Command::Unknown { .. }));
    }
}

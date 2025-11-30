# CLI Overview

Entrenar provides two command-line tools for training, research, and benchmarking:

- **`entrenar`** - Main CLI for training, model operations, and research workflows
- **`entrenar-bench`** - Specialized tool for distillation benchmarking and cost analysis

## Installation

Both tools are installed when you add entrenar to your project:

```bash
cargo install entrenar
```

## Main CLI Commands

```bash
entrenar <COMMAND> [OPTIONS]

Commands:
  train      Train a model from YAML configuration
  validate   Validate a configuration file without training
  info       Display information about a configuration
  quantize   Quantize a model
  merge      Merge multiple models
  research   Academic research artifacts and workflows
```

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress all output except errors |
| `--version` | Show version information |
| `--help` | Show help |

## Quick Examples

### Training

```bash
# Train from YAML config
entrenar train config.yaml

# Train with overrides
entrenar train config.yaml --epochs 10 --lr 0.001

# Dry run (validate only)
entrenar train config.yaml --dry-run
```

### Model Operations

```bash
# Quantize a model
entrenar quantize model.safetensors --output model_q4.json --bits 4

# Merge models with TIES
entrenar merge model1.safetensors model2.safetensors --output merged.safetensors --method ties

# Merge with SLERP
entrenar merge model1.safetensors model2.safetensors --output merged.safetensors --method slerp --weight 0.7
```

### Research Workflows

```bash
# Initialize a research artifact
entrenar research init --id my-dataset --title "My Dataset" --author "Alice Smith"

# Generate citation
entrenar research cite artifact.yaml --year 2024 --format bibtex

# Create RO-Crate package
entrenar research bundle artifact.yaml --output ./package --zip
```

## Benchmark CLI Commands

```bash
entrenar-bench <COMMAND> [OPTIONS]

Commands:
  temperature       Sweep temperature hyperparameter
  alpha             Sweep alpha hyperparameter
  compare           Compare distillation strategies
  ablation          Run ablation study
  cost-performance  Analyze cost vs performance trade-offs
  recommend         Recommend configurations based on constraints
```

### Quick Examples

```bash
# Temperature sweep
entrenar-bench temperature --start 1.0 --end 8.0 --step 0.5

# Compare strategies
entrenar-bench compare --strategies kd,progressive,attention

# Cost-performance analysis
entrenar-bench cost-performance --gpu a100-80gb

# Get recommendations
entrenar-bench recommend --max-cost 50 --min-accuracy 0.85
```

## Output Formats

Both CLIs support multiple output formats:

| Format | Option | Description |
|--------|--------|-------------|
| Text | `--format text` | Human-readable tables (default) |
| JSON | `--format json` | Machine-readable JSON |
| YAML | `--format yaml` | YAML format (main CLI only) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ZENODO_TOKEN` | API token for Zenodo deposits |
| `FIGSHARE_TOKEN` | API token for Figshare deposits |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (see stderr for details) |

## See Also

- [Research Commands](./research.md) - Academic research CLI reference
- [Benchmark Commands](./benchmark.md) - Benchmarking CLI reference
- [Declarative Training](../declarative/overview.md) - YAML configuration

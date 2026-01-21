//! Notebook Exporter for Jupyter bridge (ENT-024)
//!
//! Provides export to Jupyter notebook format (.ipynb) with
//! evcxr kernel support for Rust.

use crate::research::literate::LiterateDocument;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Cell type in a notebook
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellType {
    /// Executable code cell
    Code,
    /// Markdown documentation cell
    Markdown,
    /// Raw text cell
    Raw,
}

impl std::fmt::Display for CellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Code => write!(f, "code"),
            Self::Markdown => write!(f, "markdown"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

/// A single cell in a notebook
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// Type of cell
    pub cell_type: CellType,
    /// Source content (lines)
    pub source: Vec<String>,
    /// Cell outputs (for code cells)
    pub outputs: Vec<CellOutput>,
    /// Execution count (for code cells)
    pub execution_count: Option<u32>,
    /// Cell metadata
    pub metadata: CellMetadata,
}

impl Cell {
    /// Create a new code cell
    pub fn code(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Code,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Create a new markdown cell
    pub fn markdown(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Markdown,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Create a new raw cell
    pub fn raw(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Raw,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Add an output to the cell
    pub fn with_output(mut self, output: CellOutput) -> Self {
        self.outputs.push(output);
        self
    }

    /// Set execution count
    pub fn with_execution_count(mut self, count: u32) -> Self {
        self.execution_count = Some(count);
        self
    }

    /// Get the source as a single string
    pub fn source_text(&self) -> String {
        self.source.join("")
    }
}

/// Cell output
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellOutput {
    /// Output type
    pub output_type: String,
    /// Output data
    pub data: Option<serde_json::Value>,
    /// Text output
    pub text: Option<Vec<String>>,
}

impl CellOutput {
    /// Create a stream output (stdout/stderr)
    pub fn stream(_name: &str, text: impl Into<String>) -> Self {
        Self {
            output_type: "stream".to_string(),
            data: None,
            text: Some(split_source(text.into())),
        }
    }

    /// Create an execute result output
    pub fn execute_result(data: serde_json::Value) -> Self {
        Self {
            output_type: "execute_result".to_string(),
            data: Some(data),
            text: None,
        }
    }
}

/// Cell metadata
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellMetadata {
    /// Whether the cell is collapsed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collapsed: Option<bool>,
    /// Whether the cell is scrolled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scrolled: Option<bool>,
}

/// Jupyter kernel specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    /// Display name
    pub display_name: String,
    /// Language
    pub language: String,
    /// Kernel name
    pub name: String,
}

impl Default for KernelSpec {
    fn default() -> Self {
        Self::python3()
    }
}

impl KernelSpec {
    /// Python 3 kernel
    pub fn python3() -> Self {
        Self {
            display_name: "Python 3".to_string(),
            language: "python".to_string(),
            name: "python3".to_string(),
        }
    }

    /// evcxr Rust kernel
    pub fn evcxr() -> Self {
        Self {
            display_name: "Rust".to_string(),
            language: "rust".to_string(),
            name: "rust".to_string(),
        }
    }

    /// Julia kernel
    pub fn julia() -> Self {
        Self {
            display_name: "Julia 1.9".to_string(),
            language: "julia".to_string(),
            name: "julia-1.9".to_string(),
        }
    }
}

/// Notebook exporter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookExporter {
    /// Notebook cells
    pub cells: Vec<Cell>,
    /// Kernel specification
    pub kernel: KernelSpec,
    /// Notebook metadata
    pub metadata: NotebookMetadata,
}

impl Default for NotebookExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl NotebookExporter {
    /// Create a new notebook exporter
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            kernel: KernelSpec::python3(),
            metadata: NotebookMetadata::default(),
        }
    }

    /// Create with a specific kernel
    pub fn with_kernel(kernel: KernelSpec) -> Self {
        Self {
            cells: Vec::new(),
            kernel,
            metadata: NotebookMetadata::default(),
        }
    }

    /// Add a cell to the notebook
    pub fn add_cell(&mut self, cell: Cell) {
        self.cells.push(cell);
    }

    /// Add a code cell
    pub fn add_code(&mut self, source: impl Into<String>) {
        self.cells.push(Cell::code(source));
    }

    /// Add a markdown cell
    pub fn add_markdown(&mut self, source: impl Into<String>) {
        self.cells.push(Cell::markdown(source));
    }

    /// Create from a literate document
    pub fn from_literate(doc: &LiterateDocument) -> Self {
        let mut exporter = Self::new();

        // Determine kernel from document type
        if doc.is_typst() || doc.is_markdown() {
            // Extract code blocks and intersperse with markdown
            let content = doc.content();
            let blocks = doc.extract_code_blocks();

            // Detect primary language for kernel selection
            let primary_lang = blocks
                .iter()
                .find_map(|b| b.language.as_ref())
                .map(String::as_str);

            exporter.kernel = match primary_lang {
                Some("rust") => KernelSpec::evcxr(),
                Some("julia") => KernelSpec::julia(),
                _ => KernelSpec::python3(),
            };

            // Simple parsing: add everything before first code block as markdown
            // then alternate between code and markdown
            let mut last_end = 0;

            for block in &blocks {
                // Find the start of this code block in the content
                let block_pattern = format!("```{}", block.language.as_deref().unwrap_or(""));
                if let Some(start_pos) = content[last_end..].find(&block_pattern) {
                    let absolute_start = last_end + start_pos;

                    // Add markdown before this block
                    let markdown_content = &content[last_end..absolute_start];
                    let trimmed = markdown_content.trim();
                    if !trimmed.is_empty() {
                        exporter.add_markdown(trimmed);
                    }

                    // Add the code block
                    exporter.add_code(&block.content);

                    // Find the end of this code block
                    let code_end = content[absolute_start..]
                        .find("```\n")
                        .or_else(|| content[absolute_start..].find("```"))
                        .map_or(content.len(), |p| {
                            absolute_start
                                + p
                                + content[absolute_start + p..].find('\n').unwrap_or(3)
                                + 1
                        });

                    last_end = code_end.min(content.len());
                }
            }

            // Add remaining markdown after last code block
            if last_end < content.len() {
                let remaining = content[last_end..].trim();
                if !remaining.is_empty() {
                    exporter.add_markdown(remaining);
                }
            }
        } else {
            // Raw text: add as single markdown cell
            exporter.add_markdown(doc.content());
        }

        exporter
    }

    /// Export to Jupyter notebook JSON format
    pub fn to_ipynb(&self) -> String {
        let notebook = json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": self.kernel.display_name,
                    "language": self.kernel.language,
                    "name": self.kernel.name
                },
                "language_info": {
                    "name": self.kernel.language
                }
            },
            "cells": self.cells.iter().map(|cell| {
                let mut cell_json = json!({
                    "cell_type": cell.cell_type.to_string(),
                    "source": cell.source,
                    "metadata": cell.metadata
                });

                if cell.cell_type == CellType::Code {
                    cell_json["outputs"] = json!(cell.outputs.iter().map(|o| {
                        let mut out = json!({
                            "output_type": o.output_type
                        });
                        if let Some(data) = &o.data {
                            out["data"] = data.clone();
                        }
                        if let Some(text) = &o.text {
                            out["text"] = json!(text);
                            out["name"] = json!("stdout");
                        }
                        out
                    }).collect::<Vec<_>>());
                    cell_json["execution_count"] = json!(cell.execution_count);
                }

                cell_json
            }).collect::<Vec<_>>()
        });

        serde_json::to_string_pretty(&notebook).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get the number of cells
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Get code cells only
    pub fn code_cells(&self) -> Vec<&Cell> {
        self.cells
            .iter()
            .filter(|c| c.cell_type == CellType::Code)
            .collect()
    }

    /// Get markdown cells only
    pub fn markdown_cells(&self) -> Vec<&Cell> {
        self.cells
            .iter()
            .filter(|c| c.cell_type == CellType::Markdown)
            .collect()
    }
}

/// Notebook metadata
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookMetadata {
    /// Notebook title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Authors
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub authors: Vec<String>,
}

/// Split source into lines for Jupyter format
fn split_source(source: String) -> Vec<String> {
    source
        .lines()
        .map(|line| format!("{line}\n"))
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, mut line)| {
            // Remove trailing newline from last line
            if i == source.lines().count().saturating_sub(1) {
                line.pop();
            }
            line
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_notebook_export() {
        let mut exporter = NotebookExporter::new();
        exporter.add_markdown("# Hello World");
        exporter.add_code("print('Hello!')");

        let ipynb = exporter.to_ipynb();

        assert!(ipynb.contains("nbformat"));
        assert!(ipynb.contains("Hello World"));
        assert!(ipynb.contains("print('Hello!')"));
    }

    #[test]
    fn test_code_cells_extracted() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("x = 1");
        exporter.add_markdown("Some text");
        exporter.add_code("y = 2");

        let code_cells = exporter.code_cells();
        assert_eq!(code_cells.len(), 2);
    }

    #[test]
    fn test_markdown_cells_preserved() {
        let mut exporter = NotebookExporter::new();
        exporter.add_markdown("# Title");
        exporter.add_markdown("## Subtitle");
        exporter.add_code("code");

        let md_cells = exporter.markdown_cells();
        assert_eq!(md_cells.len(), 2);
    }

    #[test]
    fn test_evcxr_kernel_metadata() {
        let exporter = NotebookExporter::with_kernel(KernelSpec::evcxr());
        let ipynb = exporter.to_ipynb();

        assert!(ipynb.contains("\"language\": \"rust\""));
        assert!(ipynb.contains("\"name\": \"rust\""));
        assert!(ipynb.contains("\"display_name\": \"Rust\""));
    }

    #[test]
    fn test_from_literate_python() {
        let content = r"
# My Analysis

Here's some code:

```python
import numpy as np
x = np.array([1, 2, 3])
```

More explanation here.
";

        let doc = LiterateDocument::parse_markdown(content);
        let exporter = NotebookExporter::from_literate(&doc);

        assert!(exporter.cell_count() >= 2);
        assert_eq!(exporter.kernel.language, "python");
    }

    #[test]
    fn test_from_literate_rust() {
        let content = r#"
# Rust Example

```rust
fn main() {
    println!("Hello!");
}
```
"#;

        let doc = LiterateDocument::parse_markdown(content);
        let exporter = NotebookExporter::from_literate(&doc);

        assert_eq!(exporter.kernel.language, "rust");
        assert_eq!(exporter.kernel.name, "rust");
    }

    #[test]
    fn test_cell_with_output() {
        let cell = Cell::code("print('hello')")
            .with_output(CellOutput::stream("stdout", "hello\n"))
            .with_execution_count(1);

        assert_eq!(cell.outputs.len(), 1);
        assert_eq!(cell.execution_count, Some(1));
    }

    #[test]
    fn test_cell_source_text() {
        let cell = Cell::code("line1\nline2\nline3");
        let text = cell.source_text();

        assert!(text.contains("line1"));
        assert!(text.contains("line2"));
        assert!(text.contains("line3"));
    }

    #[test]
    fn test_cell_type_display() {
        assert_eq!(format!("{}", CellType::Code), "code");
        assert_eq!(format!("{}", CellType::Markdown), "markdown");
        assert_eq!(format!("{}", CellType::Raw), "raw");
    }

    #[test]
    fn test_kernel_specs() {
        let python = KernelSpec::python3();
        assert_eq!(python.language, "python");

        let rust = KernelSpec::evcxr();
        assert_eq!(rust.language, "rust");

        let julia = KernelSpec::julia();
        assert_eq!(julia.language, "julia");
    }

    #[test]
    fn test_raw_cell() {
        let cell = Cell::raw("raw content");
        assert_eq!(cell.cell_type, CellType::Raw);
    }

    #[test]
    fn test_notebook_json_structure() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("x = 1");

        let ipynb = exporter.to_ipynb();
        let parsed: serde_json::Value = serde_json::from_str(&ipynb).unwrap();

        assert_eq!(parsed["nbformat"], 4);
        assert_eq!(parsed["nbformat_minor"], 5);
        assert!(parsed["metadata"]["kernelspec"].is_object());
        assert!(parsed["cells"].is_array());
    }

    #[test]
    fn test_empty_notebook() {
        let exporter = NotebookExporter::new();
        let ipynb = exporter.to_ipynb();

        let parsed: serde_json::Value = serde_json::from_str(&ipynb).unwrap();
        assert!(parsed["cells"].as_array().unwrap().is_empty());
    }
}

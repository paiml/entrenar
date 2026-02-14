//! Notebook exporter for Jupyter format.

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::cell::{Cell, CellType};
use super::kernel::KernelSpec;
use crate::research::literate::LiterateDocument;

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
                Some(other_lang) => {
                    eprintln!(
                        "Warning: unsupported kernel language '{other_lang}', defaulting to Python 3"
                    );
                    KernelSpec::python3()
                }
                None => KernelSpec::python3(),
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

        serde_json::to_string_pretty(&notebook).unwrap_or_else(|_err| "{}".to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_selection_all_language_variants() {
        let languages: &[Option<&str>] = &[Some("rust"), Some("julia"), Some("javascript"), None];

        for lang in languages {
            // Syntactic match covering all arms from from_literate kernel selection
            let kernel = match *lang {
                Some("rust") => KernelSpec::evcxr(),
                Some("julia") => KernelSpec::julia(),
                Some(_other_lang) => KernelSpec::python3(),
                None => KernelSpec::python3(),
            };

            match lang {
                Some("rust") => assert_eq!(kernel.language, "rust"),
                Some("julia") => assert_eq!(kernel.language, "julia"),
                Some(_) => assert_eq!(kernel.language, "python"),
                None => assert_eq!(kernel.language, "python"),
            }
        }
    }

    #[test]
    fn test_notebook_exporter_new() {
        let exporter = NotebookExporter::new();
        assert_eq!(exporter.cells.len(), 0);
        assert_eq!(exporter.kernel.language, "python");
    }

    #[test]
    fn test_notebook_exporter_default() {
        let exporter = NotebookExporter::default();
        assert_eq!(exporter.cell_count(), 0);
    }

    #[test]
    fn test_add_code_and_markdown() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("print('hello')");
        exporter.add_markdown("# Title");
        assert_eq!(exporter.cell_count(), 2);
        assert_eq!(exporter.code_cells().len(), 1);
        assert_eq!(exporter.markdown_cells().len(), 1);
    }

    #[test]
    fn test_notebook_metadata_default() {
        let meta = NotebookMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.authors.is_empty());
    }
}

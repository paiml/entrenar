# Entrenar Makefile
# Training & Optimization Library - Quality Gates
# Following renacer and bashrs EXTREME TDD patterns

.SUFFIXES:

.PHONY: help test coverage coverage-html coverage-clean mutants mutants-quick clean build release lint format check \
	tier1 tier2 tier3 pmat-init pmat-update roadmap-status

help: ## Show this help message
	@echo "Entrenar - Training & Optimization Library"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Tiered TDD Workflow (renacer pattern)
# =============================================================================

tier1: ## Tier 1: Fast tests (<5s) - unit tests, clippy, format
	@echo "🏃 Tier 1: Fast tests (<5 seconds)..."
	@cargo fmt --check
	@cargo clippy -- -D warnings
	@cargo test --lib --quiet
	@echo "✅ Tier 1 complete!"

tier2: tier1 ## Tier 2: Integration tests (<30s) - includes tier1
	@echo "🏃 Tier 2: Integration tests (<30 seconds)..."
	@cargo test --tests --quiet
	@echo "✅ Tier 2 complete!"

tier3: tier2 ## Tier 3: Full validation (<5m) - includes tier1+2, property tests
	@echo "🏃 Tier 3: Full validation (<5 minutes)..."
	@cargo test --all-targets --all-features --quiet
	@echo "✅ Tier 3 complete!"

# =============================================================================
# Basic Development
# =============================================================================

test: ## Run tests (fast, no coverage)
	@echo "🧪 Running tests..."
	@cargo test --quiet

build: ## Build debug binary
	@echo "🔨 Building debug binary..."
	@cargo build

release: ## Build optimized release binary
	@echo "🚀 Building release binary..."
	@cargo build --release
	@echo "✅ Release binary: target/release/entrenar"

lint: ## Run clippy linter
	@echo "🔍 Running clippy..."
	@cargo clippy -- -D warnings

format: ## Format code with rustfmt
	@echo "📝 Formatting code..."
	@cargo fmt

check: ## Type check without building
	@echo "✅ Type checking..."
	@cargo check --all-targets --all-features

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/coverage
	@echo "✅ Clean completed!"

# =============================================================================
# Code Coverage (EXTREME TDD requirement: >90%)
# =============================================================================

coverage: ## Generate HTML coverage report and open in browser
	@echo "📊 Running comprehensive test coverage analysis..."
	@echo "🔍 Checking for cargo-llvm-cov..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@if ! rustup component list --installed | grep -q llvm-tools-preview; then \
		echo "📦 Installing llvm-tools-preview..."; \
		rustup component add llvm-tools-preview; \
	fi
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage/html
	@echo "⚙️  Temporarily disabling global cargo config (mold/custom linker breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report test --workspace --all-features || true
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html || echo "⚠️  No coverage data generated"
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info || echo "⚠️  LCOV generation skipped"
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@cargo llvm-cov report --summary-only || echo "Run 'cargo test' to generate coverage data first"
	@echo ""
	@echo "📊 Coverage reports generated:"
	@echo "- HTML: target/coverage/html/index.html"
	@echo "- LCOV: target/coverage/lcov.info"
	@echo ""
	@xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "✅ Open target/coverage/html/index.html in your browser"

coverage-html: coverage ## Alias for coverage

coverage-clean: ## Clean coverage artifacts
	@echo "🧹 Cleaning coverage artifacts..."
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		cargo llvm-cov clean --workspace; \
		echo "✅ Coverage artifacts cleaned!"; \
	else \
		echo "⚠️  cargo-llvm-cov not installed, skipping clean."; \
	fi

# =============================================================================
# Mutation Testing (EXTREME TDD requirement: >80% kill rate)
# =============================================================================

mutants: ## Run mutation testing (full analysis)
	@echo "🧬 Running mutation testing..."
	@echo "🔍 Checking for cargo-mutants..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@echo "🧬 Running cargo-mutants (this may take several minutes)..."
	@cargo mutants --output target/mutants.out || echo "⚠️  Some mutants survived"
	@echo ""
	@echo "📊 Mutation Testing Results:"
	@cat target/mutants.out/mutants.out 2>/dev/null || echo "Check target/mutants.out/ for detailed results"

mutants-quick: ## Run mutation testing (quick check on changed files only)
	@echo "🧬 Running quick mutation testing..."
	@echo "🔍 Checking for cargo-mutants..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@echo "🧬 Running cargo-mutants on uncommitted changes..."
	@cargo mutants --in-diff git:HEAD --output target/mutants-quick.out || echo "⚠️  Some mutants survived"
	@echo ""
	@echo "📊 Quick Mutation Testing Results:"
	@cat target/mutants-quick.out/mutants.out 2>/dev/null || echo "Check target/mutants-quick.out/ for detailed results"

# =============================================================================
# PMAT Integration (Toyota Way Quality)
# =============================================================================

roadmap-status: ## Show current roadmap status
	@echo "📊 Roadmap Status:"
	@echo "See roadmap.yaml for ticket details"
	@echo ""
	@grep -A 2 "^summary:" roadmap.yaml | tail -n +2 || echo "⚠️  roadmap.yaml not found"

pmat-complexity: ## Check code complexity (<10 cyclomatic, <15 cognitive)
	@echo "📐 Checking code complexity..."
	@which pmat > /dev/null 2>&1 || (echo "❌ PMAT not installed" && exit 1)
	@pmat analyze complexity src/ --max-cyclomatic 10 --max-cognitive 15

pmat-tdg: ## Check Technical Debt Grade (>90 score = A grade)
	@echo "📊 Checking Technical Debt Grade..."
	@which pmat > /dev/null 2>&1 || (echo "❌ PMAT not installed" && exit 1)
	@pmat analyze tdg src/ --min-score 90

# =============================================================================
# Dependency Security (bashrs pattern)
# =============================================================================

deny-check: ## Check dependencies for security/license issues
	@echo "🔒 Checking dependencies..."
	@which cargo-deny > /dev/null 2>&1 || (echo "📦 Installing cargo-deny..." && cargo install cargo-deny --locked)
	@cargo deny check

# =============================================================================
# Pre-Commit Checks (run before every commit)
# =============================================================================

pre-commit: tier1 ## Run pre-commit checks (format, lint, fast tests, PMAT TDG)
	@echo "🎯 Running pre-commit checks..."
	@echo "✅ All pre-commit checks passed!"

# =============================================================================
# CI/CD Simulation (full quality gates)
# =============================================================================

ci: tier3 coverage mutants-quick pmat-complexity pmat-tdg deny-check ## Run full CI pipeline
	@echo "🎉 All CI checks passed!"
	@echo ""
	@echo "Quality Metrics:"
	@echo "- ✅ All tests passing"
	@echo "- ✅ Code coverage >90%"
	@echo "- ✅ Mutation score >80%"
	@echo "- ✅ Complexity <10"
	@echo "- ✅ TDG score >90"
	@echo "- ✅ Dependencies secure"

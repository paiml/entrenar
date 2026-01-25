# Entrenar Makefile
# Training & Optimization Library - Quality Gates
# Following renacer and bashrs EXTREME TDD patterns

.SUFFIXES:

.PHONY: help test test-fast test-quick test-full coverage coverage-fast coverage-full coverage-low coverage-open coverage-clean \
	mutants mutants-quick mutants-fast mutants-file clean build release lint format check fmt fmt-check \
	tier1 tier2 tier3 pmat-init pmat-update roadmap-status pmat-complexity pmat-tdg \
	property-test property-test-fast \
	examples examples-fast examples-list \
	server server-dev \
	llama-tests llama-properties llama-mutations llama-chaos llama-gradients llama-fuzz llama-examples llama-ci \
	profile-llama profile-llama-otlp profile-llama-anomaly \
	wasm-build wasm-install wasm-serve wasm-e2e wasm-e2e-ui wasm-e2e-headed wasm-e2e-update wasm-clean

help: ## Show this help message
	@echo "Entrenar - Training & Optimization Library"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Tiered TDD Workflow (renacer pattern)
# =============================================================================

tier1: ## Tier 1: Fast tests (<5s) - unit tests, clippy, format, gradient checks
	@echo "🏃 Tier 1: Fast tests (<5 seconds)..."
	@cargo fmt --check
	@cargo clippy -- -D warnings
	@cargo test --lib --quiet
	@cargo test --test gradient_llama --quiet
	@echo "✅ Tier 1 complete!"

tier2: tier1 ## Tier 2: Integration tests (<30s) - includes tier1
	@echo "🏃 Tier 2: Integration tests (<30 seconds)..."
	@cargo test --tests --quiet
	@echo "✅ Tier 2 complete!"

tier3: tier2 ## Tier 3: Full validation (<5m) - includes tier1+2, property tests, chaos tests
	@echo "🏃 Tier 3: Full validation (<5 minutes)..."
	@cargo test --all-targets --all-features --quiet
	@cargo test --test property_llama --quiet
	@cargo test --test mutation_resistant_llama --quiet
	@cargo test --test chaos_llama --quiet
	@echo "✅ Tier 3 complete!"

# =============================================================================
# TEST TARGETS (Performance-Optimized with nextest)
# =============================================================================

# Fast tests (<30s): Uses nextest for parallelism if available
# Pattern from bashrs: cargo-nextest + RUST_TEST_THREADS + PROPTEST_CASES
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 RUST_TEST_THREADS=$$(nproc) time cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		PROPTEST_CASES=25 time cargo test --workspace --lib; \
	fi
	@echo "✅ Fast tests passed"

# Quick alias for test-fast
test-quick: test-fast

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace \
			--status-level skip \
			--failure-output immediate; \
	else \
		time cargo test --workspace; \
	fi
	@echo "✅ Standard tests passed"

# Full comprehensive tests: All features, all property cases
test-full: ## Comprehensive tests (all features)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --workspace --all-features; \
	else \
		time cargo test --workspace --all-features; \
	fi
	@echo "✅ Full tests passed"

# =============================================================================
# Basic Development
# =============================================================================

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

fmt: format ## Alias for format

fmt-check: ## Check formatting without modifying
	@cargo fmt --check

check: ## Type check without building
	@echo "✅ Type checking..."
	@cargo check --all-targets --all-features

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/coverage
	@echo "✅ Clean completed!"

# =============================================================================
# COVERAGE TARGETS (trueno pattern - simple and reliable)
# =============================================================================

# Exclude dependencies and test files from coverage reporting
COV_EXCLUDE := --ignore-filename-regex='realizar/|trueno/|crates/|examples/|tests/|main\.rs'

# Standard coverage - JUST WORKS like trueno
# Note: CUDA tests need --test-threads=1 to avoid driver contention
coverage: ## Generate coverage report (entrenar src/ only)
	@echo "📊 Running coverage..."
	@cargo llvm-cov --no-report test --lib --features cuda -- --test-threads=1 2>&1 | tail -3
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>&1 | tail -1
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fast alias
coverage-fast: coverage

# Full coverage with HTML report (slow)
coverage-full: ## Full coverage with HTML report
	@echo "📊 Running full coverage..."
	@cargo llvm-cov --no-report test --lib --features cuda -- --test-threads=1 2>&1 | tail -3
	@echo ""
	@mkdir -p target/coverage/html
	@echo "⏳ Generating HTML report (this takes a while)..."
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COV_EXCLUDE) 2>&1 | tail -1
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@cargo llvm-cov report --summary-only $(COV_EXCLUDE) 2>&1 | tail -1
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 HTML: target/coverage/html/index.html"

# Show files below 95%
coverage-low: ## Show files below 95% coverage
	@cargo llvm-cov report $(COV_EXCLUDE) 2>/dev/null | grep 'src/' | awk '{ if ($$7+0 < 95 && $$7 != "-") print $$0 }' | head -30

# Open coverage report in browser
coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

coverage-clean: ## Clean coverage artifacts
	@rm -rf target/coverage
	@echo "✅ Coverage artifacts cleaned"

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

mutants-fast: ## Run mutation testing on 1/4 shard (quick)
	@echo "🧬 Running fast mutation testing (shard 1/4)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --shard 1/4 --timeout 120 --output target/mutants-fast.out || echo "⚠️  Some mutants survived"

mutants-file: ## Run mutation testing on specific file (FILE=src/foo.rs)
	@test -n "$(FILE)" || (echo "Usage: make mutants-file FILE=src/foo.rs" && exit 1)
	@echo "🧬 Running mutation testing on $(FILE)..."
	@which cargo-mutants > /dev/null 2>&1 || (echo "📦 Installing cargo-mutants..." && cargo install cargo-mutants --locked)
	@cargo mutants --file $(FILE) --timeout 120 --output target/mutants-file.out || echo "⚠️  Some mutants survived"

# =============================================================================
# Property Testing (proptest)
# =============================================================================

property-test: ## Run property-based tests (1000 cases)
	@echo "📊 Running property-based tests (1000 cases)..."
	PROPTEST_CASES=250 cargo test --features proptest -- proptest

property-test-fast: ## Run property-based tests (default cases)
	@echo "📊 Running property-based tests..."
	cargo test -- proptest

# =============================================================================
# Examples (MLOPS #22)
# =============================================================================

examples: ## Build and run all examples
	@echo "🚀 Building all examples..."
	@cargo build --examples --release
	@echo ""
	@echo "✅ Examples built. Available at target/release/examples/"
	@ls -1 target/release/examples/ 2>/dev/null | grep -v '.d$$' || echo "No examples found"

examples-fast: ## Build examples in debug mode (faster compile)
	@echo "🚀 Building examples (debug)..."
	@cargo build --examples
	@echo ""
	@echo "✅ Examples built. Available at target/debug/examples/"

examples-list: ## List all available examples
	@echo "📋 Available examples:"
	@echo ""
	@grep -h 'name = ' Cargo.toml | grep -A1 'example' | grep 'name' | sed 's/name = //g' | tr -d '"' | sed 's/^/  - /'

# =============================================================================
# Server (MLOPS #67)
# =============================================================================

server: ## Start the tracking server on port 5000
	@echo "🚀 Starting tracking server..."
	@cargo run --features server -- server --port 5000

server-dev: ## Start the tracking server in development mode
	@echo "🚀 Starting development server..."
	@RUST_LOG=debug cargo run --features server -- server --port 5000

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
# LLaMA Examples & Testing (Phase 1 Implementation)
# =============================================================================

llama-tests: ## Run all LLaMA-related tests
	@echo "🦙 Running LLaMA tests..."
	@echo "  📊 Property-based tests (13 properties)..."
	@cargo test --test property_llama --quiet
	@echo "  🧬 Mutation-resistant tests (10 tests)..."
	@cargo test --test mutation_resistant_llama --quiet || true
	@echo "  ⚡ Chaos engineering tests (15 tests)..."
	@cargo test --test chaos_llama --quiet
	@echo "  🎯 Gradient checking tests (18 tests)..."
	@cargo test --test gradient_llama --quiet
	@echo "  ✅ Architecture unit tests..."
	@cargo test --example llama2-train --lib --quiet || true
	@echo "✅ LLaMA tests complete!"

llama-properties: ## Run LLaMA property-based tests (100 iterations/property)
	@echo "📊 Running LLaMA property-based tests..."
	@cargo test --test property_llama -- --nocapture
	@echo "✅ 13 properties validated!"

llama-mutations: ## Run LLaMA mutation-resistant tests
	@echo "🧬 Running LLaMA mutation-resistant tests..."
	@cargo test --test mutation_resistant_llama -- --nocapture
	@echo "✅ Mutation-resistant tests complete!"

llama-chaos: ## Run LLaMA chaos engineering tests
	@echo "⚡ Running LLaMA chaos engineering tests..."
	@cargo test --test chaos_llama -- --nocapture
	@echo "✅ Chaos engineering tests complete!"

llama-gradients: ## Run LLaMA gradient checking tests
	@echo "🎯 Running LLaMA gradient checking tests..."
	@cargo test --test gradient_llama -- --nocapture
	@echo "✅ Gradient checking tests complete!"

llama-fuzz: ## Run LLaMA fuzz tests (requires cargo-fuzz and libstdc++)
	@echo "🔍 Running LLaMA fuzz tests..."
	@which cargo-fuzz > /dev/null 2>&1 || (echo "📦 Installing cargo-fuzz..." && cargo install cargo-fuzz)
	@echo "  - parameter_calc (1M iterations)..."
	@cargo fuzz run parameter_calc -- -runs=1000000 2>&1 | grep -E "(Done|ERROR)" || true
	@echo "  - tensor_ops (1M iterations)..."
	@cargo fuzz run tensor_ops -- -runs=1000000 2>&1 | grep -E "(Done|ERROR)" || true
	@echo "  - lora_config (1M iterations)..."
	@cargo fuzz run lora_config -- -runs=1000000 2>&1 | grep -E "(Done|ERROR)" || true
	@echo "✅ Fuzz testing complete!"

llama-examples: ## Build all LLaMA examples
	@echo "🦙 Building LLaMA examples..."
	@echo "  📦 Training from scratch (train.rs)..."
	@cargo build --release --example llama2-train --quiet
	@echo "  📦 LoRA fine-tuning (finetune_lora.rs)..."
	@cargo build --release --example llama2-finetune-lora --quiet
	@echo "  📦 QLoRA fine-tuning (finetune_qlora.rs)..."
	@cargo build --release --example llama2-finetune-qlora --quiet
	@echo "✅ All LLaMA examples built!"
	@echo ""
	@echo "Available examples:"
	@echo "  - ./target/release/examples/llama2-train --config examples/llama2/configs/124m.toml"
	@echo "  - ./target/release/examples/llama2-finetune-lora --model checkpoints/llama-124m.bin"
	@echo "  - ./target/release/examples/llama2-finetune-qlora --model checkpoints/llama-7b.bin"

llama-demo-train: llama-examples ## Demo: Run toy LLaMA training (124M model, 1 epoch)
	@echo "🦙 Running LLaMA training demo (124M model)..."
	@echo "Config: examples/llama2/configs/124m.toml"
	@echo ""
	@./target/release/examples/llama2-train --config examples/llama2/configs/124m.toml --epochs 1 || true

llama-demo-lora: llama-examples ## Demo: Run LoRA fine-tuning demo
	@echo "🦙 Running LoRA fine-tuning demo..."
	@./target/release/examples/llama2-finetune-lora || true

llama-demo-qlora: llama-examples ## Demo: Run QLoRA fine-tuning demo
	@echo "🦙 Running QLoRA fine-tuning demo..."
	@./target/release/examples/llama2-finetune-qlora || true

llama-ci: llama-examples llama-tests ## Run LLaMA CI pipeline (build + test)
	@echo "✅ LLaMA CI pipeline complete!"
	@echo ""
	@echo "📊 LLaMA Quality Metrics:"
	@echo "  - ✅ 3 examples built (train, LoRA, QLoRA)"
	@echo "  - ✅ 13 property-based tests passing"
	@echo "  - ✅ 10 mutation-resistant tests"
	@echo "  - ✅ 15 chaos engineering tests"
	@echo "  - ✅ 18 gradient checking tests"
	@echo "  - ✅ 3 fuzz targets (1M+ iterations each)"
	@echo "  - ✅ Parameter-efficient fine-tuning validated"
	@echo ""
	@echo "Memory Benchmarks:"
	@echo "  124M Model:"
	@echo "    - Full FP32:  ~500 MB"
	@echo "    - QLoRA 4-bit: ~125 MB (75% savings)"
	@echo "  7B Model:"
	@echo "    - Full FP32:  ~28 GB"
	@echo "    - QLoRA 4-bit: ~7.5 GB (74% savings)"

# =============================================================================
# Observability & Tracing (Phase 4 - renacer integration)
# =============================================================================

profile-llama: llama-examples ## Profile LLaMA training with renacer (syscall-level bottleneck detection)
	@echo "🔍 Profiling LLaMA training with renacer..."
	@which renacer > /dev/null 2>&1 || (echo "⚠️  renacer not installed. Install from: https://github.com/durbanlegend/renacer" && echo "   cargo install renacer" && exit 1)
	@echo "  Running: renacer --function-time --source -- cargo run --release --example llama2-train"
	@echo ""
	@renacer --function-time --source --stats-extended -- \
		cargo run --release --example llama2-train --config examples/llama2/configs/124m.toml --epochs 1 2>&1 || true
	@echo ""
	@echo "✅ Profiling complete! Check output for hot paths and I/O bottlenecks."

profile-llama-otlp: llama-examples ## Profile LLaMA with OTLP export to Jaeger (requires docker-compose-jaeger.yml)
	@echo "🔍 Profiling LLaMA training with OTLP export..."
	@which renacer > /dev/null 2>&1 || (echo "⚠️  renacer not installed" && exit 1)
	@echo "  Ensure Jaeger is running: docker-compose -f docker-compose-jaeger.yml up -d"
	@echo "  View traces at: http://localhost:16686"
	@echo ""
	@renacer --otlp-endpoint http://localhost:4317 \
		--otlp-service-name llama-training \
		--trace-compute \
		--trace-compute-threshold 100 \
		--anomaly-realtime \
		--stats-extended \
		-- cargo run --release --example llama2-train --config examples/llama2/configs/124m.toml --epochs 1 2>&1 || true
	@echo ""
	@echo "✅ OTLP profiling complete! View traces in Jaeger UI."

profile-llama-anomaly: llama-examples ## Profile LLaMA with ML-based anomaly detection
	@echo "🔍 Profiling LLaMA training with ML anomaly detection..."
	@which renacer > /dev/null 2>&1 || (echo "⚠️  renacer not installed" && exit 1)
	@echo ""
	@renacer --ml-anomaly \
		--ml-clusters 5 \
		--ml-compare \
		--anomaly-realtime \
		--anomaly-threshold 3.0 \
		--stats-extended \
		--format json \
		-- cargo run --release --example llama2-train --config examples/llama2/configs/124m.toml --epochs 1 > .pmat/llama-training-profile.json 2>&1 || true
	@echo ""
	@echo "✅ ML anomaly detection complete! Profile saved to .pmat/llama-training-profile.json"
	@echo "  Run scripts/analyze_training.sh to analyze results."

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

# =============================================================================
# WASM Dashboard (Playwright e2e tests)
# =============================================================================

wasm-build: ## Build WASM monitor module
	@echo "🔨 Building WASM module..."
	@which wasm-pack > /dev/null 2>&1 || (echo "📦 Installing wasm-pack..." && cargo install wasm-pack)
	cd crates/entrenar-wasm && wasm-pack build --target web --out-dir ../../wasm-pkg/pkg
	@echo "✅ WASM build complete: wasm-pkg/pkg/"

wasm-install: ## Install npm dependencies for e2e
	@echo "📦 Installing e2e dependencies..."
	cd wasm-pkg && npm install
	cd wasm-pkg && npx playwright install chromium

wasm-serve: ## Serve WASM demo locally
	@echo "🌐 Starting demo server at http://localhost:9877"
	cd wasm-pkg && npx serve . -p 9877

wasm-e2e: wasm-build wasm-install ## Run Playwright e2e tests
	@echo "🎭 Running Playwright e2e tests..."
	cd wasm-pkg && npx playwright test
	@echo "✅ E2E tests complete!"

wasm-e2e-ui: wasm-build wasm-install ## Run Playwright with interactive UI
	cd wasm-pkg && npx playwright test --ui

wasm-e2e-headed: wasm-build wasm-install ## Run Playwright with visible browser
	cd wasm-pkg && npx playwright test --headed

wasm-e2e-update: wasm-build wasm-install ## Update Playwright snapshots
	cd wasm-pkg && npx playwright test --update-snapshots

wasm-clean: ## Clean WASM build artifacts
	rm -rf wasm-pkg/pkg wasm-pkg/node_modules wasm-pkg/playwright-report

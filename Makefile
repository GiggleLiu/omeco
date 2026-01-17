.DEFAULT_GOAL := help
.PHONY: help build build-release check fmt fmt-check clippy test check-all clean doc doc-private serve-docs python-dev python-build python-test benchmark version bump-patch bump-minor bump-major release publish publish-crates publish-all github-release

CARGO ?= cargo
PYTHON ?= python3
DOC_PORT ?= 8000
DOC_HOST ?= 127.0.0.1
DOC_CRATE ?= omeco

# Extract current version from omeco/Cargo.toml
VERSION := $(shell grep -m1 '^version' omeco/Cargo.toml | sed 's/.*"\(.*\)"/\1/')
MAJOR := $(shell echo $(VERSION) | cut -d. -f1)
MINOR := $(shell echo $(VERSION) | cut -d. -f2)
PATCH := $(shell echo $(VERSION) | cut -d. -f3)

help:
	@printf "Rust targets:\n"
	@printf "  build         Build the workspace\n"
	@printf "  build-release Build release binaries\n"
	@printf "  check         Run cargo check\n"
	@printf "  fmt           Format code\n"
	@printf "  fmt-check     Check formatting\n"
	@printf "  clippy        Run clippy (deny warnings)\n"
	@printf "  test          Run the test suite\n"
	@printf "  check-all     Run fmt-check, clippy, and test\n"
	@printf "  doc           Build rustdoc and open it\n"
	@printf "  serve-docs    Serve rustdoc at http://%s:%s/%s\n" "$(DOC_HOST)" "$(DOC_PORT)" "$(DOC_CRATE)"
	@printf "  clean         Clean build artifacts\n"
	@printf "\nPython targets:\n"
	@printf "  python-dev    Build and install Python package locally\n"
	@printf "  python-build  Build Python wheel\n"
	@printf "  python-test   Run Python tests\n"
	@printf "\nBenchmarks:\n"
	@printf "  benchmark     Run Python vs Julia benchmark\n"
	@printf "\nRelease targets:\n"
	@printf "  version         Show current version\n"
	@printf "  bump-patch      Bump patch version ($(VERSION) -> $(MAJOR).$(MINOR).$$(($(PATCH)+1)))\n"
	@printf "  bump-minor      Bump minor version ($(VERSION) -> $(MAJOR).$$(($(MINOR)+1)).0)\n"
	@printf "  bump-major      Bump major version ($(VERSION) -> $$(($(MAJOR)+1)).0.0)\n"
	@printf "  release         Create git tag and push (use after bump-*)\n"
	@printf "  publish-crates  Publish Rust crate to crates.io\n"
	@printf "  publish         Publish Python package to PyPI\n"
	@printf "  publish-all     Publish to both crates.io and PyPI\n"
	@printf "  github-release  Create GitHub release (requires gh CLI)\n"

build:
	$(CARGO) build --workspace

build-release:
	$(CARGO) build --workspace --release

check:
	$(CARGO) check --workspace

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --workspace --all-features

check-all: fmt-check clippy test
	@echo "All checks passed."

doc:
	$(CARGO) doc --no-deps --all-features --open -p omeco

doc-private:
	$(CARGO) doc --no-deps --document-private-items --open -p omeco

serve-docs:
	$(CARGO) doc --no-deps --all-features -p omeco
	@echo "Serving rustdoc at http://$(DOC_HOST):$(DOC_PORT)/$(DOC_CRATE)"
	$(PYTHON) -m http.server $(DOC_PORT) --directory target/doc --bind $(DOC_HOST)

clean:
	$(CARGO) clean

# Python targets
python-dev:
	cd omeco-python && maturin develop

python-build:
	cd omeco-python && maturin build --release

python-test: python-dev
	cd omeco-python && pip install -e ".[test]" && $(PYTHON) -m pytest -v

# Benchmarks
benchmark: python-dev
	$(PYTHON) benchmarks/benchmark_python.py
	cd benchmarks && julia --project=. benchmark_julia.jl

# Version management
version:
	@echo "Current version: $(VERSION)"

# Helper to update version in all files
define update_version
	@echo "Updating version from $(VERSION) to $(1)..."
	@sed -i 's/^version = "$(VERSION)"/version = "$(1)"/' omeco/Cargo.toml
	@sed -i 's/^version = "$(VERSION)"/version = "$(1)"/' omeco-python/pyproject.toml
	@echo "Updating Cargo.lock..."
	@cd omeco && $(CARGO) update -p omeco --precise $(1) 2>/dev/null || $(CARGO) check --quiet
	@echo "Updated version to $(1)"
	@git add omeco/Cargo.toml omeco-python/pyproject.toml Cargo.lock
	@git commit -m "Bump version to $(1)"
	@echo "Committed version bump"
endef

bump-patch:
	$(call update_version,$(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))

bump-minor:
	$(call update_version,$(MAJOR).$(shell echo $$(($(MINOR)+1))).0)

bump-major:
	$(call update_version,$(shell echo $$(($(MAJOR)+1))).0.0)

release:
	@echo "Creating release v$(VERSION)..."
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin master --tags
	@echo "Released v$(VERSION)"

publish-crates:
	@echo "Publishing omeco v$(VERSION) to crates.io..."
	cd omeco && $(CARGO) publish
	@echo "✓ Published to crates.io"
	@echo "  View at: https://crates.io/crates/omeco/$(VERSION)"

publish:
	@echo "Publishing omeco v$(VERSION) to PyPI..."
	cd omeco-python && maturin publish --skip-existing
	@echo "✓ Published to PyPI"
	@echo "  View at: https://pypi.org/project/omeco/$(VERSION)/"

publish-all: publish-crates publish
	@echo ""
	@echo "=========================================="
	@echo "✨ Published omeco v$(VERSION) to:"
	@echo "  - crates.io: https://crates.io/crates/omeco/$(VERSION)"
	@echo "  - PyPI: https://pypi.org/project/omeco/$(VERSION)/"
	@echo "=========================================="

github-release:
	@echo "Creating GitHub release for v$(VERSION)..."
	@if ! command -v gh >/dev/null 2>&1; then \
		echo "Error: gh CLI not found. Install from https://cli.github.com/"; \
		exit 1; \
	fi
	@echo "Generating release notes..."
	@gh release create v$(VERSION) \
		--title "v$(VERSION)" \
		--generate-notes \
		--verify-tag
	@echo "✓ GitHub release created"
	@echo "  View at: https://github.com/GiggleLiu/omeco/releases/tag/v$(VERSION)"

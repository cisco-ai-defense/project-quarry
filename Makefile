UV ?= uv
UV_RUN ?= $(UV) run --frozen --extra inference

.PHONY: help install install-inference lock-check test validate validate-rules e2e-signatures \
	language-generate language-check lint format format-check typecheck build ci

help:  ## Show available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

install:  ## Install the project and development dependencies
	$(UV) sync

install-inference:  ## Install the project with model provider support
	$(UV) sync --extra inference

lock-check:  ## Verify pyproject.toml and uv.lock are synchronized
	$(UV) lock --check

lint:  ## Check code quality with Ruff
	$(UV_RUN) ruff check .

format:  ## Format code with Ruff
	$(UV_RUN) ruff format .

format-check:  ## Check formatting without changing files
	$(UV_RUN) ruff format --check .

typecheck:  ## Type check the package
	$(UV_RUN) ty check quarry

test:  ## Run the test suite
	$(UV_RUN) pytest

validate: e2e-signatures  ## Run deterministic example checks

e2e-signatures:  ## Verify each example rule against its paired data
	$(UV_RUN) python scripts/e2e_signature_test.py

language-generate:  ## Regenerate the packaged DSL catalog and examples
	$(UV_RUN) python -m scripts.generate_language_spec

language-check:  ## Check generated DSL catalog and examples
	$(UV_RUN) python -m scripts.generate_language_spec --check

validate-rules: e2e-signatures  ## Load and execute every example rule with deterministic resources

build: lock-check  ## Build and validate wheel and source distribution
	$(UV) build
	$(UV_RUN) twine check dist/*

ci: lock-check language-check lint format-check typecheck test e2e-signatures  ## Run the CI checks locally

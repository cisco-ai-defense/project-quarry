# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

UV ?= uv
UV_RUN ?= $(UV) run --frozen --extra inference

.PHONY: help install install-inference lock-check test validate validate-rules validate-packs e2e-signatures \
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

validate: e2e-signatures validate-packs  ## Check language examples and detection rule packs

e2e-signatures:  ## Verify each example rule against its paired data
	$(UV_RUN) python scripts/e2e_signature_test.py

language-generate:  ## Regenerate packaged language and plugin schemas
	$(UV_RUN) python -m scripts.generate_language_spec

language-check:  ## Check generated language and plugin schemas
	$(UV_RUN) python -m scripts.generate_language_spec --check

validate-rules: e2e-signatures  ## Load and execute every example rule with deterministic resources

validate-packs:  ## Check rule pack layout and compile detection rules
	$(UV_RUN) python rules/validate.py

build: lock-check  ## Build and validate wheel and source distribution
	$(UV) build
	$(UV_RUN) twine check dist/*

ci: lock-check language-check lint format-check typecheck test e2e-signatures validate-packs  ## Run the CI checks locally

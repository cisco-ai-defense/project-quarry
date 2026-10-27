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

"""End-to-end signature functionality test.

This is a black-box test of the Quarry engine and the shipped example rules.
Even when unit tests pass, this verifies that the library actually parses,
loads, and evaluates every example signature against real conversation data.

Contract enforced by this script:

1. Every rule in ``examples/rules/``, ``examples/policy-only/``, or
   ``examples/legacy-v1/`` has exactly one matching
   ``examples/data/<stem>.json`` dataset. The latter two sets are not loaded
   by the general CLI scanner config.
2. Each rule, loaded on its own, MUST match its paired dataset. This proves the
   signature and the feature of the DSL it exercises execute correctly.
3. If an optional ``<stem>.negative.json`` dataset exists, its rule MUST NOT
   match that dataset. This checks the boundary each example claims to enforce.
4. No rule may match the benign control dataset(s). This guards against
   signatures that accidentally match everything.
5. Split examples must score multiple smaller chunks, and the span example
   must collect the expected email span.
6. The documented general scanner config must load only shared rules and
   complete an ordinary scan when its model resources are configured.

Model-backed examples run through a deterministic in-process adapter, and
agent-drift examples use the explicit dependency-free prober stubs. The gate
therefore exercises every shipped example without API keys, network calls, or
model downloads.

Exit code is 0 when every check passes and 1 otherwise, so it can gate merges.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.policy import compile_policy
from quarry.scanner import DSLScanner
from quarry.types.results import Score, Span

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
RULES_DIR = EXAMPLES_DIR / "rules"
POLICY_RULES_DIR = EXAMPLES_DIR / "policy-only"
LEGACY_RULES_DIR = EXAMPLES_DIR / "legacy-v1"
DATA_DIR = EXAMPLES_DIR / "data"
DRIFT_DIR = EXAMPLES_DIR / "drift"
GEMMA4_DIR = EXAMPLES_DIR / "gemma4"
MANIFEST_PATH = EXAMPLES_DIR / "corpus-manifest.json"

# Datasets that intentionally have no paired rule. They must NOT match any rule.
BENIGN_DATASETS = ["benign.json", "benign_multiturn.json"]
SPLIT_EXAMPLES = {"model_split", "model_split_compat", "llm_split_conversation"}
EXAMPLE_OPTIONS = yaml.safe_load((EXAMPLES_DIR / "runner-options.yaml").read_text())


def _load_dataset(data_file: Path) -> tuple[dict, dict]:
    """Load a CLI-compatible conversation and its separate runner settings."""
    payload = json.loads(data_file.read_text())
    options_key = data_file.resolve().relative_to(EXAMPLES_DIR).as_posix()
    options = EXAMPLE_OPTIONS.get(options_key, {})
    return payload, options


def _joined_text(messages) -> str:
    return " ".join(
        str(message.get("content") or "") for message in messages if isinstance(message, dict)
    )


class _DeterministicModelAdapter:
    """Implement every model operation used by the examples without external I/O."""

    def __init__(self) -> None:
        self.scored_message_counts: list[int] = []

    @staticmethod
    def _probability(text: str) -> float:
        if "QUARRY_MODEL_BELOW_SCAN" in text:
            return 0.49
        if "QUARRY_MODEL_BELOW_EIGHTY" in text:
            return 0.79
        return 0.9 if "QUARRY_MODEL_MATCH" in text or "@example.net" in text else 0.1

    async def classify(self, _config, text, *, timeout=None):
        probability = self._probability(text)
        return SimpleNamespace(
            chunks=[SimpleNamespace(probabilities=[probability, 0.7 if probability > 0.5 else 0.1])]
        )

    async def classify_messages(self, _config, messages, _tools, *, timeout=None):
        probability = self._probability(_joined_text(messages))
        return SimpleNamespace(
            chunks=[SimpleNamespace(probabilities=[probability, 0.7 if probability > 0.5 else 0.1])]
        )

    async def score(self, _config, messages, *, timeout=None):
        self.scored_message_counts.append(len(messages))
        return self._probability(_joined_text(messages))

    async def complete(self, _config, messages, _tools, *, timeout=None):
        text = _joined_text(messages)
        if "QUARRY_COMPLETION_NEAR_MISS" in text:
            return "QUARRY_BLOCKED_COMPLETIO"
        if "QUARRY_MODEL_MATCH" in text:
            return "QUARRY_BLOCKED_COMPLETION"
        return "safe"

    async def complete_message(self, _config, messages, _tools, *, timeout=None):
        text = _joined_text(messages)
        if "QUARRY_COMPLETION_NEAR_MISS" in text:
            content = "QUARRY_BLOCKED_COMPLETIO"
        else:
            content = "QUARRY_BLOCKED_COMPLETION" if "QUARRY_MODEL_MATCH" in text else "safe"
        return {"role": "assistant", "content": content}

    async def tokenize(self, config, messages, tools, *, timeout=None):
        count = sum(
            len(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, dict)
        ) + len(tools)
        return {config.tokens.count_field: count}

    async def decode_spans(self, _config, _raw_result, messages, _threshold):
        for message_index, message in enumerate(messages):
            match = re.search(r"[A-Za-z0-9._%+-]+@example\.net", message.content)
            if match:
                return [Span(message_index, match.start(), match.end(), Score(0.9))]
        return []


def _score_config(version: int) -> dict:
    result: dict = {"message_builder": "example", "boundary_seed": 7}
    if version == 1:
        result.update(
            {
                "tokens": {"negative": "0", "positive": "1"},
                "extraction": {"probabilities": "probabilities"},
            }
        )
    return result


def _model_config(version: int, rule_file: Path | None) -> QuarryConfig:
    def model(*, classify=False, score=False, tokens=False, spans=False, category=None):
        value: dict = {"adapter": "fake", "model": "example/model"}
        if version == 2 and classify:
            value["classify"] = {"class_index": 0}
        if score:
            value["score"] = _score_config(version)
        if tokens:
            value["tokens"] = {
                "adapter": "fake",
                "endpoint": None,
                "extraction": {"count": "count"},
            }
        if version == 2 and spans:
            value["spans"] = {"class_index": 0}
        if category is not None:
            value["category"] = category
        return value

    mapping = {
        "config_version": version,
        "models": {
            "classifier": model(classify=True),
            "completion": model(),
            "email": model(classify=True, spans=True, category="EMAIL"),
            "gemma4": model(score=True),
            "judge": model(score=True, tokens=True),
            "prompt_injection": model(classify=True),
        },
        "texts": {"policy": "Flag the deterministic model marker."},
        "allow_stub_probers": True,
        "rule_timeout": 2.0,
    }
    if rule_file is not None:
        mapping["rules"] = [str(rule_file)]
    return QuarryConfig.from_mapping(
        mapping,
        EXAMPLES_DIR.parent,
        require_rules=rule_file is not None,
    )


def _message_builder(_constitution, model_input, *args):
    if hasattr(model_input, "to_openai_messages"):
        return model_input.to_openai_messages()
    return model_input


def _model_backend(
    config: QuarryConfig, adapter: _DeterministicModelAdapter | None = None
) -> ModelInferenceBackend:
    return ModelInferenceBackend(
        config,
        adapters={"fake": adapter or _DeterministicModelAdapter()},
        message_builders={"example": _message_builder},
    )


def _config_version_for(options: dict) -> int:
    """Read an explicit config version from dataset metadata (defaulting to v2)."""
    version = options.get("config_version", 2)
    if version not in (1, 2):
        raise ValueError(f"unsupported example config_version: {version!r}")
    return version


class ExampleRunner:
    """Execute one example rule against ordinary conversation datasets."""

    def __init__(
        self,
        rule_name: str,
        scan_data,
        model_adapter: _DeterministicModelAdapter | None = None,
    ) -> None:
        self.rule_name = rule_name
        self._scan_data = scan_data
        self.model_adapter = model_adapter

    def scan_file(self, data_file: Path):
        conversation, options = _load_dataset(data_file)
        return self._scan_data(conversation, options)


def _incomplete_scan_detail(result) -> str | None:
    """Describe a non-authoritative scan result, or return None when complete."""
    if result.is_complete:
        return None
    issues = "; ".join(result.completion_issues) or "unknown reason"
    return f"{result.completion_state} scan: {issues}"


class Reporter:
    """Collects and prints pass/fail results."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.rows: list[tuple[bool, str, str]] = []

    def record(self, ok: bool, name: str, detail: str = "") -> None:
        """Record a test result."""
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.rows.append((ok, name, detail))

    def summary(self) -> int:
        """Print the summary of the test results."""
        print("\n" + "=" * 72)
        print("SIGNATURE E2E SUMMARY")
        print("=" * 72)
        for ok, name, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {name}")
            if detail:
                print(f"       {detail}")
        print("-" * 72)
        print(f"Total: {self.passed} passed, {self.failed} failed")
        print("=" * 72)
        return 1 if self.failed else 0


def check_manifest(reporter: Reporter) -> None:
    """Keep the shipped example inventory fixed until it is deliberately reviewed."""
    groups = ("rules", "datasets", "support")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError) as exc:
        reporter.record(False, "example corpus manifest", f"{type(exc).__name__}: {exc}")
        return
    if not isinstance(manifest, dict) or set(manifest) != set(groups):
        reporter.record(False, "example corpus manifest", "expected rules, datasets, support lists")
        return
    for group in groups:
        paths = manifest[group]
        if (
            not isinstance(paths, list)
            or not all(isinstance(path, str) for path in paths)
            or paths != sorted(set(paths))
            or any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths)
        ):
            reporter.record(False, "example corpus manifest", f"invalid {group} list")
            return

    root = EXAMPLES_DIR.parent
    actual_files = sorted(
        path.relative_to(root).as_posix()
        for path in EXAMPLES_DIR.rglob("*")
        if path.is_file()
        and path != MANIFEST_PATH
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    actual = {
        "rules": [path for path in actual_files if path.endswith(".rule")],
        "datasets": [path for path in actual_files if path.endswith(".json")],
        "support": [path for path in actual_files if not path.endswith((".rule", ".json"))],
    }
    missing = sorted(set().union(*(set(manifest[group]) - set(actual[group]) for group in groups)))
    unexpected = sorted(
        set().union(*(set(actual[group]) - set(manifest[group]) for group in groups))
    )
    reporter.record(
        not missing and not unexpected,
        "example corpus manifest",
        (
            f"missing={missing}, unexpected={unexpected}"
            if missing or unexpected
            else f"{len(manifest['rules'])} rules, {len(manifest['datasets'])} datasets, "
            f"{len(manifest['support'])} support files"
        ),
    )

    compile_errors = []
    for name in manifest["rules"]:
        try:
            compiled = compile_rule(DSLParser().parse((root / name).read_text()))
            if not compiled.ok:
                compile_errors.append(
                    (name, [item.message for item in compiled.diagnostics.errors])
                )
        except Exception as exc:  # noqa: BLE001 - report as a gate failure
            compile_errors.append((name, f"{type(exc).__name__}: {exc}"))
    reporter.record(
        not compile_errors,
        "every shipped example rule compiles",
        str(compile_errors) if compile_errors else f"compiled {len(manifest['rules'])} rules",
    )


def load_single_rule_runner(rule_file: Path, data_file: Path) -> ExampleRunner:
    """Build a deterministic runner for exactly one rule and its paired dataset."""
    rule_file = rule_file.resolve()
    data_file = data_file.resolve()
    _, options = _load_dataset(data_file)
    version = _config_version_for(options)
    if rule_file.parent == POLICY_RULES_DIR:
        if "policy_parameters" not in options:
            raise ValueError("policy-only example requires policy_parameters in its dataset")
        config = _model_config(version, None)
        model_adapter = _DeterministicModelAdapter()
        policy = compile_policy(
            rule_file.read_text(),
            policy_parameters=options.get("policy_parameters", {}),
            runtime_config=config,
            model_inference_backend=_model_backend(config, model_adapter),
        )
        parsed = DSLParser().parse(rule_file.read_text())
        default_scan_parameters = options.get("scan_parameters", {})
        return ExampleRunner(
            parsed.name,
            lambda conversation, dataset_options: policy.scan(
                conversation,
                {**default_scan_parameters, **dataset_options.get("scan_parameters", {})},
            ),
            model_adapter,
        )

    if "policy_parameters" in options:
        raise ValueError("parameterized example must be in examples/policy-only/")
    if rule_file.parent == LEGACY_RULES_DIR and version != 1:
        raise ValueError("legacy-v1 example requires config_version: 1")
    if rule_file.parent == RULES_DIR and version != 2:
        raise ValueError("general scanner example requires config_version: 2")

    config = _model_config(version, rule_file)
    model_adapter = _DeterministicModelAdapter()
    scanner = DSLScanner(
        config=config, model_inference_backend=_model_backend(config, model_adapter)
    )
    rule_names = list(scanner.loaded_rules.keys())
    if len(rule_names) != 1:
        raise ValueError(f"expected exactly 1 rule, loaded {rule_names}")
    return ExampleRunner(
        rule_names[0], lambda conversation, _options: scanner.scan_data(conversation), model_adapter
    )


def check_pairing(reporter: Reporter) -> tuple[list[Path], dict[Path, Path]]:
    """Verify strict 1:1 rule <-> dataset pairing and return the pairs."""
    rule_files = sorted(
        [
            *RULES_DIR.glob("*.rule"),
            *POLICY_RULES_DIR.glob("*.rule"),
            *LEGACY_RULES_DIR.glob("*.rule"),
        ]
    )
    data_files = {p.name for p in DATA_DIR.glob("*.json")}

    pairs: dict[Path, Path] = {}
    stems: dict[str, Path] = {}
    for rule_file in rule_files:
        if previous := stems.get(rule_file.stem):
            reporter.record(
                False,
                f"duplicate rule stem: {rule_file.stem}",
                f"{previous} and {rule_file} share one dataset",
            )
            continue
        stems[rule_file.stem] = rule_file
        data_name = f"{rule_file.stem}.json"
        data_file = DATA_DIR / data_name
        if data_file.exists():
            pairs[rule_file] = data_file
            reporter.record(True, f"pairing: {rule_file.name} <-> {data_name}")
        else:
            reporter.record(
                False,
                f"pairing: {rule_file.name} <-> {data_name}",
                f"missing dataset {data_file}",
            )

    # Datasets with no corresponding rule (benign controls are allowed).
    expected_data = {f"{p.stem}.json" for p in rule_files} | set(BENIGN_DATASETS)
    expected_data |= {f"{p.stem}.negative.json" for p in rule_files}
    orphans = sorted(data_files - expected_data)
    for orphan in orphans:
        reporter.record(
            False,
            f"orphan dataset: {orphan}",
            "dataset has no corresponding .rule file",
        )

    return rule_files, pairs


def check_general_scanner(reporter: Reporter) -> None:
    """Exercise the documented config's rule selection through an ordinary scan."""
    try:
        template = QuarryConfig.from_yaml(EXAMPLES_DIR / "all-rules.yaml")
        config = replace(
            _model_config(template.config_version, None),
            quarry_rules_paths=template.quarry_rules_paths,
        )
        scanner = DSLScanner(config=config, model_inference_backend=_model_backend(config))
        excluded_files = [*POLICY_RULES_DIR.glob("*.rule"), *LEGACY_RULES_DIR.glob("*.rule")]
        excluded_names = {DSLParser().parse(path.read_text()).name for path in excluded_files}
        loaded_names = set(scanner.loaded_rules)
        expected_names = {
            DSLParser().parse(path.read_text()).name for path in RULES_DIR.glob("*.rule")
        }
        reporter.record(
            loaded_names == expected_names,
            "general scanner loads every shared rule",
            (
                f"missing={sorted(expected_names - loaded_names)}, "
                f"unexpected={sorted(loaded_names - expected_names)}"
                if loaded_names != expected_names
                else f"loaded {len(loaded_names)} rules"
            ),
        )
        excluded_loaded = sorted(excluded_names & loaded_names)
        reporter.record(
            not excluded_loaded,
            "general scanner excludes policy-only and legacy-v1 rules",
            f"unexpectedly loaded {excluded_loaded}"
            if excluded_loaded
            else "specialized rules excluded",
        )
        result = scanner.scan_data(_load_dataset(DATA_DIR / "benign.json")[0])
        incomplete_detail = _incomplete_scan_detail(result)
        reporter.record(
            incomplete_detail is None,
            "general scanner completes an ordinary scan",
            incomplete_detail or f"loaded {len(loaded_names)} rules",
        )
    except Exception as exc:  # noqa: BLE001 - report load/scan failures as gate results
        reporter.record(
            False, "general scanner loads examples/all-rules.yaml", f"{type(exc).__name__}: {exc}"
        )


def check_standalone_examples(reporter: Reporter) -> None:
    """Run the optional drift and model-provider examples without external services."""
    drift_data = DRIFT_DIR / "data"
    if not DRIFT_DIR.is_dir() or not drift_data.is_dir():
        reporter.record(False, "drift examples exist", "examples/drift/ or its data/ is missing")
    else:
        drift_rules = sorted(DRIFT_DIR.glob("*.rule"))
        reporter.record(
            bool(drift_rules), "drift examples exist", f"found {len(drift_rules)} rules"
        )
        expected_data = {
            name
            for rule in drift_rules
            for name in (f"{rule.stem}.json", f"{rule.stem}.negative.json")
        }
        for orphan in sorted({path.name for path in drift_data.glob("*.json")} - expected_data):
            reporter.record(False, f"orphan drift dataset: {orphan}")
        for rule_file in drift_rules:
            positive = drift_data / f"{rule_file.stem}.json"
            negative = drift_data / f"{rule_file.stem}.negative.json"
            if not positive.is_file() or not negative.is_file():
                reporter.record(False, f"drift pairing: {rule_file.name}", "missing paired dataset")
                continue
            reporter.record(True, f"drift pairing: {rule_file.name}")
            try:
                runner = load_single_rule_runner(rule_file, positive)
                for name, data_file, expected_match in (
                    ("positive", positive, True),
                    ("negative", negative, False),
                ):
                    result = runner.scan_file(data_file)
                    incomplete = _incomplete_scan_detail(result)
                    matched = runner.rule_name in result.rule_matches
                    reporter.record(
                        incomplete is None and matched is expected_match,
                        f"drift {name}: {rule_file.name}",
                        incomplete or f"matched={matched}, expected={expected_match}",
                    )
            except Exception as exc:  # noqa: BLE001 - report as a gate failure
                reporter.record(
                    False, f"drift load: {rule_file.name}", f"{type(exc).__name__}: {exc}"
                )

    gemma_rule = GEMMA4_DIR / "prompt_injection.rule"
    gemma_data = GEMMA4_DIR / "conversation.json"
    other_rules = sorted(path.name for path in GEMMA4_DIR.glob("*.rule") if path != gemma_rule)
    if other_rules:
        reporter.record(False, "uncovered model-provider rules", f"uncovered={other_rules}")
    if not gemma_rule.is_file() or not gemma_data.is_file():
        reporter.record(False, "model-provider example exists", "rule or conversation is missing")
        return

    class _ExampleJudge(_DeterministicModelAdapter):
        async def score(self, _config, messages, *, timeout=None):
            self.scored_message_counts.append(len(messages))
            return 1.0

    try:
        compiled = compile_rule(DSLParser().parse(gemma_rule.read_text()))
        reporter.record(
            compiled.ok,
            "model-provider rule compiles",
            "; ".join(item.message for item in compiled.diagnostics.errors),
        )
        config = _model_config(2, gemma_rule)
        adapter = _ExampleJudge()
        scanner = DSLScanner(config=config, model_inference_backend=_model_backend(config, adapter))
        rule_name = next(iter(scanner.loaded_rules))
        for label, path, expected_match in (
            ("positive", gemma_data, True),
            ("benign", DATA_DIR / "benign.json", False),
        ):
            result = scanner.scan_data(_load_dataset(path)[0])
            incomplete = _incomplete_scan_detail(result)
            matched = rule_name in result.rule_matches
            reporter.record(
                incomplete is None and matched is expected_match,
                f"model-provider {label}: {path.name}",
                incomplete or f"matched={matched}, expected={expected_match}",
            )
        reporter.record(bool(adapter.scored_message_counts), "model-provider judge executed")
    except Exception as exc:  # noqa: BLE001 - report as a gate failure
        reporter.record(False, "model-provider rule loads", f"{type(exc).__name__}: {exc}")


def main() -> int:
    """Main function to run the signature E2E test."""
    print("=" * 72)
    print("Quarry Signature E2E Test")
    print(f"  rules: {RULES_DIR}")
    print(f"  policy-only: {POLICY_RULES_DIR}")
    print(f"  legacy-v1: {LEGACY_RULES_DIR}")
    print(f"  data:  {DATA_DIR}")
    print("=" * 72)

    reporter = Reporter()
    check_manifest(reporter)

    if (
        not RULES_DIR.is_dir()
        or not POLICY_RULES_DIR.is_dir()
        or not LEGACY_RULES_DIR.is_dir()
        or not DATA_DIR.is_dir()
    ):
        print(f"[ERROR] examples directories not found under {EXAMPLES_DIR}", file=sys.stderr)
        return 1

    rule_files, pairs = check_pairing(reporter)
    check_general_scanner(reporter)
    check_standalone_examples(reporter)

    benign_files = [DATA_DIR / name for name in BENIGN_DATASETS if (DATA_DIR / name).exists()]

    for rule_file in rule_files:
        data_file = pairs.get(rule_file)
        if data_file is None:
            continue
        try:
            runner = load_single_rule_runner(rule_file, data_file)
        except Exception as exc:  # noqa: BLE001 - report any load/parse failure
            reporter.record(False, f"load: {rule_file.name}", f"{type(exc).__name__}: {exc}")
            continue
        rule_name = runner.rule_name

        # Each signature has its own adapter-call assertions, as in a fresh process.
        result_cache.clear()

        # Positive: the rule must match its paired dataset.
        result = runner.scan_file(data_file)
        incomplete_detail = _incomplete_scan_detail(result)
        if incomplete_detail:
            reporter.record(
                False,
                f"match: [{rule_name}] vs {data_file.name}",
                incomplete_detail,
            )
        else:
            matched = rule_name in result.rule_matches
            reporter.record(
                matched,
                f"match: [{rule_name}] vs {data_file.name}",
                "matched (expected)" if matched else "rule did NOT match its dataset",
            )

        if rule_file.stem in SPLIT_EXAMPLES and runner.model_adapter is not None:
            scored_counts = runner.model_adapter.scored_message_counts
            message_count = len(_load_dataset(data_file)[0]["messages"])
            reporter.record(
                len(scored_counts) > 1 and max(scored_counts) < message_count,
                f"split: [{rule_name}] scores multiple smaller chunks",
                f"scored message counts={scored_counts}, input message count={message_count}",
            )

        if rule_file.stem == "model_spans":
            messages = _load_dataset(data_file)[0]["messages"]
            output = result.outputs.get(rule_name)
            spans = [
                item.value for item in getattr(output, "items", ()) if isinstance(item.value, Span)
            ]
            extracted = [
                messages[span.message_index]["content"][span.start : span.end] for span in spans
            ]
            reporter.record(
                "alice@example.net" in extracted,
                f"collect: [{rule_name}] returns the email span",
                f"extracted spans={extracted}",
            )

        # The optional paired negative must sit just outside the rule's boundary.
        negative_file = DATA_DIR / f"{rule_file.stem}.negative.json"
        if negative_file.exists():
            result = runner.scan_file(negative_file)
            incomplete_detail = _incomplete_scan_detail(result)
            if incomplete_detail:
                reporter.record(
                    False,
                    f"negative: [{rule_name}] vs {negative_file.name}",
                    incomplete_detail,
                )
            else:
                clean = rule_name not in result.rule_matches
                reporter.record(
                    clean,
                    f"negative: [{rule_name}] vs {negative_file.name}",
                    "no match (expected)" if clean else "rule wrongly matched negative data",
                )

        for benign_file in benign_files:
            result = runner.scan_file(benign_file)
            incomplete_detail = _incomplete_scan_detail(result)
            if incomplete_detail:
                reporter.record(
                    False,
                    f"benign: [{rule_name}] vs {benign_file.name}",
                    incomplete_detail,
                )
                continue
            clean = rule_name not in result.rule_matches
            reporter.record(
                clean,
                f"benign: [{rule_name}] vs {benign_file.name}",
                "no match (expected)" if clean else "rule wrongly matched benign data",
            )

    return reporter.summary()


if __name__ == "__main__":
    sys.exit(main())

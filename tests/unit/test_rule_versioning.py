# Copyright 2025 Cisco Systems, Inc. and its affilitates
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

"""Contract for optional ``requires_quarry`` rule metadata.

A rule may declare the Quarry versions it supports. Admission rejects every incompatible rule in
one typed error, before any resource is prepared, through every public entry point.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from quarry import __version__
from quarry.cli import BANNER, EXIT_INCOMPLETE, scan_command
from quarry.config import QuarryConfig
from quarry.engine.contracts import COMPILER_VERSION
from quarry.policy import compile_policy
from quarry.rule_versioning import (
    REQUIREMENT_METADATA_KEY,
    InvalidQuarryRequirement,
    check_rule_compatibility,
    parse_quarry_requirement,
)
from quarry.scanner import DSLScanner
from quarry.sdk import QuarryGuard
from quarry.types.exceptions import RuleCompatibilityError, RuleValidationError

from ..conftest import rule_matches


def _rule_text(name: str, requirement: str | None = None, *, condition: str = "true") -> str:
    metadata = (
        f'metadata:\n    {REQUIREMENT_METADATA_KEY} = "{requirement}"\n'
        if requirement is not None
        else ""
    )
    return f'rule "{name}"\n{metadata}condition: {condition}\n'


def _programmatic_rule(declaration: object):
    return SimpleNamespace(
        name="Programmatic", metadata={REQUIREMENT_METADATA_KEY: declaration}, source_file=None
    )


def test_the_package_banner_and_language_spec_agree_on_the_version() -> None:
    language_spec = json.loads(
        (Path(__file__).parents[2] / "quarry" / "language" / "spec.json").read_text(
            encoding="utf-8"
        )
    )

    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]
    assert project["version"] == __version__
    try:
        installed_version = importlib.metadata.version("cisco-project-quarry")
    except importlib.metadata.PackageNotFoundError:
        pass  # Source-tree test runs do not need an editable install.
    else:
        assert installed_version == __version__
    assert f"v{__version__}" in BANNER
    assert language_spec["language_version"] == __version__
    assert language_spec["compiler_version"] == COMPILER_VERSION


@pytest.mark.parametrize(
    "declaration",
    [
        "",
        "0.2.0",
        "==0.2.0",
        ">0.2.0",
        "~=0.2.0",
        ">=0.2.0,!=0.2.1",
        ">=0.2.0,<=0.3.0",
        ">=0.2.0,<0.3.0,<0.4.0",
        ">=0.2.0 || >=0.4.0",
        " >=0.2.0",
        ">=0.2.0, <0.3.0",
        ">=not-a-version",
        ">=0.4.0,<0.3.0",
        None,
        2,
        [">=0.2.0"],
    ],
)
def test_a_malformed_or_unsupported_requirement_is_reported_as_invalid(declaration) -> None:
    with pytest.raises(InvalidQuarryRequirement):
        parse_quarry_requirement(declaration)

    issue = check_rule_compatibility(
        _programmatic_rule(declaration), source="unit-test", running_version="0.2.0"
    )
    assert (issue.reason, issue.source, issue.rule_name, issue.declaration) == (
        "invalid_requirement",
        "unit-test",
        "Programmatic",
        declaration,
    )
    assert issue.running_version == "0.2.0"


@pytest.mark.parametrize(
    ("declaration", "running_version", "reason"),
    [
        (">=1.0", "0.9.9", "runtime_below_minimum"),
        (">=1.0", "1.0", None),
        (">=1.0", "1.0+build.7", None),
        (">=1.0,<2.0", "1.0", None),
        (">=1.0,<2.0", "1.9.9", None),
        (">=1.0,<2.0", "2.0", "runtime_at_or_above_upper_bound"),
        (">=1.0,<2.0", "2.0.dev1", None),
        (">=1.0", "1.0rc1", "runtime_below_minimum"),
        (">=1.0.dev1,<1.0", "1.0.dev1", None),
        (">=1.0.dev1,<1.0", "1.0a1", None),
        (">=1.0.dev1,<1.0", "1.0rc1", None),
        (">=1.0.dev1,<1.0", "1.0", "runtime_at_or_above_upper_bound"),
    ],
)
def test_compatibility_uses_direct_version_ordering_at_every_boundary(
    declaration: str, running_version: str, reason: str | None
) -> None:
    issue = check_rule_compatibility(
        _programmatic_rule(declaration), running_version=running_version
    )

    assert (issue.reason if issue is not None else None) == reason


def test_a_rule_without_a_requirement_loads() -> None:
    scanner = DSLScanner(QuarryConfig(quarry_rule_texts=(_rule_text("Unannotated"),)))

    assert list(scanner.loaded_rules) == ["Unannotated"]


def test_a_compatible_requirement_does_not_change_the_verdict() -> None:
    conversation = {"messages": [{"role": "user", "content": "danger"}]}
    condition = 'user_said("danger")'

    assert rule_matches(_rule_text("Plain", condition=condition), conversation) is True
    assert (
        rule_matches(_rule_text("Annotated", ">=0,<9999", condition=condition), conversation)
        is True
    )


def test_admission_reports_every_incompatible_rule_in_source_order(tmp_path: Path) -> None:
    future = tmp_path / "a-future.rule"
    malformed = tmp_path / "b-malformed.rule"
    expired = tmp_path / "c-expired.rule"
    compatible = tmp_path / "d-compatible.rule"
    # The future rule names a resource nobody configured: admission must fail before it is needed.
    future.write_text(
        _rule_text("Future", ">=9999", condition="topic_drift(system_prompt) > 0.5"),
        encoding="utf-8",
    )
    malformed.write_text(_rule_text("Malformed", "==0.2.0"), encoding="utf-8")
    expired.write_text(_rule_text("Expired", ">=0,<0.1"), encoding="utf-8")
    compatible.write_text(_rule_text("Compatible", ">=0"), encoding="utf-8")

    with pytest.raises(RuleCompatibilityError) as exc_info:
        DSLScanner(
            QuarryConfig(
                quarry_rules_paths=[tmp_path],
                quarry_rule_texts=[_rule_text("Inline future", ">=9998")],
            ),
        )

    error = exc_info.value
    assert error.running_version == __version__
    assert [
        (issue.rule_name, issue.source, issue.declaration, issue.reason) for issue in error.issues
    ] == [
        ("Future", str(future), ">=9999", "runtime_below_minimum"),
        ("Malformed", str(malformed), "==0.2.0", "invalid_requirement"),
        ("Expired", str(expired), ">=0,<0.1", "runtime_at_or_above_upper_bound"),
        ("Inline future", "inline:0", ">=9998", "runtime_below_minimum"),
    ]
    assert error.details == {
        "error_type": "rule_compatibility",
        "running_version": __version__,
        "issues": [issue.to_dict() for issue in error.issues],
    }


def test_a_duplicate_rule_name_is_reported_before_compatibility(tmp_path: Path) -> None:
    (tmp_path / "a.rule").write_text(_rule_text("Duplicate", ">=9999"), encoding="utf-8")
    (tmp_path / "b.rule").write_text(_rule_text("Duplicate", "not-a-requirement"), encoding="utf-8")

    with pytest.raises(RuleValidationError, match="Duplicate rule name") as exc_info:
        DSLScanner(QuarryConfig(quarry_rules_paths=[tmp_path]))

    assert not isinstance(exc_info.value, RuleCompatibilityError)


def _incompatible_config(tmp_path: Path, name: str) -> Path:
    rule_path = tmp_path / "incompatible.rule"
    rule_path.write_text(_rule_text(name, ">=9999"), encoding="utf-8")
    config_path = tmp_path / "quarry.yaml"
    config_path.write_text(yaml.safe_dump({"rules": [str(rule_path)]}), encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    "entry_point",
    [
        pytest.param(lambda config, _source: QuarryGuard(config), id="sdk-guard"),
        pytest.param(lambda _config, source: compile_policy(source), id="compiled-policy"),
    ],
)
def test_every_public_entry_point_raises_the_typed_admission_error(tmp_path, entry_point):
    config_path = _incompatible_config(tmp_path, "Public gate")

    with pytest.raises(RuleCompatibilityError) as exc_info:
        entry_point(config_path, _rule_text("Public gate", ">=9999"))

    assert [issue.reason for issue in exc_info.value.issues] == ["runtime_below_minimum"]


def test_the_cli_reports_incompatibility_as_an_initialization_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _incompatible_config(tmp_path, "CLI gate")
    target = tmp_path / "conversation.json"
    target.write_text('{"messages":[{"role":"user","content":"hello"}]}', encoding="utf-8")
    args = SimpleNamespace(
        config=config_path,
        target=target,
        no_limits=False,
        recursive=False,
        verbose=False,
        batch=False,
        validate_only=False,
        output_format="json",
        explain=False,
        show_timing=False,
        output=None,
    )

    assert scan_command(args) == EXIT_INCOMPLETE
    captured = capsys.readouterr()
    assert "Failed to initialize scanner" in captured.err
    assert REQUIREMENT_METADATA_KEY in captured.err
    assert "below minimum" in captured.err

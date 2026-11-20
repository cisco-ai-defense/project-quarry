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

"""Pack validation without reading datasets or loading providers."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import pytest

from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.scanner import DSLScanner
from rules.validate import validate_repository

_REQUIRED_METADATA = {
    "description": "Detect an example marker.",
    "author": "Quarry contributors",
    "revision": "0",
    "requires_quarry": ">=0.1.0,<0.2.0",
}


@pytest.fixture
def repository(tmp_path: Path, pytestconfig: pytest.Config) -> Path:
    packs = tmp_path / "rules/rule-packs"
    packs.mkdir(parents=True)
    copyfile(
        pytestconfig.rootpath / "rules/rule-packs/metadata-schema.yaml",
        packs / "metadata-schema.yaml",
    )
    return tmp_path


def _add_rule(
    root: Path,
    *,
    pack: str = "core/fixture-pack",
    name: str | None = None,
    condition: str = 'any msg in user_messages where msg.content contains "example marker"',
    metadata: dict[str, str] | None = None,
) -> Path:
    stem = "example_marker"
    path = root / "rules/rule-packs" / pack / stem / f"{stem}.rule"
    path.parent.mkdir(parents=True)
    values = _REQUIRED_METADATA | (metadata or {})
    path.write_text(
        f"rule {json.dumps(name or f'{pack}/{stem}')}\nmetadata:\n"
        + "".join(f"    {key} = {json.dumps(value)}\n" for key, value in values.items())
        + f"condition:\n    {condition}\n",
        encoding="utf-8",
    )
    path.with_name("README.md").write_text(
        "# Example marker\n\nCovers messages containing an example marker.\n", encoding="utf-8"
    )
    return path


def test_core_and_community_rules_compile_with_readmes(repository: Path) -> None:
    _add_rule(repository)
    _add_rule(repository, pack="community/example-author/example-pack")

    report = validate_repository(repository)

    assert report.ok, report.errors
    assert report.compiled_rules == 2


@pytest.mark.parametrize("readme_text", [None, "<!-- copyright notice -->\n"])
def test_rule_readme_must_have_an_overview(repository: Path, readme_text: str | None) -> None:
    readme = _add_rule(repository).with_name("README.md")
    if readme_text is None:
        readme.unlink()
    else:
        readme.write_text(readme_text, encoding="utf-8")

    report = validate_repository(repository)

    assert not report.ok
    assert any("README.md" in error for error in report.errors)


@pytest.mark.parametrize(
    ("name", "contents"), [("sample.bin", b"example"), ("policy.txt", b"before\x00after")]
)
def test_pack_rejects_unsupported_or_nontext_files(
    repository: Path, name: str, contents: bytes
) -> None:
    rule = _add_rule(repository)
    (rule.parent / name).write_bytes(contents)

    report = validate_repository(repository)

    assert not report.ok
    assert any(name in error for error in report.errors)


def test_symlinked_rules_directory_is_rejected(repository: Path) -> None:
    _add_rule(repository)
    rules = repository / "rules"
    other = rules.rename(repository / "otherwise-valid-rules")
    rules.symlink_to(other, target_is_directory=True)

    report = validate_repository(repository)

    assert not report.ok
    assert any("rules: expected a non-symlink directory" in error for error in report.errors)


def test_symlinked_rule_resource_is_rejected(repository: Path) -> None:
    rule = _add_rule(repository)
    (rule.parent / "linked.txt").symlink_to(rule.with_name("README.md"))

    report = validate_repository(repository)

    assert not report.ok
    assert any("linked.txt" in error for error in report.errors)


def test_misplaced_rule_is_rejected(repository: Path) -> None:
    rule = _add_rule(repository)
    rule.rename(rule.with_name("other_name.rule"))

    report = validate_repository(repository)

    assert not report.ok
    assert report.compiled_rules == 0
    assert any("expected core/<pack>/<stem>/<stem>.rule" in error for error in report.errors)


def test_duplicate_rule_names_are_rejected(repository: Path) -> None:
    _add_rule(repository, name="shared name")
    _add_rule(repository, pack="core/another-pack", name="shared name")

    report = validate_repository(repository)

    assert not report.ok
    assert any("duplicate declared rule name" in error for error in report.errors)


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"author": ""}, "required metadata 'author'"),
        ({"revision": ""}, "required metadata 'revision'"),
        ({"requires_quarry": ">=99.0.0"}, "compile failed"),
    ],
)
def test_invalid_metadata_is_rejected(
    repository: Path, metadata: dict[str, str], error: str
) -> None:
    _add_rule(repository, metadata=metadata)

    report = validate_repository(repository)

    assert not report.ok
    assert report.compiled_rules == 0
    assert any(error in item for item in report.errors)


def test_missing_revision_is_rejected(repository: Path) -> None:
    rule = _add_rule(repository)
    lines = rule.read_text(encoding="utf-8").splitlines(keepends=True)
    rule.write_text(
        "".join(line for line in lines if not line.startswith("    revision = ")), encoding="utf-8"
    )

    report = validate_repository(repository)

    assert not report.ok
    assert report.compiled_rules == 0
    assert any("required metadata 'revision'" in error for error in report.errors)


def test_rules_compile_without_a_scanner_or_model_backend(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_rule(repository)
    model_rule = _add_rule(
        repository,
        pack="core/model-pack",
        condition='model("unconfigured").score(messages, text_ref("guidance")) > 0.5',
    )
    resources = model_rule.parent / "resources"
    resources.mkdir()
    (resources / "guidance.txt").write_text("Judge the example marker.\n", encoding="utf-8")

    def fail_if_loaded(*args: object, **kwargs: object) -> None:
        pytest.fail("Rule-pack validation must compile without initializing runtime services")

    monkeypatch.setattr(ModelInferenceBackend, "__init__", fail_if_loaded)
    monkeypatch.setattr(DSLScanner, "__init__", fail_if_loaded)
    report = validate_repository(repository)

    assert report.ok, report.errors
    assert report.compiled_rules == 2

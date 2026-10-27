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

"""Generate or verify Quarry's packaged language catalog and examples."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from quarry.language.generate import write_language_outputs

ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_RELATIVE = Path("quarry/language")


def _output_paths(root: Path) -> set[Path]:
    language = root / LANGUAGE_RELATIVE
    examples = language / "examples"
    return {
        LANGUAGE_RELATIVE / "spec.json",
        LANGUAGE_RELATIVE / "reference.md",
        *(path.relative_to(root) for path in examples.iterdir() if path.is_file()),
    }


def _check() -> int:
    with tempfile.TemporaryDirectory(prefix="quarry-language-") as temporary:
        generated_root = Path(temporary)
        (generated_root / LANGUAGE_RELATIVE).mkdir(parents=True)
        write_language_outputs(generated_root)
        generated = _output_paths(generated_root)
        current = _output_paths(ROOT)
        stale = sorted(
            path
            for path in current | generated
            if path not in current
            or path not in generated
            or (ROOT / path).read_bytes() != (generated_root / path).read_bytes()
        )
        if stale:
            for path in stale:
                print(f"out of date: {path}")
            return 1
    print("Quarry language artifacts are current")
    return 0


def main() -> int:
    """Generate or verify Quarry's packaged language catalog and examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify generated files without editing"
    )
    args = parser.parse_args()
    if args.check:
        return _check()
    write_language_outputs(ROOT)
    print("Generated Quarry language catalog, reference, and examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

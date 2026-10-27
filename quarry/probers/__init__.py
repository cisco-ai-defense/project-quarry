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

"""Prober / classifier backends that ground raw trace content into semantic predicates.

Agent-drift signatures need *semantic* predicates the structural DSL can't express by
pattern-matching alone — "is this action off-topic vs. the user's original intent?",
"what capability class does this tool call exercise?". These are produced by **probers**:
small, swappable grounders that turn text into a boolean/score.

Probers plug into the engine through the same injectable external-detector seam that
``model_score`` / ``model_scan`` / ``yara_scan`` use (see ``quarry/scanner.py`` and
``quarry/engine/oracle.py``). The DSL exposes them as external functions
(``topic_drift`` / ``off_topic`` / ``capability_tags`` in ``quarry/dsl/functions.py``);
the compiler lowers each to a cost-tiered ``external`` atom; the scanner's detector
bridge routes the atom to the live prober through the rule's ``ScanContext``.

The backends here are **deterministic stubs** with zero ML dependencies. They are
installed only when config explicitly sets ``allow_stub_probers: true``; otherwise
drift rules fail to load without registered backends. Each stub can be replaced by a
real backend (embedding model, transformer classifier) on the ``ResourceManager`` —
no DSL, compiler, or rule change required.
"""

from quarry.probers.capability import CapabilityClassifier
from quarry.probers.topic import TopicProber

__all__ = ["CapabilityClassifier", "TopicProber"]

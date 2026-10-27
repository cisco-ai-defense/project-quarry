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

"""Parser for the DSL"""

import re

from lark import Lark
from lark.exceptions import GrammarError, UnexpectedInput, UnexpectedToken, VisitError

from quarry.core.logging import get_app_logger
from quarry.dsl.grammar import DSL_GRAMMAR
from quarry.dsl.transformer import DSLTransformer
from quarry.types.exceptions import RuleParseError
from quarry.types.rules import DSLRule

logger = get_app_logger()


class DSLParser:
    """Parse DSL text into tree and transform into Rule object"""

    def __init__(self):
        self.parser = Lark(DSL_GRAMMAR, parser="lalr")
        self.transformer = DSLTransformer()

    def parse(self, rule_text: str, *, source: str | None = None) -> DSLRule:
        """Parse DSL text -> tree -> Rule object"""
        try:
            tree = self.parser.parse(rule_text)
            rule = self.transformer.transform(tree)
            return rule
        except (GrammarError, UnexpectedToken, UnexpectedInput) as e:
            line = getattr(e, "line", None)
            logger.error(f"[ERROR] Rule grammar failed: error_type={type(e).__name__} line={line}")
            message = _single_equals_message(e, rule_text) or str(e)
            raise RuleParseError(
                rule_name="<unknown>", line=line, message=message, source=source
            ) from e
        except Exception as e:
            logger.error(f"[ERROR] Rule parse failed: error_type={type(e).__name__}")
            message = _NESTED_TOO_DEEPLY if _exceeds_recursion_limit(e) else str(e)
            raise RuleParseError(
                rule_name="<unknown>", line=None, message=message, source=source
            ) from e


_NESTED_TOO_DEEPLY = (
    "the rule is nested too deeply to parse (parentheses or comprehensions nested more than "
    "about 50 levels deep, or more than about 200 `and`/`or` terms in one `variables:` "
    "binding); split it into several rules"
)

_KEYWORD_ARGUMENT = re.compile(r"[(,]\s*([A-Za-z_]\w*)\s*$")


def _single_equals_message(error: Exception, rule_text: str) -> str | None:
    """Explain a stray `=` outside metadata, strings, and variable definitions."""
    if not isinstance(error, UnexpectedToken) or error.token.type != "EQUAL":
        return None
    where = f"line {error.line}, column {error.column}"
    keyword = _KEYWORD_ARGUMENT.search(rule_text[: error.token.start_pos])
    if keyword:
        return (
            f"unexpected `=` after `{keyword.group(1)}` at {where}: function arguments are "
            "positional; pass the value without a name"
        )
    return (
        f"unexpected `=` at {where}: use `==` to compare values; `=` only defines values "
        "in `metadata:`, `strings:`, and `variables:`"
    )


def _exceeds_recursion_limit(error: Exception) -> bool:
    """Recognize recursion failure directly or through Lark's transformer wrapper."""
    if isinstance(error, VisitError):
        return isinstance(error.orig_exc, RecursionError)
    return isinstance(error, RecursionError)

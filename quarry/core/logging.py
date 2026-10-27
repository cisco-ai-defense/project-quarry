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

"""Privacy-aware application logging configuration."""

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


def setup_logging():
    """Send application logs to stderr without creating persistent files."""

    logger.remove()

    def _ensure_channel(record: "Record") -> None:
        record["extra"].setdefault("channel", "app")

    logger.configure(patcher=_ensure_channel)
    logger.add(
        sink=sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> "
            "| <magenta>{extra[channel]}</magenta> - <level>{message}</level>"
        ),
        colorize=True,
    )


def get_app_logger():
    """Return the application logger."""

    return logger.bind(channel="app")

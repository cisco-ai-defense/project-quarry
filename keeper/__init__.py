"""The Keeper: a guard and organizing system for AI agents, built on the Quarry engine.

Phase 1: subjects (one desk per subject), standing rules, a library of opt-in checks,
and a guard that checks a conversation and fails closed with an alert. Offline only.
"""

from keeper.guard import CheckReport, check, check_file
from keeper.layout import add_subject, init_root, list_subjects, load_subject

__all__ = [
    "CheckReport",
    "add_subject",
    "check",
    "check_file",
    "init_root",
    "list_subjects",
    "load_subject",
]

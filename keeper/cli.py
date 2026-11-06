"""keeper: command line for the Keeper (phase 1).

    keeper init <folder>
    keeper subject add <folder> <name> [--description TEXT] [--use CHECK ...]
    keeper subjects <folder>
    keeper rules
    keeper check <folder> <subject> <conversation.json> [--json] [--no-log]

Exit codes for `check`: 0 PASS, 1 STOP (a rule broke), 2 STOP (the check could not finish).
Every other command exits 0 on success and 2 on any error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from keeper.guard import check_file, write_log
from keeper.layout import (
    LayoutError,
    add_subject,
    init_root,
    library_rules,
    list_subjects,
    standing_rules,
)


def _rule_summary(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("plain ="):
            return stripped.split("=", 1)[1].strip().strip('"')
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keeper", description="Keep AI agents on their rules.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a Keeper folder")
    init.add_argument("folder", type=Path)

    subject = commands.add_parser("subject", help="manage subjects")
    subject_commands = subject.add_subparsers(dest="subject_command", required=True)
    add = subject_commands.add_parser("add", help="add a subject")
    add.add_argument("folder", type=Path)
    add.add_argument("name")
    add.add_argument("--description", default="")
    add.add_argument(
        "--use", nargs="*", default=[], metavar="CHECK", help="library checks to turn on"
    )

    subjects = commands.add_parser("subjects", help="list subjects")
    subjects.add_argument("folder", type=Path)

    commands.add_parser("rules", help="list standing rules and library checks")

    run = commands.add_parser("check", help="check a conversation against a subject's rules")
    run.add_argument("folder", type=Path)
    run.add_argument("subject")
    run.add_argument("conversation", type=Path)
    run.add_argument("--json", action="store_true", help="print the full report as JSON")
    run.add_argument("--no-log", action="store_true", help="don't write to the daily log")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            root = init_root(args.folder)
            print(f"Keeper folder ready: {root}")
        elif args.command == "subject":
            made = add_subject(args.folder, args.name, description=args.description, use=args.use)
            print(f"Subject added: {made.path}")
        elif args.command == "subjects":
            for item in list_subjects(args.folder):
                print(f"{item.name}: {item.description or '(no description)'}")
                print(f"  library checks: {', '.join(item.use) or 'none'}")
                print(f"  own checks: {len(item.custom_checks())}")
        elif args.command == "rules":
            print("Standing rules (always on):")
            for path in standing_rules():
                print(f"  {path.stem}: {_rule_summary(path)}")
            print("Library checks (turn on per subject with --use):")
            for name, path in library_rules().items():
                print(f"  {name}: {_rule_summary(path)}")
        elif args.command == "check":
            report = check_file(args.folder, args.subject, args.conversation)
            if not args.no_log:
                try:
                    write_log(args.folder, report)
                except OSError as error:
                    print(f"(could not write the log: {error})", file=sys.stderr)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(report.alert_text())
            return report.exit_code
    except LayoutError as error:
        print(
            f"ALERT: {error}\n\nIn simple terms: the Keeper couldn't find or set up what you asked for.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

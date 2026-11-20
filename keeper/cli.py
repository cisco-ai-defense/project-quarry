"""keeper: command line for the Keeper (phase 1).

    keeper init <folder>
    keeper subject add <folder> <name> [--description TEXT] [--use CHECK ...]
    keeper subjects <folder>
    keeper rules
    keeper check <folder> <subject> <conversation.json> [--json] [--no-log]
    keeper job start <folder> <subject> <title> --kind KIND --session NAME [...]
    keeper job step|record|reject|accept <folder> <subject> <job> [...]
    keeper ghost <folder> <kind>
    keeper rejections list|forget <folder> [...]

Exit codes for `check`: 0 PASS, 1 STOP (a rule broke), 2 STOP (the check could not finish).
Every other command exits 0 on success and 2 on any error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from keeper import logbook
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

    job = commands.add_parser("job", help="the logbook: start, record, reject, accept")
    job_commands = job.add_subparsers(dest="job_command", required=True)
    start = job_commands.add_parser("start", help="start a job and print its ghost lap")
    start.add_argument("folder", type=Path)
    start.add_argument("subject")
    start.add_argument("title")
    start.add_argument("--kind", required=True, help="kind of job, e.g. sprite-walk-cycle")
    start.add_argument("--session", required=True, help="short session name for rejections")
    start.add_argument("--ask", default="", help="what Al asked, in his words")
    start.add_argument("--keywords", nargs="*", default=[])
    start.add_argument("--refs", nargs="*", default=[], help="files in the subject's references/")
    start.add_argument("--new-direction", default="", help="lift the ghost lap for this job")
    step = job_commands.add_parser("step", help="record one step")
    step.add_argument("folder", type=Path)
    step.add_argument("subject")
    step.add_argument("job")
    step.add_argument("--tool", required=True)
    step.add_argument("--input", default="")
    step.add_argument("--output", default="")
    step.add_argument("--note", default="")
    record = job_commands.add_parser("record", help="record every tool call in a conversation")
    record.add_argument("folder", type=Path)
    record.add_argument("subject")
    record.add_argument("job")
    record.add_argument("conversation", type=Path)
    rejected = job_commands.add_parser("reject", help="Al rejects the current try")
    rejected.add_argument("folder", type=Path)
    rejected.add_argument("subject")
    rejected.add_argument("job")
    rejected.add_argument("--why", required=True, help="the reason, in Al's words")
    rejected.add_argument("--summary", default="", help="short name for the wrong direction")
    accepted = job_commands.add_parser("accept", help="Al accepts the work: write SUCCESS.md")
    accepted.add_argument("folder", type=Path)
    accepted.add_argument("subject")
    accepted.add_argument("job")
    accepted.add_argument("--note", default="")
    accepted.add_argument("--better", action="store_true", help="Al likes it better: new basis")

    ghost = commands.add_parser("ghost", help="print the ghost lap for a kind of job")
    ghost.add_argument("folder", type=Path)
    ghost.add_argument("kind")
    ghost.add_argument("--keywords", nargs="*", default=[])

    rejections = commands.add_parser("rejections", help="list or clear rejected tries")
    rejection_commands = rejections.add_subparsers(dest="rejection_command", required=True)
    listing = rejection_commands.add_parser("list", help="list rejections")
    listing.add_argument("folder", type=Path)
    listing.add_argument("--session", default=None)
    forget = rejection_commands.add_parser("forget", help="delete rejections (Al only)")
    forget.add_argument("folder", type=Path)
    forget.add_argument("session_folder", help="e.g. 2026-11-06-hero-walk-sprites")
    forget.add_argument("files", nargs="*", help="only these files; default is the whole folder")
    forget.add_argument("--yes", action="store_true", help="skip the typed confirmation")

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
        elif args.command == "job":
            return _job(args)
        elif args.command == "ghost":
            print(logbook.ghost_lap(args.folder, args.kind, keywords=args.keywords).to_text())
        elif args.command == "rejections":
            return _rejections(args)
    except (LayoutError, logbook.LogbookError) as error:
        print(
            f"ALERT: {error}\n\nIn simple terms: the Keeper couldn't find or set up what you asked for.",
            file=sys.stderr,
        )
        return 2
    return 0


def _job(args: argparse.Namespace) -> int:
    if args.job_command == "start":
        made, lap = logbook.start_job(
            args.folder,
            args.subject,
            args.title,
            kind=args.kind,
            session=args.session,
            ask=args.ask,
            keywords=args.keywords,
            references=args.refs,
            new_direction=args.new_direction,
        )
        print(f"Job started: {made.id}\n")
        print(lap.to_text())
    elif args.job_command == "step":
        logbook.record_step(
            args.folder,
            args.subject,
            args.job,
            tool=args.tool,
            input=args.input,
            output=args.output,
            note=args.note,
        )
        print("Step recorded.")
    elif args.job_command == "record":
        try:
            conversation = json.loads(args.conversation.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise logbook.LogbookError(
                f"The conversation file could not be read: {error}"
            ) from error
        count = logbook.record_conversation(args.folder, args.subject, args.job, conversation)
        print(f"Recorded {count} step(s).")
    elif args.job_command == "reject":
        path = logbook.reject(
            args.folder, args.subject, args.job, why=args.why, summary=args.summary
        )
        print(f"Rejection saved: {path}\nThe next try has started.")
    elif args.job_command == "accept":
        success, basis = logbook.accept(
            args.folder, args.subject, args.job, note=args.note, better=args.better
        )
        print(f"SUCCESS written: {success}")
        if basis:
            print(f"New basis for this kind of job: {basis}")
        else:
            print("The existing basis was kept. Use --better if you like this one more.")
    return 0


def _rejections(args: argparse.Namespace) -> int:
    if args.rejection_command == "list":
        items = logbook.list_rejections(args.folder, session=args.session)
        if not items:
            print("(no rejections on file)")
        for item in items:
            print(f"{item['file']}  [{item.get('kind', '?')}] {item.get('why', '')}")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise logbook.LogbookError("Forgetting needs --yes or a typed confirmation.")
        typed = input(f"Type the session folder name to delete ({args.session_folder}): ")
        if typed.strip() != args.session_folder:
            print("Nothing deleted.")
            return 2
    removed = logbook.forget_rejections(args.folder, args.session_folder, args.files)
    print(f"Deleted {len(removed)} rejection file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

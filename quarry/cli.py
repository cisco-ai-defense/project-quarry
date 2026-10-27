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

"""
Quarry command line tool

https://github.com/cisco-ai-defense/project-quarry

Example usage:
- quarry scan -c config.yaml -t data/  # Scan conversations
"""

import argparse
import csv
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path

import orjson
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from quarry import __version__
from quarry.config import QuarryConfig
from quarry.safety.quality import quality_warnings
from quarry.safety.validator import Complexity, Severity, ValidationWarning, validate_rules
from quarry.scanner import DSLScanner
from quarry.types.results import CollectionResult, Span
from quarry.types.rules import ScanResult

BANNER = f"Project Quarry v{__version__}\nAI detection DSL"

# Process exit codes. INCOMPLETE (2) is distinct from CLEAN/MATCH so a caller can tell a
# scan that could not authoritatively complete (invalid input, a rule that errored/timed
# out, config/IO failure) from a genuine clean pass — the CLI fails closed too, not just
# the SDK guard.
EXIT_CLEAN = 0
EXIT_MATCH = 1
EXIT_INCOMPLETE = 2


def _typed_output_tree(
    output: CollectionResult,
    *,
    include_items: bool,
) -> Tree:
    """Render one typed collection as a summary with optional item details."""

    count = len(output.items)
    category = f" [{output.category}]" if output.category else ""
    noun = "item" if count == 1 else "items"
    tree = Tree(Text(f"output: {output.output_type}{category} ({count} {noun})"))
    if not include_items:
        return tree

    for item in output.items:
        value = item.value
        if isinstance(value, Span):
            tree.add(
                Text(
                    f"message {value.message_index}, characters {value.start}:{value.end}, "
                    f"score {value.score.value:g}"
                )
            )
        else:
            tree.add(Text(orjson.dumps(value.to_dict()).decode()))
    return tree


def create_timing_bar_chart(timings: dict[str, list[float]], max_width: int = 40) -> Table:
    """Create text-based bar chart for timing data."""
    if not timings:
        return Table()

    aggregates = []
    for rule_name, times in timings.items():
        avg_time = sum(times) / len(times)
        max_time = max(times)
        call_count = len(times)
        aggregates.append((rule_name, avg_time, max_time, call_count))

    aggregates.sort(key=lambda x: x[1], reverse=True)
    max_avg = max(agg[1] for agg in aggregates) if aggregates else 1.0

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Rule", style="white", no_wrap=False, max_width=35)
    table.add_column("Avg (ms)", justify="right", style="cyan", width=10)
    table.add_column("Max (ms)", justify="right", style="yellow", width=10)
    table.add_column("Calls", justify="right", style="dim", width=7)
    table.add_column("Performance", width=max_width)

    for rule_name, avg_time, max_time, call_count in aggregates:
        avg_ms = avg_time * 1000
        max_ms = max_time * 1000

        bar_length = int((avg_time / max_avg) * max_width) if max_avg > 0 else 0
        bar_length = max(1, bar_length)

        if avg_ms < 10:
            bar_style = "dim"
        elif avg_ms < 100:
            bar_style = "yellow"
        else:
            bar_style = "red"

        text_bar = Text("█" * bar_length, style=bar_style)

        table.add_row(
            rule_name,
            f"{avg_ms:.2f}",
            f"{max_ms:.2f}",
            str(call_count),
            text_bar,
        )

    return table


def format_results(
    results: list[ScanResult],
    format_type: str = "human",
    total_rules: int = 0,
    scanner=None,
    show_timing: bool = False,
    explain_mode: bool = False,
) -> str:
    """Format scan results for output."""
    if format_type == "json":
        output = []
        for result in results:
            matches = []
            for rule_name in result.rule_matches:
                match = {"rule": rule_name}
                typed_output = result.outputs.get(rule_name)
                if typed_output is not None:
                    match["output"] = typed_output.to_dict()
                matches.append(match)
            result_dict = {
                "file": str(result.file_path),
                "matches": matches,
                "scan_time": result.scan_time,
                "completion_state": result.completion_state,
                "complete": result.is_complete,
                "error": result.error,
                "validation_error": result.validation_error,
                "evaluation_errors": result.evaluation_errors,
                "was_truncated": result.was_truncated,
                "truncation_info": result.truncation_info,
                "timed_out_rules": result.timed_out_rules,
                "skipped_rules": result.skipped_rules,
                "evaluation_warnings": result.evaluation_warnings,
            }

            if result.detailed_matches:
                result_dict["detailed_matches"] = [
                    match.to_dict() for match in result.detailed_matches
                ]
            output.append(result_dict)

        return orjson.dumps(output, option=orjson.OPT_INDENT_2).decode()

    if format_type == "csv":
        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(
            [
                "file",
                "rule",
                "scan_time",
                "error",
                "captures",
                "output",
                "completion_state",
                "validation_error",
                "evaluation_errors",
                "was_truncated",
                "truncation_info",
                "timed_out_rules",
                "skipped_rules",
                "evaluation_warnings",
            ]
        )

        for result in results:
            completion_cells = [
                result.completion_state,
                result.validation_error or "",
                orjson.dumps(result.evaluation_errors).decode() if result.evaluation_errors else "",
                str(result.was_truncated).lower(),
                orjson.dumps(result.truncation_info).decode() if result.truncation_info else "",
                orjson.dumps(result.timed_out_rules).decode() if result.timed_out_rules else "",
                orjson.dumps(result.skipped_rules).decode() if result.skipped_rules else "",
                orjson.dumps(result.evaluation_warnings).decode()
                if result.evaluation_warnings
                else "",
            ]
            if result.rule_matches:
                evidence_by_rule = {}
                if explain_mode and result.detailed_matches:
                    for match in result.detailed_matches:
                        evidence_by_rule[match.rule_name] = match.to_dict()

                for rule_name in result.rule_matches:
                    evidence_json = ""
                    if rule_name in evidence_by_rule:
                        evidence_json = orjson.dumps(evidence_by_rule[rule_name]).decode()
                    typed_output = result.outputs.get(rule_name)
                    output_json = (
                        orjson.dumps(typed_output.to_dict()).decode() if typed_output else ""
                    )

                    writer.writerow(
                        [
                            str(result.file_path),
                            rule_name,
                            result.scan_time,
                            result.error or "",
                            evidence_json,
                            output_json,
                            *completion_cells,
                        ]
                    )
            else:
                writer.writerow(
                    [
                        str(result.file_path),
                        "",
                        result.scan_time,
                        result.error or "",
                        "",
                        "",
                        *completion_cells,
                    ]
                )

        return output.getvalue()

    console = Console()
    total_matches = sum(len(r.rule_matches) for r in results)
    files_with_matches = sum(1 for r in results if r.rule_matches)
    incomplete_files = sum(1 for r in results if not r.is_complete)

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="cyan", no_wrap=True)
    stats.add_column(style="white")
    stats.add_row("Total rules:", str(total_rules))
    stats.add_row("Files scanned:", str(len(results)))
    stats.add_row("Files with matches:", str(files_with_matches))
    stats.add_row("Total rule matches:", str(total_matches))
    if incomplete_files:
        stats.add_row("Invalid/incomplete files:", str(incomplete_files))

    console.print()
    console.print(Panel(stats, title="Scan Summary", border_style="blue"))
    console.print()

    if show_timing:
        all_timings: dict[str, list[float]] = {}
        for result in results:
            if result.rule_timings:
                for rule_name, timing in result.rule_timings.items():
                    all_timings.setdefault(rule_name, []).append(timing)

        if all_timings:
            timing_table = create_timing_bar_chart(all_timings)
            console.print(Panel(timing_table, title="Rule Performance", border_style="yellow"))
            console.print()

    for result in results:
        # Surface non-authoritative scans distinctly: INVALID (bad input) / INCOMPLETE (a
        # rule errored or timed out). A partial match list must never read as a clean pass.
        if not result.is_complete:
            label = "INVALID" if result.completion_state == "invalid" else "INCOMPLETE"
            console.print(f"[red][{label}][/red] {result.file_path}")
            for issue in result.completion_issues:
                console.print(f"  - {issue}")
            if result.rule_matches:
                console.print(
                    "  [yellow]Partial matches (not an authoritative decision):[/yellow] "
                    + ", ".join(result.rule_matches)
                )
            console.print()
            continue

        if result.rule_matches:
            console.print(
                f"[green][MATCH][/green] {result.file_path} ({len(result.rule_matches)} matches)"
            )

            is_batch_mode = any(":" in r and r.startswith("conv_") for r in result.rule_matches)

            if is_batch_mode:
                conv_groups: dict[str, list[tuple[str, str]]] = {}
                for rule_name in result.rule_matches:
                    if ":" in rule_name and rule_name.startswith("conv_"):
                        conv_key, actual_rule_name = rule_name.split(":", 1)
                        if conv_key not in conv_groups:
                            conv_groups[conv_key] = []
                        conv_groups[conv_key].append((actual_rule_name, rule_name))
                    else:
                        if "other" not in conv_groups:
                            conv_groups["other"] = []
                        conv_groups["other"].append((rule_name, rule_name))

                for conv_key in sorted(conv_groups.keys()):
                    tree = Tree(f"[bold cyan]{conv_key}[/bold cyan]")
                    for rule_name, output_key in conv_groups[conv_key]:
                        rule_tree = tree.add(rule_name)
                        typed_output = result.outputs.get(output_key)
                        if typed_output is not None:
                            rule_tree.add(
                                _typed_output_tree(
                                    typed_output,
                                    include_items=explain_mode,
                                )
                            )
                    console.print(Padding(tree, (0, 0, 0, 2)))

            else:
                detailed_match_map = {}
                if explain_mode and result.detailed_matches:
                    for match in result.detailed_matches:
                        detailed_match_map[match.rule_name] = match

                for rule_name in result.rule_matches:
                    if scanner:
                        rule = scanner.loaded_rules.get(rule_name)
                        if rule:
                            description = rule.metadata.get("description", "")
                            source = f" ({rule.source_file.name})" if rule.source_file else ""
                            console.print(f"  [yellow]\\[{rule_name}][/yellow]{source}")

                            if description:
                                console.print(f"    [bold]description:[/bold] {description}")

                            if explain_mode and rule_name in detailed_match_map:
                                evidence = detailed_match_map[rule_name]
                                if (
                                    evidence.steps
                                    or evidence.captured_value
                                    or evidence.model_results
                                ):
                                    tree = Tree("[cyan]Why it matched[/cyan]")
                                    if evidence.formula_note:
                                        tree.add(f"[dim]reason:[/dim] {evidence.formula_note}")
                                    if evidence.captured_value is not None:
                                        tree.add(
                                            f'[dim]captured:[/dim] "{evidence.captured_value}"'
                                        )
                                    if evidence.steps:
                                        w = tree.add("witness (load-bearing turns)")
                                        for step in evidence.steps:
                                            label = f"[{step.position}] {step.source_path}"
                                            if step.excerpt:
                                                label += f': "{step.excerpt}"'
                                            w.add(label)
                                        if evidence.omitted_steps:
                                            w.add(f"[dim]… {evidence.omitted_steps} more[/dim]")
                                    if evidence.model_results:
                                        model_tree = tree.add("model results")
                                        for model_result in evidence.model_results:
                                            aliases = ", ".join(model_result.models)
                                            label = Text(
                                                f"{model_result.operation} ({aliases}): "
                                                f"{model_result.expression_value}"
                                            )
                                            if model_result.operator is not None:
                                                label.append(
                                                    f" {model_result.operator} "
                                                    f"{model_result.threshold}"
                                                )
                                            model_tree.add(label)

                                    console.print(Padding(tree, (0, 0, 0, 4)))
                        else:
                            console.print(f"  [yellow]\\[{rule_name}][/yellow]")
                    else:
                        console.print(f"  {rule_name}")

                    typed_output = result.outputs.get(rule_name)
                    if typed_output is not None:
                        console.print(
                            Padding(
                                _typed_output_tree(
                                    typed_output,
                                    include_items=explain_mode,
                                ),
                                (0, 0, 0, 4),
                            )
                        )

            console.print()

    return ""


def scan_command(args):
    """Handle the scan subcommand."""
    # The human report prints while it is formatted, so it cannot be redirected to a file.
    if args.output and args.output_format == "human":
        print(
            "[ERROR] --output is only supported with json or csv output formats",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE

    if args.output_format == "human":
        print(BANNER)
    else:
        print(BANNER, file=sys.stderr)

    try:
        config = QuarryConfig.from_yaml(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return EXIT_INCOMPLETE

    limits = config.get_safety_limits()
    if args.no_limits:
        limits = replace(limits, enforce_limits=False)

    try:
        scanner = DSLScanner(
            config=config,
            recursive=args.recursive,
            verbose=args.verbose,
            limits=limits,
            batch_mode=args.batch,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize scanner: {e}", file=sys.stderr)
        return EXIT_INCOMPLETE

    if not scanner.loaded_rules:
        print("[ERROR] No valid rules loaded", file=sys.stderr)
        return EXIT_INCOMPLETE

    if args.validate_only:
        validation_results = validate_rules(scanner.loaded_rules)
        validator_warning_count = sum(len(r.warnings) for r in validation_results.values())
        advisory_by_rule: dict[str, list[ValidationWarning]] = {}
        for rule_name, rule in scanner.loaded_rules.items():
            try:
                advisories = list(quality_warnings(rule, scanner.compiled_rules[rule_name]))
                for warning in advisories:
                    # Advisory analysis is best effort, including its returned shape.
                    _ = warning.severity.value, warning.message, warning.suggestion
                advisory_by_rule[rule_name] = advisories
            except Exception:
                advisory_by_rule[rule_name] = []
        total_warnings = validator_warning_count + sum(
            len(warnings) for warnings in advisory_by_rule.values()
        )
        # Existing validator warnings still block; medium quality advisories do not.
        blocking_warnings = validator_warning_count + sum(
            warning.severity in {Severity.HIGH, Severity.EXTREME}
            for warnings in advisory_by_rule.values()
            for warning in warnings
        )
        warnings_summary: list[str] = []

        for rule_name, result in validation_results.items():
            rule = scanner.loaded_rules[rule_name]

            complexity_colors = {
                Complexity.LOW: "\033[92m",
                Complexity.MEDIUM: "\033[93m",
                Complexity.HIGH: "\033[91m",
                Complexity.EXTREME: "\033[95m",
            }
            color = complexity_colors.get(result.estimated_complexity, "")
            reset = "\033[0m"

            print(f"{color}● {rule_name}{reset}")
            print(f"  Complexity: {result.estimated_complexity.value.upper()}")
            print(f"  Source: {rule.source_file}")

            rule_warnings = [*result.warnings, *advisory_by_rule[rule_name]]
            if rule_warnings:
                print("  Warnings: (see summary below)")
                for w in rule_warnings:
                    warnings_summary.append(
                        f"[{w.severity.value.upper()}] {rule_name}: {w.message} -> {w.suggestion}"
                    )
            else:
                print("  No issues found")

            print()

        print(f"\n{'=' * 60}")
        print(f"Validated {len(validation_results)} rules")
        print(f"Total warnings: {total_warnings}")
        if total_warnings == 0:
            print("All rules passed validation")
        elif blocking_warnings == 0:
            print("No blocking warnings")
        print(f"{'=' * 60}\n")

        if warnings_summary:
            print("Warning Summary:")
            for item in warnings_summary:
                print(f"  {item}")
            print()

        return 1 if blocking_warnings else 0

    results = []
    if args.target.is_file():
        results = [scanner.scan_file(args.target, enable_explain=args.explain)]
    elif args.target.is_dir():
        pattern = "**/*.json" if args.recursive else "*.json"
        files = sorted(args.target.glob(pattern))

        if not files:
            print(f"[ERROR] No JSON files found in {args.target}", file=sys.stderr)
            return EXIT_INCOMPLETE

        # For JSON/CSV, write progress to stderr so stdout can be used for piping
        console = Console(stderr=True) if args.output_format != "human" else Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Scanning {len(files)} files...", total=len(files))

            for json_file in files:
                progress.console.print(f"[dim]├─[/dim] {json_file.name}")
                result = scanner.scan_file(json_file, enable_explain=args.explain)
                results.append(result)
                progress.update(task, advance=1)
    else:
        print(f"[ERROR] Target not found: {args.target}", file=sys.stderr)
        return EXIT_INCOMPLETE

    formatted_output = format_results(
        results,
        args.output_format,
        len(scanner.loaded_rules),
        scanner,
        args.show_timing,
        args.explain,
    )

    if args.output:
        if args.output.exists() and args.output.is_dir():
            print(
                f"[ERROR] Output path is a directory: {args.output}",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE

        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(formatted_output)
        except OSError as exc:
            print(
                f"[ERROR] Failed to write output file {args.output}: {exc}",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE

        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(formatted_output)
    # Fail closed: any non-authoritative file makes the whole run INCOMPLETE, regardless of
    # matches — a caller keying on exit status must not read an incomplete scan as clean.
    if any(not result.is_complete for result in results):
        return EXIT_INCOMPLETE
    total_matches = sum(len(r.rule_matches) for r in results)
    return EXIT_MATCH if total_matches > 0 else EXIT_CLEAN


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan conversations with rules
  quarry scan -c config.yaml -t conversations/

  # Scan batch files (arrays of conversations)
  quarry scan -c config.yaml -t dataset/ --batch
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan conversations with rules",
        description="Scan conversations with rules",
    )
    scan_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file (controls rules, models, timeouts, and limits)",
    )
    scan_parser.add_argument(
        "-t",
        "--target",
        type=Path,
        required=True,
        help="Target JSON file or directory to scan",
    )
    scan_parser.add_argument(
        "--recursive", action="store_true", help="Recursively scan directories"
    )
    scan_parser.add_argument(
        "-o",
        "--output-format",
        choices=["human", "json", "csv"],
        default="human",
        help="Output format (default: human)",
    )
    scan_parser.add_argument(
        "--output",
        type=Path,
        help="Write json or csv results to the specified file instead of stdout",
    )
    scan_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    scan_parser.add_argument(
        "--show-timing", action="store_true", help="Show per-rule execution timing"
    )
    scan_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate rules without scanning",
    )
    scan_parser.add_argument(
        "--explain",
        action="store_true",
        help="Enable detailed match tracing and capture information",
    )
    scan_parser.add_argument(
        "--no-limits", action="store_true", help="Disable safety limits (for testing)"
    )
    scan_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process JSON files as arrays of conversations (one conversation per array element)",
    )
    scan_parser.set_defaults(func=scan_command)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())

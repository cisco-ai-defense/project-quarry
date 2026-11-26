"""The guard: check one conversation against a subject's rules and fail closed with an alert.

Verdicts:
- PASS: every rule was checked and none broke.
- STOP (rule): one or more rules broke. The alert says which, where, and why.
- STOP (error): the check could not finish (bad input, timeout, broken rule, network
  attempt). There is no "probably fine": an unfinished check is a stop.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from keeper.layout import Subject, library_rules, load_subject, standing_rules
from keeper.offline import network_lock, require_offline_config

Verdict = Literal["PASS", "STOP"]
StopKind = Literal["none", "rule", "error"]

_MAX_EXCERPT = 160

# Never echo a secret back in an alert. Same shapes as the "Secret in output" rule.
_SECRET = re.compile(
    r"(sk-ant-[A-Za-z0-9_-]{8,}|sk-(proj-)?[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{8,}"
    r"|github_pat_[A-Za-z0-9_]{8,}|AKIA[0-9A-Z]{8,}|xox[abpr]-[A-Za-z0-9-]{6,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


@dataclass
class BrokenRule:
    rule: str
    severity: str
    description: str
    plain: str
    where: list[str] = field(default_factory=list)


@dataclass
class CheckReport:
    subject: str
    verdict: Verdict
    stop_kind: StopKind
    rules_loaded: list[str]
    message_count: int = 0
    tool_call_count: int = 0
    broken: list[BrokenRule] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def exit_code(self) -> int:
        return {"none": 0, "rule": 1, "error": 2}[self.stop_kind]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exit_code"] = self.exit_code
        data["alert"] = self.alert_text() if self.verdict == "STOP" else None
        return data

    def alert_text(self) -> str:
        """Al's fail-closed format: alert line, step-by-step breakdown, In simple terms."""

        if self.verdict == "PASS":
            return (
                f'PASS: subject "{self.subject}" kept all {len(self.rules_loaded)} rules '
                f"({self.message_count} messages, {self.tool_call_count} tool calls checked)."
            )
        steps: list[str] = []
        if self.stop_kind == "rule":
            names = ", ".join(item.rule for item in self.broken)
            alert = (
                f'ALERT: Keeper stopped this step in subject "{self.subject}": '
                f"{len(self.broken)} rule(s) broken ({names})."
            )
            steps.append(f'Loaded {len(self.rules_loaded)} rules for subject "{self.subject}".')
            steps.append(
                f"Checked the conversation: {self.message_count} messages, "
                f"{self.tool_call_count} tool call(s)."
            )
            for item in self.broken:
                lines = [
                    f'Rule "{item.rule}" ({item.severity}) broke.',
                    f"   Why: {item.description}",
                ]
                lines.extend(f"   Where: {spot}" for spot in item.where or ["(whole conversation)"])
                steps.append("\n".join(lines))
            simple = " ".join(dict.fromkeys(item.plain for item in self.broken if item.plain))
        else:
            alert = (
                f'ALERT: Keeper could not finish checking subject "{self.subject}", '
                "so it stopped this step."
            )
            if self.rules_loaded:
                steps.append(f'Loaded {len(self.rules_loaded)} rules for subject "{self.subject}".')
            steps.extend(f"Problem: {issue}" for issue in self.issues or ["unknown problem"])
            simple = (
                "The guard couldn't finish reading the work. When it can't be sure, it says stop."
            )
        steps.append("Stopped. Nothing was retried or worked around. Waiting for Al.")
        numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
        return f"{alert}\n\nWhat happened, step by step:\n{numbered}\n\nIn simple terms: {simple}"


def rule_files_for(subject: Subject) -> list[Path]:
    library = library_rules()
    return [*standing_rules(), *(library[name] for name in subject.use), *subject.custom_checks()]


def check(root: Path, subject_name: str, conversation: Any) -> CheckReport:
    """Check one conversation. Never raises for a bad check: it returns a STOP report."""

    rules_loaded: list[str] = []
    try:
        subject = load_subject(root, subject_name)
    except Exception as error:  # noqa: BLE001 - every failure must become a STOP
        return _error_report(subject_name, rules_loaded, f"subject could not load: {error}")

    messages = conversation.get("messages") if isinstance(conversation, dict) else None
    message_count = len(messages) if isinstance(messages, list) else 0
    tool_call_count = sum(
        len(message.get("tool_calls") or [])
        for message in (messages if isinstance(messages, list) else [])
        if isinstance(message, dict)
    )

    try:
        with network_lock():
            # Imported inside the lock so nothing it pulls in can reach the network.
            from quarry.config import QuarryConfig
            from quarry.scanner import DSLScanner

            document = {
                "config_version": 2,
                "rules": [str(path) for path in rule_files_for(subject)],
            }
            require_offline_config(document)
            scanner = DSLScanner(QuarryConfig.from_mapping(document, subject.path))
            rules_loaded = sorted(scanner.loaded_rules)
            result = scanner.scan_data(conversation, enable_explain=True)
    except Exception as error:  # noqa: BLE001 - every failure must become a STOP
        report = _error_report(subject.name, rules_loaded, f"{type(error).__name__}: {error}")
        report.message_count, report.tool_call_count = message_count, tool_call_count
        return report

    if not result.is_complete:
        report = _error_report(subject.name, rules_loaded, *result.completion_issues)
        report.message_count, report.tool_call_count = message_count, tool_call_count
        return report

    evidence = {item.rule_name: item for item in result.detailed_matches or []}
    broken = []
    for name in result.rule_matches:
        metadata = dict(scanner.loaded_rules[name].metadata)
        broken.append(
            BrokenRule(
                rule=name,
                severity=str(metadata.get("severity", "high")),
                description=str(metadata.get("description", "")),
                plain=str(metadata.get("plain", "")),
                where=_where(evidence.get(name)),
            )
        )
    return CheckReport(
        subject=subject.name,
        verdict="STOP" if broken else "PASS",
        stop_kind="rule" if broken else "none",
        rules_loaded=rules_loaded,
        message_count=message_count,
        tool_call_count=tool_call_count,
        broken=broken,
    )


def check_file(root: Path, subject_name: str, conversation_path: Path) -> CheckReport:
    try:
        conversation = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return _error_report(subject_name, [], f"conversation file could not be read: {error}")
    return check(root, subject_name, conversation)


def write_log(root: Path, report: CheckReport) -> Path:
    """Append one line per check to log/<date>.md."""

    log_dir = Path(root).expanduser().resolve() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{report.checked_at[:10]}.md"
    detail = (
        ", ".join(item.rule for item in report.broken)
        if report.stop_kind == "rule"
        else "; ".join(report.issues)
    )
    line = f"- {report.checked_at} · {report.subject} · {report.verdict}"
    if report.verdict == "STOP":
        line += f" ({report.stop_kind}: {detail})"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def _error_report(subject: str, rules_loaded: list[str], *issues: str) -> CheckReport:
    return CheckReport(
        subject=subject,
        verdict="STOP",
        stop_kind="error",
        rules_loaded=rules_loaded,
        issues=list(issues),
    )


def _where(evidence: Any) -> list[str]:
    if evidence is None:
        return []
    spots = []
    steps = [step for step in evidence.to_dict().get("witness") or [] if step.get("excerpt")]
    for step in steps or evidence.to_dict().get("witness") or []:
        label = step.get("path", "?")
        if step.get("tool_name"):
            label += f" (tool {step['tool_name']})"
        excerpt = _SECRET.sub("[hidden secret]", str(step.get("excerpt") or "")).replace("\n", " ")
        if len(excerpt) > _MAX_EXCERPT:
            excerpt = excerpt[: _MAX_EXCERPT - 1] + "…"
        spots.append(f'{label}: "{excerpt}"' if excerpt else label)
    return spots

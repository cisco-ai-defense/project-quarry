"""The logbook: job records, SUCCESS files, rejections, and the ghost lap.

How it fits together (the coach model):

- A **job** is one task inside a subject. Starting a job hands the agent the **ghost lap**:
  the current basis (best liked result) for that kind of job, plus the **marked crashes**
  (rejected directions still on file) to avoid.
- Every step the agent takes is appended to the job's ``path.log`` as it happens.
- When Al rejects an attempt, a rejection file is written under
  ``rejections/<date>-<session>/`` and the next attempt starts.
- When Al accepts, ``SUCCESS.md`` is written in the job folder. If there is no basis for
  that kind yet, or Al says this one is better, it becomes the new basis in
  ``bases/<kind>/BASIS.md`` and the old one moves to ``bases/<kind>/archive/``.
- "New direction" lifts the ghost lap and the avoid list for that one job, and is logged.
- Only Al deletes rejections (``forget_rejections``); agents are stopped from doing it by
  the standing rule "Agent deleting the memory".
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from keeper.guard import rule_files_for
from keeper.layout import LayoutError, Subject, load_subject, require_root

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_MAX_FIELD = 2000

SECRET = re.compile(
    r"(sk-ant-[A-Za-z0-9_-]{8,}|sk-(proj-)?[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{8,}"
    r"|github_pat_[A-Za-z0-9_]{8,}|AKIA[0-9A-Z]{8,}|xox[abpr]-[A-Za-z0-9-]{6,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


class LogbookError(ValueError):
    """A job, basis or rejection is missing, closed, or malformed."""


@dataclass
class Job:
    id: str
    subject: str
    path: Path
    title: str
    ask: str
    kind: str
    session: str
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    status: str = "open"
    attempt: int = 1
    new_direction: str = ""
    created: str = ""

    @property
    def log_path(self) -> Path:
        return self.path / "path.log"

    def save(self) -> None:
        data = {
            "id": self.id,
            "subject": self.subject,
            "title": self.title,
            "ask": self.ask,
            "kind": self.kind,
            "session": self.session,
            "keywords": self.keywords,
            "references": self.references,
            "status": self.status,
            "attempt": self.attempt,
            "new_direction": self.new_direction,
            "created": self.created,
        }
        (self.path / "job.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def steps(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@dataclass
class GhostLap:
    """What the coach hands an agent at the start of a job."""

    kind: str
    basis: Path | None
    basis_text: str
    avoid: list[dict[str, Any]]
    new_direction: str = ""

    def to_text(self) -> str:
        lines = [f'# Ghost lap for "{self.kind}"', ""]
        if self.new_direction:
            lines += [
                f"New direction from Al: {self.new_direction}",
                "The ghost lap and the avoid list are lifted for this job.",
            ]
            return "\n".join(lines) + "\n"
        if self.basis is None:
            lines += [
                "No basis yet. This is the first lap: explore, and every step is recorded.",
                "",
            ]
        else:
            lines += [f"Basis: {self.basis}", "", self.basis_text.strip(), ""]
        lines.append('## Marked crashes (avoid unless Al says "new direction")')
        lines.append("")
        if not self.avoid:
            lines.append("(none on file)")
        for item in self.avoid:
            lines.append(f"- {item.get('summary') or '(no summary)'}: {item.get('why', '')}")
        return "\n".join(lines) + "\n"


# --- jobs ---------------------------------------------------------------------------------


def start_job(
    root: Path,
    subject_name: str,
    title: str,
    *,
    kind: str,
    session: str,
    ask: str = "",
    keywords: list[str] | None = None,
    references: list[str] | None = None,
    new_direction: str = "",
) -> tuple[Job, GhostLap]:
    root = require_root(root)
    subject = load_subject(root, subject_name)
    _require_slug(kind, "kind")
    _require_slug(session, "session")
    missing = [ref for ref in references or [] if not (subject.path / "references" / ref).exists()]
    if missing:
        raise LogbookError(
            "These references are not in the subject's references/ folder: " + ", ".join(missing)
        )
    today = _today()
    base_id = f"{today}-{_slugify(title)}"
    job_id, counter = base_id, 2
    while (subject.path / "jobs" / job_id).exists():
        job_id, counter = f"{base_id}-{counter}", counter + 1
    job = Job(
        id=job_id,
        subject=subject.name,
        path=subject.path / "jobs" / job_id,
        title=title,
        ask=ask,
        kind=kind,
        session=session,
        keywords=sorted({word.lower() for word in keywords or []}),
        references=list(references or []),
        new_direction=new_direction,
        created=_now(),
    )
    job.path.mkdir(parents=True)
    job.save()
    _append(job, {"event": "start", "note": title})
    if new_direction:
        _append(job, {"event": "new-direction", "note": new_direction})
    return job, ghost_lap(root, kind, keywords=job.keywords, new_direction=new_direction)


def load_job(root: Path, subject_name: str, job_id: str) -> Job:
    subject = load_subject(root, subject_name)
    path = subject.path / "jobs" / job_id
    meta = path / "job.yaml"
    if not meta.is_file():
        raise LogbookError(f"Job '{job_id}' does not exist in subject '{subject.name}'.")
    data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
    return Job(path=path, **{key: data[key] for key in data if key in Job.__dataclass_fields__})


def record_step(
    root: Path,
    subject_name: str,
    job_id: str,
    *,
    tool: str,
    input: str = "",  # noqa: A002 - matches the record's field name
    output: str = "",
    note: str = "",
) -> dict[str, Any]:
    job = _open_job(root, subject_name, job_id)
    return _append(
        job, {"event": "step", "tool": tool, "input": input, "output": output, "note": note}
    )


def record_conversation(root: Path, subject_name: str, job_id: str, conversation: Any) -> int:
    """Turn every tool call (and its result) in a conversation into steps. Returns the count."""

    job = _open_job(root, subject_name, job_id)
    messages = conversation.get("messages") if isinstance(conversation, dict) else None
    if not isinstance(messages, list):
        raise LogbookError("The conversation has no 'messages' list.")
    results = {
        message.get("tool_call_id"): message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            _append(
                job,
                {
                    "event": "step",
                    "tool": str(function.get("name", "?")),
                    "input": str(function.get("arguments", "")),
                    "output": _text(results.get(call.get("id"))),
                    "note": "",
                },
            )
            count += 1
    return count


def reject(root: Path, subject_name: str, job_id: str, *, why: str, summary: str = "") -> Path:
    """Al rejected the current attempt. Writes one rejection file and starts the next attempt."""

    if not why.strip():
        raise LogbookError("A rejection needs a reason, in Al's words.")
    root = require_root(root)
    job = _open_job(root, subject_name, job_id)
    attempt_steps = [step for step in job.steps() if step.get("attempt") == job.attempt]
    folder = root / "rejections" / f"{job.created[:10]}-{job.session}"
    folder.mkdir(parents=True, exist_ok=True)
    number = len(list(folder.glob("*.md"))) + 1
    path = folder / f"{number:02d}-{job.subject}-{_slugify(job.title)}-try{job.attempt}.md"
    front = {
        "kind": job.kind,
        "subject": job.subject,
        "job": job.id,
        "attempt": job.attempt,
        "date": _now(),
        "summary": _clean(summary) or _direction(attempt_steps),
        "why": _clean(why),
        "keywords": job.keywords,
    }
    body = [
        "---",
        yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip(),
        "---",
        "",
        f"# Rejected: {job.title} (try {job.attempt})",
        "",
        f"**Why (Al):** {front['why']}",
        "",
        "## Steps in this try",
        "",
        *_step_lines(attempt_steps),
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    _append(job, {"event": "rejected", "note": front["why"]})
    job.attempt += 1
    job.save()
    return path


def accept(
    root: Path, subject_name: str, job_id: str, *, note: str = "", better: bool = False
) -> tuple[Path, Path | None]:
    """Al accepted the work. Writes SUCCESS.md; sets the basis if none exists or ``better``."""

    root = require_root(root)
    job = _open_job(root, subject_name, job_id)
    subject = load_subject(root, subject_name)
    _append(job, {"event": "accepted", "note": note})
    job.status = "accepted"
    job.save()
    success = job.path / "SUCCESS.md"
    success.write_text(_success_md(root, subject, job, note), encoding="utf-8")

    basis_dir = root / "bases" / job.kind
    basis = basis_dir / "BASIS.md"
    if basis.exists() and not better:
        return success, None
    basis_dir.mkdir(parents=True, exist_ok=True)
    if basis.exists():
        archive = basis_dir / "archive"
        archive.mkdir(exist_ok=True)
        shutil.move(str(basis), archive / f"{_now().replace(':', '')}-replaced.md")
    shutil.copyfile(success, basis)
    return success, basis


# --- ghost lap ----------------------------------------------------------------------------


def ghost_lap(
    root: Path, kind: str, *, keywords: list[str] | None = None, new_direction: str = ""
) -> GhostLap:
    """The one best basis for this kind of job, plus the rejections still on file."""

    root = require_root(root)
    if new_direction:
        return GhostLap(kind=kind, basis=None, basis_text="", avoid=[], new_direction=new_direction)
    basis = root / "bases" / kind / "BASIS.md"
    if not basis.is_file():
        basis = _closest_basis(root, set(keywords or []))
    avoid = [item for item in list_rejections(root) if item.get("kind") == kind]
    return GhostLap(
        kind=kind,
        basis=basis,
        basis_text=basis.read_text(encoding="utf-8") if basis else "",
        avoid=avoid,
    )


def _closest_basis(root: Path, keywords: set[str]) -> Path | None:
    """No basis for this exact kind: pick the one sharing the most keywords, if any."""

    best, best_score = None, 0
    for path in sorted((root / "bases").glob("*/BASIS.md")):
        match = re.search(r"^- \*\*Keywords:\*\* (.*)$", path.read_text(encoding="utf-8"), re.M)
        theirs = {word.strip() for word in match.group(1).split(",")} if match else set()
        score = len(keywords & theirs)
        if score > best_score:
            best, best_score = path, score
    return best


# --- rejections (only Al deletes) ---------------------------------------------------------


def list_rejections(root: Path, session: str | None = None) -> list[dict[str, Any]]:
    root = require_root(root)
    folders = sorted(path for path in (root / "rejections").iterdir() if path.is_dir())
    items = []
    for folder in folders:
        if session and not folder.name.endswith(f"-{session}") and folder.name != session:
            continue
        for path in sorted(folder.glob("*.md")):
            front = _front_matter(path)
            front["file"] = str(path.relative_to(root))
            front["session_folder"] = folder.name
            items.append(front)
    return items


def forget_rejections(
    root: Path, session_folder: str, files: list[str] | None = None
) -> list[Path]:
    """Delete one session's rejections, or only the named files in it. Al's command only."""

    root = require_root(root)
    folder = (root / "rejections" / session_folder).resolve()
    if folder.parent != (root / "rejections").resolve() or not folder.is_dir():
        raise LogbookError(f"No rejection session folder named '{session_folder}'.")
    targets = [folder / name for name in files] if files else sorted(folder.glob("*.md"))
    for target in targets:
        if target.resolve().parent != folder or not target.is_file():
            raise LogbookError(f"'{target.name}' is not a rejection file in '{session_folder}'.")
    for target in targets:
        target.unlink()
    if not any(folder.iterdir()):
        folder.rmdir()
    return targets


# --- helpers ------------------------------------------------------------------------------


def _open_job(root: Path, subject_name: str, job_id: str) -> Job:
    job = load_job(root, subject_name, job_id)
    if job.status != "open":
        raise LogbookError(f"Job '{job_id}' is already {job.status}. Start a new job instead.")
    return job


def _append(job: Job, record: dict[str, Any]) -> dict[str, Any]:
    entry = {"time": _now(), "attempt": job.attempt}
    entry.update(
        {key: _clean(value) if isinstance(value, str) else value for key, value in record.items()}
    )
    with job.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _success_md(root: Path, subject: Subject, job: Job, note: str) -> str:
    steps = job.steps()
    tries = job.attempt
    accepted = [
        step for step in steps if step.get("attempt") == tries and step.get("event") == "step"
    ]
    rejected_notes = [step for step in steps if step.get("event") == "rejected"]
    tool_counts = Counter(step.get("tool") for step in accepted if step.get("tool"))
    rules = [path.stem for path in rule_files_for(subject)]
    lines = [
        f"# SUCCESS: {job.title}",
        "",
        f"- **Subject:** {job.subject}",
        f"- **Kind:** {job.kind}",
        f"- **Keywords:** {', '.join(job.keywords)}",
        f"- **References:** {', '.join(job.references) or 'none'}",
        f"- **Session:** {job.session}",
        f"- **Accepted:** {_now()} after {tries} tr{'y' if tries == 1 else 'ies'}",
    ]
    if job.new_direction:
        lines.append(f"- **New direction (Al):** {job.new_direction}")
    lines += ["", "## The ask", "", job.ask or job.title, ""]
    if note:
        lines += ["## Al's note on acceptance", "", _clean(note), ""]
    lines += ["## Rules in force", "", *(f"- {name}" for name in rules), ""]
    lines += ["## The path, in time order", ""]
    for step in steps:
        lines.append(_timeline_line(step))
    lines += ["", "## Rejected directions (avoid these)", ""]
    if not rejected_notes:
        lines.append("(none: accepted on the first try)")
    for number in range(1, tries):
        try_steps = [s for s in steps if s.get("attempt") == number and s.get("event") == "step"]
        why = next((s.get("note") for s in rejected_notes if s.get("attempt") == number), "")
        lines.append(f"- **Try {number}:** {_direction(try_steps)}. Rejected because: {why}")
    lines += ["", "## Accepted direction", "", *_step_lines(accepted), ""]
    lines += ["## Best tools", ""]
    lines += [f"- {tool} (used {count}x)" for tool, count in tool_counts.most_common()] or [
        "(none)"
    ]
    lines += ["", "## Replay recipe", ""]
    lines += [
        f"{index}. {step.get('tool')}" + (f": {step.get('note')}" if step.get("note") else "")
        for index, step in enumerate(accepted, 1)
    ] or ["(no tool steps recorded)"]
    lines.append("")
    return "\n".join(lines)


def _timeline_line(step: dict[str, Any]) -> str:
    stamp, attempt, event = step.get("time", ""), step.get("attempt"), step.get("event")
    if event == "step":
        detail = f"{step.get('tool')}" + (f" ({step.get('note')})" if step.get("note") else "")
    else:
        detail = f"{event}" + (f": {step.get('note')}" if step.get("note") else "")
    return f"- {stamp} · try {attempt} · {detail}"


def _step_lines(steps: list[dict[str, Any]]) -> list[str]:
    tool_steps = [step for step in steps if step.get("event") == "step"]
    if not tool_steps:
        return ["(no tool steps recorded)"]
    lines = []
    for index, step in enumerate(tool_steps, 1):
        lines.append(
            f"{index}. **{step.get('tool')}**"
            + (f": {step.get('note')}" if step.get("note") else "")
        )
        if step.get("input"):
            lines.append(f"   - in: {_short(step['input'])}")
        if step.get("output"):
            lines.append(f"   - out: {_short(step['output'])}")
    return lines


def _direction(steps: list[dict[str, Any]]) -> str:
    tools = [str(step.get("tool")) for step in steps if step.get("event") == "step"]
    return " → ".join(tools) if tools else "no tool steps recorded"


def _front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}
    data = yaml.safe_load(match.group(1))
    return data if isinstance(data, dict) else {}


def _clean(value: str) -> str:
    text = SECRET.sub("[hidden secret]", value)
    return text if len(text) <= _MAX_FIELD else text[: _MAX_FIELD - 1] + "…"


def _short(value: str, limit: int = 200) -> str:
    flat = str(value).replace("\n", " ")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50].strip("-")
    return slug or "job"


def _require_slug(value: str, label: str) -> None:
    if not SLUG.match(value):
        raise LayoutError(
            f"The {label} '{value}' is not allowed. Use lowercase letters, digits and dashes."
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _today() -> str:
    return _now()[:10]

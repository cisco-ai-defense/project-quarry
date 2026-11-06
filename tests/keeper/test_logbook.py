"""Tests for the Keeper logbook (phase 2): jobs, SUCCESS files, rejections, ghost lap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from keeper import cli, logbook
from keeper.guard import check
from keeper.layout import add_subject, init_root

FAKE_KEY = "sk-ant-api03-" + "b" * 30


@pytest.fixture
def root(tmp_path: Path) -> Path:
    keeper_root = init_root(tmp_path / "keeper")
    add_subject(keeper_root, "art")
    add_subject(keeper_root, "music")
    (keeper_root / "subjects" / "art" / "references" / "hero.png").write_bytes(b"png")
    return keeper_root


def start(root: Path, title: str = "Hero walk", **extra):
    options = {"kind": "sprite-walk", "session": "hero-walk", "keywords": ["sprite", "walk"]}
    options.update(extra)
    return logbook.start_job(root, "art", title, **options)


def test_first_job_has_no_basis(root: Path) -> None:
    job, lap = start(root, references=["hero.png"])
    assert lap.basis is None and lap.avoid == []
    assert "first lap" in lap.to_text()
    assert job.steps()[0]["event"] == "start"


def test_missing_reference_is_refused(root: Path) -> None:
    with pytest.raises(logbook.LogbookError):
        start(root, references=["nope.png"])


def test_bad_kind_or_session_is_refused(root: Path) -> None:
    with pytest.raises(ValueError):
        start(root, kind="Bad Kind")
    with pytest.raises(ValueError):
        start(root, session="../x")


def test_same_title_same_day_gets_a_new_id(root: Path) -> None:
    first, _ = start(root)
    second, _ = start(root)
    assert first.id != second.id


def test_full_coach_loop(root: Path) -> None:
    job, _ = start(root, ask="Build from hero.png")
    logbook.record_step(root, "art", job.id, tool="image_generate", note="redrew the hero")
    rejection = logbook.reject(root, "art", job.id, why="Build from the reference")
    assert rejection.parent.name.endswith("-hero-walk")
    assert logbook.load_job(root, "art", job.id).attempt == 2

    logbook.record_step(root, "art", job.id, tool="load_reference", input="hero.png")
    success, basis = logbook.accept(root, "art", job.id, note="Perfect")
    text = success.read_text(encoding="utf-8")
    for heading in (
        "## The path, in time order",
        "## Rejected directions (avoid these)",
        "## Accepted direction",
        "## Best tools",
        "## Replay recipe",
        "## Rules in force",
    ):
        assert heading in text
    assert "Try 1:** image_generate. Rejected because: Build from the reference" in text
    assert "1. load_reference" in text
    assert basis is not None and basis.read_text(encoding="utf-8") == text

    _, lap = start(root, "Villain walk", session="villain-walk")
    assert lap.basis == basis
    assert [item["why"] for item in lap.avoid] == ["Build from the reference"]


def test_accepted_job_is_closed(root: Path) -> None:
    job, _ = start(root)
    logbook.accept(root, "art", job.id)
    with pytest.raises(logbook.LogbookError):
        logbook.record_step(root, "art", job.id, tool="x")


def test_basis_changes_only_when_liked_better(root: Path) -> None:
    first, _ = start(root, "First")
    _, basis = logbook.accept(root, "art", first.id, note="good")
    original = basis.read_text(encoding="utf-8")

    second, _ = start(root, "Second")
    _, kept = logbook.accept(root, "art", second.id)
    assert kept is None and basis.read_text(encoding="utf-8") == original

    third, _ = start(root, "Third")
    _, replaced = logbook.accept(root, "art", third.id, better=True)
    assert replaced == basis and "SUCCESS: Third" in basis.read_text(encoding="utf-8")
    archived = list((basis.parent / "archive").glob("*.md"))
    assert len(archived) == 1 and "SUCCESS: First" in archived[0].read_text(encoding="utf-8")


def test_new_direction_lifts_ghost_lap(root: Path) -> None:
    job, _ = start(root)
    logbook.reject(root, "art", job.id, why="too bright")
    logbook.accept(root, "art", job.id)
    fresh, lap = start(root, "Dark walk", new_direction="try a darker palette")
    assert lap.basis is None and lap.avoid == []
    assert "lifted" in lap.to_text()
    assert any(step["event"] == "new-direction" for step in fresh.steps())


def test_closest_basis_by_keywords(root: Path) -> None:
    job, _ = start(root)
    logbook.accept(root, "art", job.id)
    lap = logbook.ghost_lap(root, "sprite-run", keywords=["sprite", "run"])
    assert lap.basis is not None and lap.basis.parent.name == "sprite-walk"
    assert logbook.ghost_lap(root, "song-mix", keywords=["drums"]).basis is None


def test_record_conversation_steps(root: Path) -> None:
    job, _ = start(root)
    conversation = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "write", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "file text"},
            {"role": "tool", "tool_call_id": "b", "content": "saved"},
        ]
    }
    assert logbook.record_conversation(root, "art", job.id, conversation) == 2
    steps = [step for step in job.steps() if step["event"] == "step"]
    assert [(step["tool"], step["output"]) for step in steps] == [
        ("read", "file text"),
        ("write", "saved"),
    ]


def test_secrets_never_reach_the_logbook(root: Path) -> None:
    job, _ = start(root)
    logbook.record_step(root, "art", job.id, tool="x", input=f"key {FAKE_KEY}")
    rejection = logbook.reject(root, "art", job.id, why=f"leaked {FAKE_KEY}")
    success, _ = logbook.accept(root, "art", job.id)
    for path in (job.log_path, rejection, success):
        assert FAKE_KEY not in path.read_text(encoding="utf-8")


def test_reject_needs_a_reason(root: Path) -> None:
    job, _ = start(root)
    with pytest.raises(logbook.LogbookError):
        logbook.reject(root, "art", job.id, why="  ")


# --- forgetting ---------------------------------------------------------------------------


def make_rejections(root: Path, count: int) -> str:
    job, _ = start(root)
    for number in range(count):
        logbook.reject(root, "art", job.id, why=f"no {number}")
    return logbook.list_rejections(root)[0]["session_folder"]


def test_forget_some_rejections(root: Path) -> None:
    folder = make_rejections(root, 3)
    files = sorted(path.name for path in (root / "rejections" / folder).glob("*.md"))
    logbook.forget_rejections(root, folder, [files[0]])
    assert len(logbook.list_rejections(root)) == 2


def test_forget_a_whole_session(root: Path) -> None:
    folder = make_rejections(root, 2)
    logbook.forget_rejections(root, folder)
    assert logbook.list_rejections(root) == []
    assert not (root / "rejections" / folder).exists()


def test_forget_cannot_escape_the_rejections_folder(root: Path) -> None:
    folder = make_rejections(root, 1)
    with pytest.raises(logbook.LogbookError):
        logbook.forget_rejections(root, "../subjects")
    with pytest.raises(logbook.LogbookError):
        logbook.forget_rejections(root, folder, ["../../KEEPER.md"])
    assert (root / "KEEPER.md").exists()


def test_list_rejections_by_session(root: Path) -> None:
    make_rejections(root, 1)
    assert len(logbook.list_rejections(root, session="hero-walk")) == 1
    assert logbook.list_rejections(root, session="other") == []


@pytest.mark.parametrize(
    "command",
    [
        "keeper rejections forget ~/keeper 2026-11-06-hero-walk --yes",
        "rm -rf ~/keeper/rejections/2026-11-06-hero-walk",
        "rm ~/keeper/subjects/art/jobs/x/SUCCESS.md",
        "unlink ~/keeper/bases/sprite-walk/BASIS.md",
    ],
)
def test_agents_are_stopped_from_forgetting(root: Path, command: str) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "clean up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ]
    }
    report = check(root, "music", conversation)
    assert "Agent deleting the memory" in {item.rule for item in report.broken}


def test_cli_logbook_flow(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = str(root)
    args = ["job", "start", folder, "art", "Hero walk", "--kind", "sprite-walk", "--session", "s1"]
    assert cli.main(args) == 0
    job_id = next((root / "subjects" / "art" / "jobs").iterdir()).name
    assert cli.main(["job", "step", folder, "art", job_id, "--tool", "draw"]) == 0
    assert cli.main(["job", "reject", folder, "art", job_id, "--why", "off model"]) == 0
    assert cli.main(["job", "accept", folder, "art", job_id]) == 0
    assert cli.main(["ghost", folder, "sprite-walk"]) == 0
    assert cli.main(["rejections", "list", folder]) == 0
    session = logbook.list_rejections(root)[0]["session_folder"]
    assert cli.main(["rejections", "forget", folder, session]) == 2  # no --yes, no terminal
    assert cli.main(["rejections", "forget", folder, session, "--yes"]) == 0
    assert logbook.list_rejections(root) == []
    assert cli.main(["job", "step", folder, "art", job_id, "--tool", "late"]) == 2
    output = capsys.readouterr().out
    assert "SUCCESS written" in output and "Ghost lap" in output

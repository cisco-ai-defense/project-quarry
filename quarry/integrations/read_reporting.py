# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bounded scalar logical-read summaries; never sum incompatible token vocabularies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from quarry.integrations.read_coverage import CoverageLimitError, ReadCoverage


class _Budget:
    def __init__(self, limits):
        self.max_groups = limits.max_aggregate_groups
        self.max_keys = limits.max_aggregate_keys
        self.max_ranges = limits.max_aggregate_ranges
        self.max_work = limits.max_aggregate_work
        self.groups = self.keys = self.ranges = self.work = 0
        self.limited = False

    @property
    def remaining(self):
        return max(0, self.max_work - self.work)

    def charge(self, work):
        if work > self.remaining:
            self.work = self.max_work
            self.limited = True
            return False
        self.work += work
        return True

    def group(self):
        # Reserve container and eventual scalar-render traversal before allocation.
        if self.groups >= self.max_groups or not self.charge(64):
            self.limited = True
            return False
        self.groups += 1
        return True


@dataclass
class _Unit:
    coverage: ReadCoverage = field(default_factory=ReadCoverage)
    states: Counter = field(default_factory=Counter)
    limited: bool = False

    def add(self, measurement, budget, *, limited=False):
        self.states[measurement.state] += 1
        self.limited |= limited
        if not measurement.intervals:
            return
        before = self.coverage.snapshot()
        self.coverage.max_buckets = before.buckets + max(0, budget.max_keys - budget.keys)
        self.coverage.max_intervals = before.intervals + max(0, budget.max_ranges - budget.ranges)
        self.coverage.max_work = before.work + budget.remaining
        try:
            self.coverage.add(measurement.intervals)
        except CoverageLimitError:
            self.limited = budget.limited = True
        after = self.coverage.snapshot()
        budget.work += after.work - before.work
        budget.keys += after.buckets - before.buckets
        budget.ranges += after.intervals - before.intervals

    def render(self, unit, *, incomplete):
        totals = self.coverage.snapshot()
        known = totals.characters if unit == "characters" else totals.text_tokens
        states = dict(sorted(self.states.items()))
        complete = not incomplete and not self.limited and set(states) <= {"measured"}
        return {
            "value": known if complete else None,
            "known_units": known,
            "complete": complete,
            "measurement_states": states,
            "coverage_limited": self.limited,
        }


def _basis_label(key):
    if key is None:
        return None
    if key[0] == "reference":
        return {"kind": "reference", "name": key[1], "version": key[2]}
    return {"kind": "supplied_intervals", "model": key[1]}


def _mixed_tokens(groups, *, incomplete):
    if not groups:
        return {
            "value": 0 if not incomplete else None,
            "known_units": 0,
            "complete": not incomplete,
            "measurement_states": {},
            "coverage_limited": False,
            "basis_compatible": True,
        }
    if len(groups) == 1:
        return {**groups[0]["coverage"], "basis_compatible": groups[0]["basis"] is not None}
    states = Counter()
    for group in groups:
        states.update(group["coverage"]["measurement_states"])
    return {
        "value": None,
        "known_units": None,
        "complete": False,
        "measurement_states": dict(sorted(states.items())),
        "coverage_limited": any(group["coverage"]["coverage_limited"] for group in groups),
        "basis_compatible": False,
    }


class _Group:
    def __init__(self):
        self.count = self.invalid = 0
        self.evaluations = set()
        self.cuts = set()
        self.cuts_complete = True
        self.models = set()
        self.outcomes = Counter()
        self.operations = Counter()
        self.characters = _Unit()
        self.tokens = {}

    def add(self, identity, event, tokens, basis, token_limited, budget):
        if not budget.charge(16):
            return False
        self.count += 1
        self.invalid += event.invalid
        self.evaluations.add(identity)
        self.models.add(event.model)
        self.operations[event.operation] += 1
        self.outcomes[event.outcome] += 1
        if event.cut is None:
            self.cuts_complete = False
        else:
            self.cuts.add(event.cut)
        self.characters.add(event.characters, budget)
        if basis not in self.tokens:
            if not budget.group():
                return False
            self.tokens[basis] = (_Unit(), set())
        unit, models = self.tokens[basis]
        models.add(event.model)
        unit.add(tokens, budget, limited=token_limited)
        return True

    def render(self, *, incomplete):
        groups = [
            {
                "basis": _basis_label(basis),
                "models": sorted(models),
                "coverage": unit.render("text_tokens", incomplete=incomplete),
            }
            for basis, (unit, models) in sorted(
                self.tokens.items(), key=lambda pair: pair[0] or ("", "", "")
            )
        ]
        coverage = {
            "characters": self.characters.render("characters", incomplete=incomplete),
            "text_tokens": _mixed_tokens(groups, incomplete=incomplete),
        }
        return {
            "observations": self.count,
            "evaluations": len(self.evaluations),
            "unique_cuts": len(self.cuts),
            "cut_identity_complete": self.cuts_complete and not incomplete,
            "models": sorted(self.models),
            "operations": dict(sorted(self.operations.items())),
            "outcomes": dict(sorted(self.outcomes.items())),
            "invalid_measurements": self.invalid,
            "observations_complete": not incomplete
            and not ({"pending", "unknown"} & self.outcomes.keys()),
            "measurements_complete": all(item["complete"] for item in coverage.values()),
            "coverage": coverage,
            "token_bases": groups,
        }


def _sum_coverage(items, *, incomplete):
    states = Counter()
    for item in items:
        states.update(item["measurement_states"])
    known = sum(item["known_units"] for item in items)
    complete = not incomplete and all(item["complete"] for item in items)
    return {
        "value": known if complete else None,
        "known_units": known,
        "complete": complete,
        "measurement_states": dict(sorted(states.items())),
        "coverage_limited": any(item["coverage_limited"] for item in items),
    }


def sum_scopes(scopes, *, incomplete=False):
    """Distinct conversation namespaces add; their input intervals never cross-deduplicate."""
    bases = {}
    operations, outcomes = Counter(), Counter()
    models = set()
    for scope in scopes:
        operations.update(scope["operations"])
        outcomes.update(scope["outcomes"])
        models.update(scope["models"])
        for group in scope["token_bases"]:
            basis = group["basis"]
            key = None if basis is None else tuple(sorted(basis.items()))
            if key not in bases:
                bases[key] = (basis, [], set())
            _, items, names = bases[key]
            items.append(group["coverage"])
            names.update(group["models"])
    groups = [
        {
            "basis": basis,
            "models": sorted(names),
            "coverage": _sum_coverage(items, incomplete=incomplete),
        }
        for _, (basis, items, names) in sorted(bases.items(), key=lambda pair: pair[0] or ())
    ]
    coverage = {
        "characters": _sum_coverage(
            [scope["coverage"]["characters"] for scope in scopes], incomplete=incomplete
        ),
        "text_tokens": _mixed_tokens(groups, incomplete=incomplete),
    }
    return {
        "observations": sum(scope["observations"] for scope in scopes),
        "evaluations": sum(scope["evaluations"] for scope in scopes),
        "unique_cuts": sum(scope["unique_cuts"] for scope in scopes),
        "cut_identity_complete": not incomplete
        and all(scope["cut_identity_complete"] for scope in scopes),
        "invalid_measurements": sum(scope["invalid_measurements"] for scope in scopes),
        "models": sorted(models),
        "operations": dict(sorted(operations.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "observations_complete": not incomplete
        and all(scope["observations_complete"] for scope in scopes),
        "measurements_complete": all(item["complete"] for item in coverage.values()),
        "coverage": coverage,
        "token_bases": groups,
    }


def build_aggregates(bindings, events, resolve_tokens, limits, *, omitted):
    budget = _Budget(limits)
    groups: dict[tuple, _Group] = {}
    omitted_memberships = 0
    for number, (reader, identity) in enumerate(bindings):
        if not budget.charge(16):
            omitted_memberships += 3 * (len(bindings) - number)
            break
        event = events[identity]
        tokens, basis, token_limited, cost = resolve_tokens(identity, event, budget.remaining)
        if not budget.charge(cost):
            omitted_memberships += 3 * (len(bindings) - number)
            break
        keys = (
            ("scope", reader.scope),
            ("model", reader.scope, event.model),
            ("reader", reader.scope, reader.rule, reader.phase, event.model),
        )
        for key in keys:
            if key not in groups:
                if not budget.group():
                    omitted_memberships += 1
                    continue
                groups[key] = _Group()
            if not groups[key].add(identity, event, tokens, basis, token_limited, budget):
                omitted_memberships += 1
    incomplete = bool(omitted or omitted_memberships or budget.limited)
    readers, models, scopes = [], [], []
    for key, group in sorted(
        groups.items(),
        key=lambda pair: (pair[0][0], pair[0][1], *(item or "" for item in pair[0][2:])),
    ):
        value = group.render(incomplete=incomplete)
        if key[0] == "reader":
            readers.append(
                {"scope": key[1], "rule": key[2], "phase": key[3], "model": key[4], **value}
            )
        elif key[0] == "model":
            models.append({"scope": key[1], "model": key[2], **value})
        else:
            scopes.append({"scope": key[1], **value})
    return {
        "complete": not incomplete
        and all(
            scope["observations_complete"] and scope["measurements_complete"] for scope in scopes
        ),
        "omitted_memberships": omitted_memberships,
        "coverage_limited": budget.limited,
        "by_reader": readers,
        "by_model": models,
        "by_scope": scopes,
        "totals": sum_scopes(scopes, incomplete=incomplete),
    }


def scope_report(report, scope):
    """Filter only scalar records; unknown scope admission never exposes another scan."""
    result = scopes_report(report, {scope})
    result.pop("scopes")
    result["scope"] = scope
    return result


def scopes_report(report, scopes):
    """Summarize only the conversation namespaces owned by a file or batch request."""
    result = {
        key: value
        for key, value in report.items()
        if key not in {"rows", "aggregates", "token_bases"}
    }
    result["scopes"] = sorted(scopes)
    result["rows"] = [row for row in report["rows"] if row["scope"] in scopes]
    aggregates = report["aggregates"]
    incomplete = bool(
        report["omitted_observations"]
        or aggregates["omitted_memberships"]
        or aggregates["coverage_limited"]
    )
    selected = [row for row in aggregates["by_scope"] if row["scope"] in scopes]
    totals = sum_scopes(selected, incomplete=incomplete)
    result["aggregates"] = {
        "complete": not incomplete
        and totals["observations_complete"]
        and totals["measurements_complete"],
        "omitted_memberships": aggregates["omitted_memberships"],
        "coverage_limited": aggregates["coverage_limited"],
        "by_reader": [row for row in aggregates["by_reader"] if row["scope"] in scopes],
        "by_model": [row for row in aggregates["by_model"] if row["scope"] in scopes],
        "by_scope": selected,
        "totals": totals,
    }
    result["observations_complete"] = not report["omitted_observations"] and all(
        row["outcome"] not in {"pending", "unknown"} for row in result["rows"]
    )
    result["measurements_complete"] = not report["omitted_observations"] and all(
        item["complete"] for row in result["rows"] for item in row["coverage"].values()
    )
    if "token_bases" in report:
        used = {row["model"] for row in result["rows"]} | set(totals["models"])
        result["token_bases"] = [basis for basis in report["token_bases"] if basis["model"] in used]
    return result

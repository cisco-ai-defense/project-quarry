# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Versioned JSON contract for full logical-read reports and result snapshots."""

from typing import Any


def logical_read_report_schema() -> dict[str, Any]:
    """Return a fresh JSON Schema 2020-12 document for version-1 full reports.

    This describes ``ReadUsageSession.report()`` and opt-in result snapshots, not
    the lower-level detail-only ``snapshot()``. It executes no validation callbacks.
    """
    count = {"type": "integer", "minimum": 0}
    label = {"type": "string", "maxLength": 512}
    boolean = {"type": "boolean"}

    def ref(name):
        return {"$ref": "#/$defs/" + name}

    def array(items):
        return {"type": "array", "items": items}

    def obj(properties, optional=()):
        return {
            "type": "object",
            "properties": properties,
            "required": [key for key in properties if key not in optional],
            "additionalProperties": False,
        }

    counters = {"type": "object", "additionalProperties": count}
    unit = {
        "value": {"type": ["integer", "null"], "minimum": 0},
        "known_units": {"type": ["integer", "null"], "minimum": 0},
        "complete": boolean,
        "measurement_states": counters,
        "coverage_limited": boolean,
    }
    coverage = obj({"characters": ref("unit"), "text_tokens": ref("unit")})
    summary = {
        "observations": count,
        "evaluations": count,
        "unique_cuts": count,
        "cut_identity_complete": boolean,
        "invalid_measurements": count,
        "models": array(label),
        "operations": counters,
        "outcomes": counters,
        "observations_complete": boolean,
        "measurements_complete": boolean,
        "coverage": obj({"characters": ref("unit"), "text_tokens": ref("tokens")}),
        "token_bases": array(
            obj({"basis": ref("basis"), "models": array(label), "coverage": ref("unit")})
        ),
    }
    detail = {
        "scope": count,
        "rule": {"type": ["string", "null"], "maxLength": 512},
        "phase": {"enum": ["decision", "evidence", "measurement"]},
        "model": label,
        "operation": {
            "enum": ["score", "classify", "spans", "complete", "input_tokens", "split", "transform"]
        },
        "outcome": {
            "enum": [
                "pending",
                "unknown",
                "succeeded",
                "refused",
                "failed",
                "cancelled",
                "not_sent",
            ]
        },
        "observations": count,
        "unique_cuts": count,
        "cut_identity_complete": boolean,
        "invalid_measurements": count,
        "coverage": coverage,
    }
    aggregates = obj(
        {
            "complete": boolean,
            "omitted_memberships": count,
            "coverage_limited": boolean,
            "by_reader": array(
                obj(
                    {
                        "scope": count,
                        "rule": detail["rule"],
                        "phase": detail["phase"],
                        "model": label,
                        **summary,
                    }
                )
            ),
            "by_model": array(obj({"scope": count, "model": label, **summary})),
            "by_scope": array(obj({"scope": count, **summary})),
            "totals": obj(summary),
        }
    )
    report = obj(
        {
            "version": {"const": 1},
            "finalized": boolean,
            "observations_complete": boolean,
            "measurements_complete": boolean,
            "omitted_observations": count,
            "rows": array(obj(detail)),
            "aggregates": aggregates,
            "scope": {"type": ["integer", "null"], "minimum": 0},
            "scopes": array(count),
            "state": {"enum": ["scope_unavailable", "foreign_process"]},
            "token_bases": array(obj({"model": label, "name": label, "version": label})),
        },
        optional=("scope", "scopes", "state", "token_bases"),
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **report,
        "$defs": {
            "unit": obj(unit),
            "tokens": obj({**unit, "basis_compatible": boolean}),
            "basis": {
                "oneOf": [
                    {"type": "null"},
                    obj({"kind": {"const": "reference"}, "name": label, "version": label}),
                    obj({"kind": {"const": "supplied_intervals"}, "model": label}),
                ]
            },
        },
    }

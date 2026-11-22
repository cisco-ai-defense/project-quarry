# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Versioned, dependency-free JSON contract for optional window observations."""


def window_observation_report_schema() -> dict:
    """Return a fresh JSON Schema for context reports and per-result snapshots."""

    def obj(properties, optional=()):
        return {
            "type": "object",
            "properties": properties,
            "required": [key for key in properties if key not in optional],
            "additionalProperties": False,
        }

    def array(items, maximum):
        return {"type": "array", "items": items, "maxItems": maximum}

    def nullable(schema):
        return {"anyOf": [schema, {"type": "null"}]}

    def ref(name):
        return {"$ref": "#/$defs/" + name}

    count = {"type": "integer", "minimum": 0, "maximum": 1 << 63}
    label = {"type": "string", "maxLength": 512}
    boolean = {"type": "boolean"}
    scalar = obj(
        {
            "state": {
                "enum": [
                    "measured",
                    "partial",
                    "pending",
                    "unavailable",
                    "limit_exceeded",
                    "failed",
                    "cancelled",
                    "abandoned",
                ]
            },
            "value": nullable(count),
        }
    )
    row = obj(
        {
            "scope": count,
            "rule": nullable(label),
            "phase": {"enum": ["decision", "evidence", "measurement"]},
            "model": label,
            "provider_model": nullable(label),
            "operation": {"enum": ["classify", "spans"]},
            "outcome": {
                "enum": [
                    "pending",
                    "succeeded",
                    "not_sent",
                    "failed",
                    "refused",
                    "cancelled",
                    "unknown",
                ]
            },
            "unit": {"enum": ["characters", "tokens"]},
            "window_count": nullable(count),
            "source_characters": ref("coverage"),
            "covered_tokens": ref("coverage"),
            "cached": nullable(boolean),
            "state": {
                "enum": [
                    "measured",
                    "partial",
                    "pending",
                    "not_sent",
                    "failed",
                    "refused",
                    "cancelled",
                    "unknown",
                    "unavailable",
                    "limit_exceeded",
                ]
            },
            "probabilities": nullable(
                array(
                    nullable(array({"type": "number", "minimum": 0, "maximum": 1}, 1_048_576)),
                    65_536,
                )
            ),
            "probability_states": nullable(
                array(
                    {
                        "enum": [
                            "measured",
                            "unavailable",
                            "invalid_probability_vector",
                            "conflicting_probability_vector",
                            "limit_exceeded",
                        ]
                    },
                    65_536,
                )
            ),
            "reads": count,
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **obj(
            {
                "version": {"const": 1, "type": "integer"},
                "enabled": boolean,
                "finalized": boolean,
                "rows": array(ref("row"), 65_536),
                "omitted": count,
                "scopes": array(count, 65_536),
                "state": {"enum": ["collected", "scope_unavailable"]},
            },
            optional=("scopes", "state"),
        ),
        "$defs": {"coverage": scalar, "row": row},
    }

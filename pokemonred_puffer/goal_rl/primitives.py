"""Shared primitives library for rubric criteria and milestone predicates.

A primitive is a pure function over a PrimitiveContext (current snapshot +
optional history) that returns a float in [0, 1] (for criteria) or a 0/1 value
(for milestone predicates — really just a float that's conventionally 0.0 or 1.0).

The LLM revision engine can only compose criteria and milestone predicates from
this library. Anything outside the library fails validation.

Design reference: docs/goal_rl.md C03 + C07.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

from pokemonred_puffer.rubric_rl.rubrics import GameStateSnapshot


# --- Context --------------------------------------------------------------


@dataclass
class PrimitiveContext:
    """Bundle of data that primitives can read.

    - snapshot: the game state being evaluated.
    - prev_snapshot: the previous eval-time snapshot (for delta / increment).
    - history: all snapshots seen this run (for first_time / rolling_avg).
    - epsilon: divide-by-zero guard.
    """

    snapshot: GameStateSnapshot
    prev_snapshot: GameStateSnapshot | None = None
    history: list[GameStateSnapshot] = field(default_factory=list)
    epsilon: float = 1e-8


# --- Field resolution -----------------------------------------------------


VALID_SNAPSHOT_FIELDS: set[str] = {f.name for f in dataclasses.fields(GameStateSnapshot)}

# Fields whose numeric interpretation is well-defined. Lists (e.g. party_levels)
# are excluded from most primitives.
NUMERIC_SNAPSHOT_FIELDS: set[str] = {
    f.name for f in dataclasses.fields(GameStateSnapshot) if f.type in ("int", "float", int, float)
}


def _get_numeric(snapshot: GameStateSnapshot, field_name: str) -> float:
    """Get a numeric field from a snapshot.

    Raises ValueError if the field is unknown or non-numeric.
    """
    if field_name not in VALID_SNAPSHOT_FIELDS:
        raise ValueError(f"Unknown snapshot field: {field_name!r}")
    value = getattr(snapshot, field_name)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"Field {field_name!r} is a sequence; primitives only accept numeric fields"
        )
    raise ValueError(f"Field {field_name!r} has non-numeric type {type(value).__name__}")


# --- Primitives -----------------------------------------------------------
#
# Each primitive takes the context as its first argument, then named kwargs.
# All return a float. Milestone-style primitives return 0.0 or 1.0.


def fraction(ctx: PrimitiveContext, field: str, max: float) -> float:
    """min(x / max, 1.0). Returns value in [0, 1]. Named `max` to match caller intuition."""
    x = _get_numeric(ctx.snapshot, field)
    if max <= 0 or x <= 0:
        return 0.0
    return min(x / max, 1.0)


def inverse_fraction(ctx: PrimitiveContext, field: str, max: float) -> float:
    """1.0 - min(x / max, 1.0). Use when low values are good (e.g. blackouts, steps used)."""
    x = _get_numeric(ctx.snapshot, field)
    if max <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    return 1.0 - min(x / max, 1.0)


def delta(ctx: PrimitiveContext, field: str) -> float:
    """x_current - x_prev_eval.  Returns 0.0 if no previous eval snapshot."""
    if ctx.prev_snapshot is None:
        return 0.0
    cur = _get_numeric(ctx.snapshot, field)
    prev = _get_numeric(ctx.prev_snapshot, field)
    return cur - prev


def ratio(ctx: PrimitiveContext, field_a: str, field_b: str) -> float:
    """a / (b + eps)."""
    a = _get_numeric(ctx.snapshot, field_a)
    b = _get_numeric(ctx.snapshot, field_b)
    return a / (b + ctx.epsilon)


def threshold_hit(ctx: PrimitiveContext, field: str, thresh: float) -> float:
    """1.0 if field >= thresh else 0.0."""
    return 1.0 if _get_numeric(ctx.snapshot, field) >= thresh else 0.0


def threshold_cross(ctx: PrimitiveContext, field: str, thresh: float) -> float:
    """1.0 if field crossed thresh upward since last eval, else 0.0.

    Crossed means: prev < thresh <= current.  No-op if no prev snapshot.
    """
    if ctx.prev_snapshot is None:
        return 0.0
    prev = _get_numeric(ctx.prev_snapshot, field)
    cur = _get_numeric(ctx.snapshot, field)
    return 1.0 if prev < thresh <= cur else 0.0


def first_time(ctx: PrimitiveContext, field: str) -> float:
    """1.0 if field is > 0 for the first time in the run's history.

    Uses ctx.history.  If history is empty, treats the current snapshot as the
    first observation (returns 1.0 iff current > 0).
    """
    cur = _get_numeric(ctx.snapshot, field)
    if cur <= 0:
        return 0.0
    for earlier in ctx.history:
        if _get_numeric(earlier, field) > 0:
            return 0.0
    return 1.0


def increment(ctx: PrimitiveContext, field: str) -> float:
    """1.0 if field increased since the previous eval snapshot, else 0.0."""
    if ctx.prev_snapshot is None:
        return 0.0
    cur = _get_numeric(ctx.snapshot, field)
    prev = _get_numeric(ctx.prev_snapshot, field)
    return 1.0 if cur > prev else 0.0


def rolling_avg(ctx: PrimitiveContext, field: str, window: int) -> float:
    """Trailing mean of field over the last `window` snapshots in history.

    The current snapshot is appended to the tail so the mean always includes
    the present value.  Returns the current value if history is empty.
    """
    if window <= 0:
        return _get_numeric(ctx.snapshot, field)
    tail = ctx.history[-(window - 1) :] if window > 1 else []
    vals = [_get_numeric(s, field) for s in tail]
    vals.append(_get_numeric(ctx.snapshot, field))
    return sum(vals) / len(vals)


# --- Registry -------------------------------------------------------------


PRIMITIVES: dict[str, Callable[..., float]] = {
    "fraction": fraction,
    "inverse_fraction": inverse_fraction,
    "delta": delta,
    "ratio": ratio,
    "threshold_hit": threshold_hit,
    "threshold_cross": threshold_cross,
    "first_time": first_time,
    "increment": increment,
    "rolling_avg": rolling_avg,
}

# Signature info for the validator: which kwargs each primitive requires and
# which of those are snapshot-field names (must be in VALID_SNAPSHOT_FIELDS).
PRIMITIVE_SIGNATURES: dict[str, dict[str, Any]] = {
    "fraction": {"fields": {"field"}, "required": {"field", "max"}},
    "inverse_fraction": {"fields": {"field"}, "required": {"field", "max"}},
    "delta": {"fields": {"field"}, "required": {"field"}},
    "ratio": {"fields": {"field_a", "field_b"}, "required": {"field_a", "field_b"}},
    "threshold_hit": {"fields": {"field"}, "required": {"field", "thresh"}},
    "threshold_cross": {"fields": {"field"}, "required": {"field", "thresh"}},
    "first_time": {"fields": {"field"}, "required": {"field"}},
    "increment": {"fields": {"field"}, "required": {"field"}},
    "rolling_avg": {"fields": {"field"}, "required": {"field", "window"}},
}


# --- Specification ---------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveCall:
    """A serializable spec for a primitive invocation.

    LLM proposals will be deserialized into PrimitiveCall instances; the
    validator checks the name + args before we build an evaluator.
    """

    primitive: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"primitive": self.primitive, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrimitiveCall":
        return cls(primitive=str(d["primitive"]), args=dict(d.get("args", {})))


class PrimitiveValidationError(ValueError):
    """Raised when a PrimitiveCall is malformed or references unknown fields."""


def validate_primitive_call(call: PrimitiveCall) -> None:
    """Raise PrimitiveValidationError if the call is invalid.

    Checks:
    - primitive name is registered.
    - all required kwargs are present.
    - field kwargs reference known snapshot fields.
    """
    if call.primitive not in PRIMITIVES:
        raise PrimitiveValidationError(
            f"Unknown primitive {call.primitive!r}; known: {sorted(PRIMITIVES)}"
        )

    sig = PRIMITIVE_SIGNATURES[call.primitive]
    missing = sig["required"] - set(call.args)
    if missing:
        raise PrimitiveValidationError(
            f"Primitive {call.primitive!r} missing required args: {sorted(missing)}"
        )

    for field_kw in sig["fields"]:
        if field_kw in call.args:
            field_name = call.args[field_kw]
            if not isinstance(field_name, str):
                raise PrimitiveValidationError(
                    f"Arg {field_kw!r} must be a string (field name), got {type(field_name).__name__}"
                )
            if field_name not in VALID_SNAPSHOT_FIELDS:
                raise PrimitiveValidationError(
                    f"Arg {field_kw}={field_name!r} is not a known GameStateSnapshot field"
                )


def build_evaluator(call: PrimitiveCall) -> Callable[[PrimitiveContext], float]:
    """Return a callable that takes a PrimitiveContext and returns a float.

    Call must already be validated.
    """
    fn = PRIMITIVES[call.primitive]
    args = dict(call.args)
    return lambda ctx: float(fn(ctx, **args))


# --- Delta-based automatic milestone detection -----------------------------


# Fields whose increment signals a milestone candidate without LLM involvement.
# These are the "discovery" events of Pokemon Red that an informed new player
# would notice on their own.  Populated from the snapshot schema at import
# time: every integer field except step counters and normalization totals.
AUTO_MILESTONE_FIELDS: tuple[str, ...] = (
    "badges",
    "completed_required_events",
    "completed_required_items",
    "completed_useful_items",
    "hm_count",
    "party_count",
    "max_level_sum",
    "seen_pokemon_count",
    "caught_pokemon_count",
    "unique_moves",
    "unique_maps_visited",
    "npcs_talked",
    "hidden_objs_found",
    "signs_read",
    "pokecenter_heals",
    "cut_tiles_used",
    "surf_tiles_used",
)


def detect_milestones(ctx: PrimitiveContext) -> list[dict[str, Any]]:
    """Return a list of milestone-candidate dicts based on the context.

    Each dict is of the form:
        {"kind": "first_time" | "increment", "field": <name>, "value": <current>}

    These are the objective, no-LLM milestones (C07 "delta-based auto-detection").
    The LLM's proposed predicates run separately in the revision engine.
    """
    out: list[dict[str, Any]] = []
    for fname in AUTO_MILESTONE_FIELDS:
        try:
            cur = _get_numeric(ctx.snapshot, fname)
        except ValueError:
            continue

        # first_time: current > 0 and no prior history had > 0
        if first_time(ctx, field=fname) > 0:
            out.append({"kind": "first_time", "field": fname, "value": cur})
            continue

        # increment: strict increase since last eval
        if increment(ctx, field=fname) > 0:
            prev = _get_numeric(ctx.prev_snapshot, fname) if ctx.prev_snapshot is not None else 0.0
            out.append(
                {
                    "kind": "increment",
                    "field": fname,
                    "value": cur,
                    "previous": prev,
                }
            )
    return out


__all__ = [
    "PrimitiveContext",
    "PrimitiveCall",
    "PrimitiveValidationError",
    "PRIMITIVES",
    "PRIMITIVE_SIGNATURES",
    "VALID_SNAPSHOT_FIELDS",
    "NUMERIC_SNAPSHOT_FIELDS",
    "AUTO_MILESTONE_FIELDS",
    "validate_primitive_call",
    "build_evaluator",
    "detect_milestones",
    # primitive functions (exported for direct use in tests)
    "fraction",
    "inverse_fraction",
    "delta",
    "ratio",
    "threshold_hit",
    "threshold_cross",
    "first_time",
    "increment",
    "rolling_avg",
]

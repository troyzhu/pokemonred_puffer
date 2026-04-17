"""LLM-driven revision engine.

The revision engine is called when a trigger fires (milestone, plateau, or
revision-ceiling).  It builds a context bundle, asks the LLM to propose a
bounded set of edits, parses and validates the response, and returns a
structured RevisionProposal that the GoalManager can apply.

Safety is enforced by the validator, not by trust: every edit is checked
against the primitives library and the weight bounds declared in
goal_manager.  Anything off-schema is rejected with a reason.

Design reference: docs/goal_rl.md C03.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pokemonred_puffer.rubric_rl.state_summarizer import StateSummarizer
from pokemonred_puffer.rubric_rl.rubrics import GameStateSnapshot

from pokemonred_puffer.goal_rl.goal_manager import (
    LAYER_TWO_DELTA_MAX,
    LAYER_TWO_DELTA_MIN,
    SUB_WEIGHT_MAX,
    SUB_WEIGHT_MIN,
    CriterionSpec,
    GoalManager,
)
from pokemonred_puffer.goal_rl.primitives import (
    PRIMITIVE_SIGNATURES,
    PrimitiveCall,
    PrimitiveValidationError,
    VALID_SNAPSHOT_FIELDS,
    validate_primitive_call,
)
from pokemonred_puffer.goal_rl.schema import (
    CATEGORY_DESCRIPTIONS,
    GoalCategory,
)
from pokemonred_puffer.goal_rl.triggers import (
    MilestonePredicate,
    TriggerEvent,
    validate_milestone_predicate,
)

logger = logging.getLogger(__name__)


# --- Data model ------------------------------------------------------------


@dataclass
class RevisionContext:
    """Everything the LLM sees when proposing a revision."""

    trigger: TriggerEvent
    constitution_text: str
    layer_one_prompt: str
    layer_two_weights: dict[str, float]
    layer_three: list[dict[str, Any]]
    custom_criteria: list[dict[str, Any]]
    current_milestone_predicates: list[dict[str, Any]]
    # Recent history.
    training_reward_trajectory: list[float]
    e2_trajectory: list[float]
    dashboard_latest: dict[str, float]
    dashboard_deltas: dict[str, float]
    # Optional: narrative from prior revision.
    prior_narrative: str | None
    # Snapshot summary (human-readable, reuses StateSummarizer).
    state_summary: str


@dataclass
class RevisionProposal:
    """Structured, validated output ready for GoalManager.apply_revision()."""

    rationale: str
    narrative: str | None
    layer_two_deltas: dict[str, float]
    added_criteria: list[CriterionSpec]
    removed_criterion_names: list[str]
    toggled_criteria: dict[str, bool]
    milestone_predicates: list[MilestonePredicate]
    # Issues encountered while parsing (non-fatal).  Each is a short string.
    warnings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.layer_two_deltas
            or self.added_criteria
            or self.removed_criterion_names
            or self.toggled_criteria
            or self.milestone_predicates
        )


class RevisionValidationError(ValueError):
    """Raised when an LLM response cannot be validated into a RevisionProposal."""


# --- Prompt template -------------------------------------------------------


_PROMPT = """You are the coach for a Pokemon Red reinforcement-learning agent. \
Every few epochs you see how the agent has been performing and you propose small, \
bounded adjustments to the rubric that shapes its training reward. You are NOT the \
agent; you are a careful, conservative supervisor.

# Current state

## Constitution (designer's intent — do not contradict)
{constitution}

## Core values (always active — anchor your reasoning)
{layer_one}

## Layer-2 session-goal weights (current)
{layer_two}

## Layer-3 constraints (frozen — you cannot edit these)
{layer_three}

## Custom criteria you've added previously
{custom_criteria}

## Milestone predicates you've added previously
{milestone_predicates}

## Latest signals
- Training-rubric mean reward, last {n_train} epochs:
  {training_trajectory}
- Frozen evaluator (E2) mean, last {n_e2} evals:
  {e2_trajectory}
- Dashboard latest values:
  {dashboard_latest}
- Dashboard deltas since last revision:
  {dashboard_deltas}

## What triggered this revision
- Trigger: {trigger_kind} — {trigger_detail}
{milestones_section}

## Your previous note-to-self (if any)
{prior_narrative}

## State summary
{state_summary}

# What you can do (allowed edits only)

1. **Multiplicative deltas on Layer-2 category weights.** Each delta in [{l2_min}, {l2_max}]. \
Only the seven categories below are legal — no new categories in V1.
2. **Add custom criteria**, each composed from the primitives library. Weight in [{w_min}, {w_max}].
3. **Remove** custom criteria you previously added (by name).
4. **Toggle** built-in criteria on/off via keys of form `"category:criterion_name"`.
5. **Add milestone predicates** composed from the primitives library.

You may NOT: write arbitrary code; invent state fields; add Layer-2 categories; \
edit Layer-1 or Layer-3; touch the frozen evaluator (E2).

# Legal Layer-2 categories
{category_list}

# Primitives library (the only operations you may compose)
{primitives_list}

# Snapshot fields you may reference
{snapshot_fields}

# Output — JSON only, no preamble, no explanation outside the JSON

{{
  "rationale": "2-3 sentences explaining the reasoning",
  "narrative": "1-2 sentences of note-to-self for your next revision",
  "layer_two_deltas": {{"<category>": <multiplicative_delta_in_range>, ...}},
  "added_criteria": [
    {{
      "name": "<short_snake_case_name>",
      "description": "<what it measures>",
      "weight": <float_in_[{w_min}, {w_max}]>,
      "primitive_call": {{"primitive": "<name>", "args": {{...}}}},
      "category": "<one of the 7 legal categories>"
    }}, ...
  ],
  "removed_criteria": ["name1", ...],
  "toggled_criteria": {{"<category>:<criterion_name>": true|false, ...}},
  "milestone_predicates": [
    {{
      "name": "<short_name>",
      "description": "<what it signals>",
      "primitive_call": {{"primitive": "<name>", "args": {{...}}}}
    }}, ...
  ]
}}

Omit any field whose value is empty.  Prefer smaller, incremental changes.  \
If things are going well, return mostly empty objects with a rationale stating so.
"""


# --- Prompt building -------------------------------------------------------


_MAX_TRAJECTORY_LEN = 12


def _format_trajectory(xs: list[float]) -> str:
    if not xs:
        return "(no history yet)"
    xs = xs[-_MAX_TRAJECTORY_LEN:]
    return ", ".join(f"{x:.4f}" for x in xs)


def _format_weights(d: dict[str, float]) -> str:
    if not d:
        return "(none)"
    return ", ".join(f"{k}={v:.3f}" for k, v in sorted(d.items()))


def _format_dashboard(d: dict[str, float]) -> str:
    if not d:
        return "(no data yet)"
    items = sorted(d.items())
    return ", ".join(f"{k}={v:.2f}" for k, v in items)


def _format_list(items: list[Any]) -> str:
    if not items:
        return "(none)"
    return json.dumps(items, indent=2, default=str)


def _format_milestones_section(trigger: TriggerEvent) -> str:
    if trigger.kind != "milestone" or not trigger.milestones:
        return ""
    lines = ["- Milestones detected this eval:"]
    for m in trigger.milestones:
        lines.append(f"    - {m}")
    return "\n".join(lines)


def _primitives_list() -> str:
    """Pretty list of primitives + required args for the prompt."""
    out = []
    for name, sig in sorted(PRIMITIVE_SIGNATURES.items()):
        req = sorted(sig["required"])
        out.append(f'- "{name}": args={req}')
    return "\n".join(out)


def _category_list() -> str:
    out = []
    for cat in GoalCategory:
        out.append(f'- "{cat.value}": {CATEGORY_DESCRIPTIONS[cat]}')
    return "\n".join(out)


def build_prompt(ctx: RevisionContext) -> str:
    return _PROMPT.format(
        constitution=ctx.constitution_text or "(none)",
        layer_one=ctx.layer_one_prompt,
        layer_two=_format_weights(ctx.layer_two_weights),
        layer_three=_format_list(ctx.layer_three),
        custom_criteria=_format_list(ctx.custom_criteria),
        milestone_predicates=_format_list(ctx.current_milestone_predicates),
        n_train=min(len(ctx.training_reward_trajectory), _MAX_TRAJECTORY_LEN),
        training_trajectory=_format_trajectory(ctx.training_reward_trajectory),
        n_e2=min(len(ctx.e2_trajectory), _MAX_TRAJECTORY_LEN),
        e2_trajectory=_format_trajectory(ctx.e2_trajectory),
        dashboard_latest=_format_dashboard(ctx.dashboard_latest),
        dashboard_deltas=_format_dashboard(ctx.dashboard_deltas),
        trigger_kind=ctx.trigger.kind,
        trigger_detail=ctx.trigger.detail,
        milestones_section=_format_milestones_section(ctx.trigger),
        prior_narrative=ctx.prior_narrative or "(none)",
        state_summary=ctx.state_summary,
        l2_min=LAYER_TWO_DELTA_MIN,
        l2_max=LAYER_TWO_DELTA_MAX,
        w_min=SUB_WEIGHT_MIN,
        w_max=SUB_WEIGHT_MAX,
        category_list=_category_list(),
        primitives_list=_primitives_list(),
        snapshot_fields=", ".join(sorted(VALID_SNAPSHOT_FIELDS)),
    )


# --- JSON extraction and parsing ------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    for attempt in (text, _strip_fences(text), _first_json_block(text)):
        if attempt is None:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    raise RevisionValidationError(f"Could not parse JSON from LLM response: {text[:300]!r}")


def _strip_fences(text: str) -> str | None:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else None


def _first_json_block(text: str) -> str | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


# --- Validation -----------------------------------------------------------


def _validate_layer_two_deltas(raw: Any, warnings: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not raw:
        return out
    if not isinstance(raw, dict):
        warnings.append(f"layer_two_deltas must be an object, got {type(raw).__name__}")
        return out
    for k, v in raw.items():
        try:
            cat = GoalCategory(k)
        except ValueError:
            warnings.append(f"Ignoring unknown Layer-2 category in deltas: {k!r}")
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            warnings.append(f"Layer-2 delta for {k} not numeric: {v!r}")
            continue
        if not (LAYER_TWO_DELTA_MIN <= val <= LAYER_TWO_DELTA_MAX):
            warnings.append(
                f"Layer-2 delta for {k}={val} outside "
                f"[{LAYER_TWO_DELTA_MIN}, {LAYER_TWO_DELTA_MAX}]; clipping"
            )
            val = max(LAYER_TWO_DELTA_MIN, min(val, LAYER_TWO_DELTA_MAX))
        out[cat.value] = val
    return out


def _validate_criterion_entry(entry: Any, warnings: list[str]) -> CriterionSpec | None:
    if not isinstance(entry, dict):
        warnings.append(f"Criterion entry not an object: {entry!r}")
        return None

    required = {"name", "weight", "primitive_call", "category"}
    missing = required - set(entry)
    if missing:
        warnings.append(f"Criterion missing fields {sorted(missing)}: {entry!r}")
        return None

    try:
        call = PrimitiveCall.from_dict(entry["primitive_call"])
        validate_primitive_call(call)
    except (PrimitiveValidationError, ValueError, KeyError) as e:
        warnings.append(f"Invalid primitive_call for criterion {entry.get('name')!r}: {e}")
        return None

    try:
        weight = float(entry["weight"])
    except (TypeError, ValueError):
        warnings.append(f"Criterion weight not numeric: {entry.get('weight')!r}")
        return None
    if not (SUB_WEIGHT_MIN <= weight <= SUB_WEIGHT_MAX):
        warnings.append(f"Criterion {entry.get('name')} weight {weight} out of bounds; clipping")
        weight = max(SUB_WEIGHT_MIN, min(weight, SUB_WEIGHT_MAX))

    try:
        cat = GoalCategory(str(entry["category"]))
    except ValueError:
        warnings.append(f"Criterion category unknown: {entry.get('category')!r}")
        return None

    name = str(entry["name"]).strip()
    if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        warnings.append(f"Criterion name not a valid identifier: {name!r}")
        return None

    return CriterionSpec(
        name=name,
        description=str(entry.get("description", "")),
        weight=weight,
        primitive_call=call,
        category=cat,
        source="llm_revision",
    )


def _validate_milestone_entry(entry: Any, warnings: list[str]) -> MilestonePredicate | None:
    if not isinstance(entry, dict):
        warnings.append(f"Milestone entry not an object: {entry!r}")
        return None
    required = {"name", "primitive_call"}
    missing = required - set(entry)
    if missing:
        warnings.append(f"Milestone missing fields {sorted(missing)}: {entry!r}")
        return None

    try:
        call = PrimitiveCall.from_dict(entry["primitive_call"])
        pred = MilestonePredicate(
            name=str(entry["name"]).strip(),
            description=str(entry.get("description", "")),
            primitive_call=call,
        )
        validate_milestone_predicate(pred)
    except (PrimitiveValidationError, ValueError, KeyError) as e:
        warnings.append(f"Invalid milestone predicate {entry.get('name')!r}: {e}")
        return None

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", pred.name):
        warnings.append(f"Milestone name not a valid identifier: {pred.name!r}")
        return None
    return pred


def _validate_toggles(raw: Any, warnings: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not raw:
        return out
    if not isinstance(raw, dict):
        warnings.append(f"toggled_criteria must be an object, got {type(raw).__name__}")
        return out
    for k, v in raw.items():
        if ":" not in str(k):
            warnings.append(f"Toggle key {k!r} must be 'category:name'; skipping")
            continue
        cat_name = str(k).split(":", 1)[0]
        try:
            GoalCategory(cat_name)
        except ValueError:
            warnings.append(f"Unknown toggle category in key {k!r}; skipping")
            continue
        if not isinstance(v, bool):
            warnings.append(f"Toggle value for {k!r} not bool: {v!r}; coercing")
            v = bool(v)
        out[str(k)] = v
    return out


def parse_revision(response_text: str) -> RevisionProposal:
    """Extract and validate a RevisionProposal from raw LLM response text.

    Raises RevisionValidationError only if no JSON can be recovered at all.
    Malformed individual edits are dropped with warnings, not a hard failure.
    """
    parsed = _extract_json(response_text)
    warnings: list[str] = []

    rationale = str(parsed.get("rationale", "")).strip()
    narrative = parsed.get("narrative")
    narrative = str(narrative).strip() if narrative else None

    l2 = _validate_layer_two_deltas(parsed.get("layer_two_deltas"), warnings)

    added_specs: list[CriterionSpec] = []
    for entry in parsed.get("added_criteria", []) or []:
        spec = _validate_criterion_entry(entry, warnings)
        if spec is not None:
            added_specs.append(spec)
    # Deduplicate by name.
    seen: set[str] = set()
    deduped: list[CriterionSpec] = []
    for s in added_specs:
        if s.name in seen:
            warnings.append(f"Duplicate added criterion name {s.name!r}; keeping first")
            continue
        seen.add(s.name)
        deduped.append(s)
    added_specs = deduped

    removed_names: list[str] = []
    for n in parsed.get("removed_criteria", []) or []:
        if isinstance(n, str):
            removed_names.append(n)
        else:
            warnings.append(f"Removed-criterion entry not a string: {n!r}")

    toggles = _validate_toggles(parsed.get("toggled_criteria"), warnings)

    milestones: list[MilestonePredicate] = []
    for entry in parsed.get("milestone_predicates", []) or []:
        pred = _validate_milestone_entry(entry, warnings)
        if pred is not None:
            milestones.append(pred)

    return RevisionProposal(
        rationale=rationale,
        narrative=narrative,
        layer_two_deltas=l2,
        added_criteria=added_specs,
        removed_criterion_names=removed_names,
        toggled_criteria=toggles,
        milestone_predicates=milestones,
        warnings=warnings,
    )


# --- Engine ---------------------------------------------------------------


@dataclass
class RevisionEngineConfig:
    """Config for the engine."""

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1200
    # If True, bypass LLM and return an empty revision (for smoke tests).
    dry_run: bool = False


class RevisionEngine:
    """Calls the LLM and returns a validated RevisionProposal."""

    def __init__(
        self,
        llm_client: Any = None,
        config: RevisionEngineConfig | None = None,
    ):
        self.client = llm_client
        self.config = config or RevisionEngineConfig()
        self._summarizer = StateSummarizer()

    # --- Public entry ---------------------------------------------------

    def propose(
        self,
        trigger: TriggerEvent,
        manager: GoalManager,
        training_reward_trajectory: list[float],
        e2_trajectory: list[float],
        latest_snapshot: GameStateSnapshot,
        prev_eval_snapshot: GameStateSnapshot | None,
        milestone_predicates: list[MilestonePredicate],
    ) -> RevisionProposal:
        """Build context, call LLM (or dry-run), parse, validate, return."""
        ctx = self._build_context(
            trigger=trigger,
            manager=manager,
            training_reward_trajectory=training_reward_trajectory,
            e2_trajectory=e2_trajectory,
            latest_snapshot=latest_snapshot,
            prev_eval_snapshot=prev_eval_snapshot,
            milestone_predicates=milestone_predicates,
        )

        if self.config.dry_run or self.client is None:
            logger.info("Revision engine: dry-run mode, returning empty proposal")
            return RevisionProposal(
                rationale="(dry-run: no LLM call)",
                narrative=None,
                layer_two_deltas={},
                added_criteria=[],
                removed_criterion_names=[],
                toggled_criteria={},
                milestone_predicates=[],
            )

        prompt = build_prompt(ctx)
        try:
            response_text = self._call_llm(prompt)
        except Exception as e:  # Catch-all: LLM failure shouldn't crash training.
            logger.warning("Revision LLM call failed: %s", e)
            return RevisionProposal(
                rationale=f"(LLM call failed: {e})",
                narrative=None,
                layer_two_deltas={},
                added_criteria=[],
                removed_criterion_names=[],
                toggled_criteria={},
                milestone_predicates=[],
                warnings=[f"llm_call_failed: {e}"],
            )

        try:
            return parse_revision(response_text)
        except RevisionValidationError as e:
            logger.warning("Revision JSON unparseable: %s", e)
            return RevisionProposal(
                rationale="(revision JSON unparseable)",
                narrative=None,
                layer_two_deltas={},
                added_criteria=[],
                removed_criterion_names=[],
                toggled_criteria={},
                milestone_predicates=[],
                warnings=[f"parse_failed: {e}"],
            )

    # --- Internals ------------------------------------------------------

    def _build_context(
        self,
        trigger: TriggerEvent,
        manager: GoalManager,
        training_reward_trajectory: list[float],
        e2_trajectory: list[float],
        latest_snapshot: GameStateSnapshot,
        prev_eval_snapshot: GameStateSnapshot | None,
        milestone_predicates: list[MilestonePredicate],
    ) -> RevisionContext:
        dashboard_latest = {
            f: float(getattr(latest_snapshot, f, 0.0))
            for f in (
                "badges",
                "completed_required_events",
                "hm_count",
                "party_count",
                "max_level_sum",
                "caught_pokemon_count",
                "seen_pokemon_count",
                "unique_maps_visited",
                "blackout_count",
                "total_steps",
            )
        }
        if prev_eval_snapshot is not None:
            dashboard_deltas = {
                f: dashboard_latest[f] - float(getattr(prev_eval_snapshot, f, 0.0))
                for f in dashboard_latest
            }
        else:
            dashboard_deltas = {}

        prior_narrative: str | None = None
        for rec in reversed(manager.revision_log):
            if rec.narrative:
                prior_narrative = rec.narrative
                break

        return RevisionContext(
            trigger=trigger,
            constitution_text=manager.config.constitution.raw_text or "",
            layer_one_prompt=manager.config.layer_one.to_prompt_fragment(),
            layer_two_weights=manager.config.layer_two.to_dict(),
            layer_three=manager.config.layer_three.to_list(),
            custom_criteria=manager.current_criterion_specs(),
            current_milestone_predicates=[p.to_dict() for p in milestone_predicates],
            training_reward_trajectory=list(training_reward_trajectory),
            e2_trajectory=list(e2_trajectory),
            dashboard_latest=dashboard_latest,
            dashboard_deltas=dashboard_deltas,
            prior_narrative=prior_narrative,
            state_summary=self._summarizer.summarize(latest_snapshot),
        )

    def _call_llm(self, prompt: str) -> str:
        """Dispatch to anthropic or openai based on client type."""
        if hasattr(self.client, "messages"):  # anthropic
            resp = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:  # openai-like
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
            )
            return resp.choices[0].message.content


__all__ = [
    "RevisionContext",
    "RevisionProposal",
    "RevisionValidationError",
    "RevisionEngineConfig",
    "RevisionEngine",
    "parse_revision",
    "build_prompt",
]

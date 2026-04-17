"""Three-layer value schema and constitution parsing.

Layer 1 — Core values (abstract, always active): competence + autonomy (SDT).
Layer 2 — Session-configurable primary goals (weighted, closed menu in V1).
Layer 3 — Constraints (Nuzlocke-style rule sets, separate from goals).

Constitution (C08): designer writes free text; a one-time LLM call parses it
into structured fields at run start.  We retain both the raw text and the
parsed structure; the revision engine sees both on every call.

Design reference: docs/goal_rl.md C01 + C08.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# --- Enums ----------------------------------------------------------------


class GoalCategory(str, Enum):
    """Closed menu of Layer-2 session-goal categories for V1.

    See plan C01 for definitions.  LLM revisions may reweight but not add
    new categories in V1; V2 will open this via a designer-approval gate.
    """

    PROGRESS = "progress"
    COMPLETENESS = "completeness"
    MASTERY = "mastery"
    DISCOVERY = "discovery"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    DIVERSITY = "diversity"

    @classmethod
    def all(cls) -> list["GoalCategory"]:
        return list(cls)


class ConstraintKind(str, Enum):
    NUZLOCKE = "nuzlocke"
    MONOTYPE = "monotype"
    LEVEL_CAP = "level_cap"
    NO_ITEMS_IN_BATTLE = "no_items_in_battle"
    CUSTOM = "custom"


CATEGORY_DESCRIPTIONS: dict[GoalCategory, str] = {
    GoalCategory.PROGRESS: "advancing the story, earning badges, clearing required events",
    GoalCategory.COMPLETENESS: "catching Pokemon, pokedex coverage, collecting items",
    GoalCategory.MASTERY: "team strength, battle wins, type coverage, level parity",
    GoalCategory.DISCOVERY: "exploring maps/tiles, NPCs, hidden objects",
    GoalCategory.EFFICIENCY: "minimizing steps and wall-clock time per milestone",
    GoalCategory.SAFETY: "avoiding blackouts, preserving HP, no soft-locks",
    GoalCategory.DIVERSITY: "varied team composition, move usage, anti-repetition",
}


# --- Dataclasses ----------------------------------------------------------


@dataclass(frozen=True)
class LayerOne:
    """Core values: SDT-derived, always on, anchor LLM semantics.

    Not really user-configurable in V1 — these are the conceptual frame that
    the LLM revision engine sees on every call.
    """

    competence_active: bool = True
    autonomy_active: bool = True

    def to_prompt_fragment(self) -> str:
        parts = []
        if self.competence_active:
            parts.append("competence (the agent feels effective at what it pursues)")
        if self.autonomy_active:
            parts.append("autonomy (the agent pursues self-selected goals)")
        return "Core values (always active): " + "; ".join(parts) + "."


@dataclass
class LayerTwo:
    """Weighted session-goals over the closed V1 menu.

    Weights are floats in [0, 10]. Categories not present default to 0.
    """

    weights: dict[GoalCategory, float] = field(default_factory=dict)

    def __post_init__(self):
        # Canonicalize weights; clip to [0, 10]; missing categories default to 0.
        canon: dict[GoalCategory, float] = {cat: 0.0 for cat in GoalCategory}
        for k, v in self.weights.items():
            cat = k if isinstance(k, GoalCategory) else GoalCategory(str(k))
            canon[cat] = max(0.0, min(float(v), 10.0))
        self.weights = canon

    def active_categories(self) -> list[GoalCategory]:
        return [cat for cat, w in self.weights.items() if w > 0]

    def to_dict(self) -> dict[str, float]:
        return {cat.value: w for cat, w in self.weights.items()}

    @classmethod
    def equal_weights(cls) -> "LayerTwo":
        return cls(weights={cat: 1.0 for cat in GoalCategory})

    @classmethod
    def from_dict(cls, d: dict[str, float] | None) -> "LayerTwo":
        if not d:
            return cls.equal_weights()
        parsed: dict[GoalCategory, float] = {}
        for name, w in d.items():
            try:
                parsed[GoalCategory(name)] = float(w)
            except (ValueError, TypeError):
                logger.warning("Ignoring unknown goal category %r in weights", name)
        if not parsed:
            return cls.equal_weights()
        return cls(weights=parsed)


@dataclass
class Constraint:
    kind: ConstraintKind
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Constraint":
        return cls(
            kind=ConstraintKind(d["kind"]),
            params=dict(d.get("params", {})),
        )


@dataclass
class LayerThree:
    constraints: list[Constraint] = field(default_factory=list)

    def has(self, kind: ConstraintKind) -> bool:
        return any(c.kind == kind for c in self.constraints)

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.constraints]

    @classmethod
    def from_list(cls, items: list[dict[str, Any]] | None) -> "LayerThree":
        if not items:
            return cls()
        constraints: list[Constraint] = []
        for item in items:
            try:
                constraints.append(Constraint.from_dict(item))
            except (ValueError, KeyError) as e:
                logger.warning("Ignoring malformed constraint %r: %s", item, e)
        return cls(constraints=constraints)


@dataclass
class Constitution:
    """Designer's written intent plus the structured extraction."""

    raw_text: str
    playstyle: str | None = None
    target_badges: int | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "playstyle": self.playstyle,
            "target_badges": self.target_badges,
            "notes": self.notes,
        }


@dataclass
class GoalRLConfig:
    """Fully resolved config for a goal-RL run."""

    constitution: Constitution
    layer_one: LayerOne
    layer_two: LayerTwo
    layer_three: LayerThree

    def to_dict(self) -> dict[str, Any]:
        return {
            "constitution": self.constitution.to_dict(),
            "layer_one": asdict(self.layer_one),
            "layer_two": self.layer_two.to_dict(),
            "layer_three": self.layer_three.to_list(),
        }


# --- Constitution parsing -------------------------------------------------


_PARSER_PROMPT = """You are parsing a designer's constitution for a Pokemon Red \
reinforcement-learning agent. The designer has written a short natural-language \
statement describing what they want the agent to do. Extract structured fields.

# Output schema (return JSON only, no preamble)

{{
  "playstyle": one of ["story_runner", "pokedex_hunter", "nuzlocker", "speedrunner", "explorer", "custom"] or null,
  "target_badges": integer in [0, 8] or null,
  "weights": object mapping goal category to weight (float in [0, 10]); include only categories the designer emphasizes,
  "constraints": array of {{"kind": string, "params": object}},
  "notes": short string summarizing any guidance not captured above
}}

# Goal categories
{category_list}

# Constraint kinds
- "nuzlocke": fainted Pokemon must be released; one catch per area
- "monotype": team restricted to one type; params={{"type": <type-name>}}
- "level_cap": party level capped below next gym leader; params={{"delta": int}}
- "no_items_in_battle": no healing/items during battle
- "custom": free-form; params={{"description": str}}

# Constitution

{text}

# Output

Return JSON only."""


def _category_list_for_prompt() -> str:
    return "\n".join(f'- "{cat.value}": {CATEGORY_DESCRIPTIONS[cat]}' for cat in GoalCategory)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Handles: plain JSON, JSON wrapped in ```json fences, JSON embedded in prose.
    """
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Find first balanced {...}
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]!r}")


# --- Heuristic (no-LLM) fallback parser -----------------------------------


_PLAYSTYLE_KEYWORDS: dict[str, list[str]] = {
    "nuzlocker": ["nuzlocke"],
    "speedrunner": ["speedrun", "speed run", "as fast"],
    "pokedex_hunter": ["pokedex", "pokédex", "catch them all", "complete the dex"],
    "explorer": ["explore", "exploration", "every area"],
    "story_runner": ["story", "beat the game", "finish the game"],
}


def _heuristic_parse(text: str) -> dict[str, Any]:
    """Regex/keyword fallback when no LLM is configured.

    Intentionally conservative — extracts only obvious signals.
    """
    lower = text.lower()
    result: dict[str, Any] = {
        "playstyle": None,
        "target_badges": None,
        "weights": {},
        "constraints": [],
        "notes": None,
    }

    for style, kws in _PLAYSTYLE_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            result["playstyle"] = style
            break

    badge_match = re.search(r"(\d+)\s*badge", lower)
    if badge_match:
        try:
            result["target_badges"] = max(0, min(8, int(badge_match.group(1))))
        except ValueError:
            pass

    if "nuzlocke" in lower:
        result["constraints"].append({"kind": "nuzlocke", "params": {}})

    monotype_match = re.search(r"mono(?:type)?\s+([a-z]+)", lower)
    if monotype_match:
        result["constraints"].append(
            {"kind": "monotype", "params": {"type": monotype_match.group(1)}}
        )

    # Playstyle-based default weights
    if result["playstyle"] == "nuzlocker":
        result["weights"] = {"progress": 0.7, "safety": 0.8, "mastery": 0.5}
    elif result["playstyle"] == "speedrunner":
        result["weights"] = {"progress": 1.0, "efficiency": 0.8}
    elif result["playstyle"] == "pokedex_hunter":
        result["weights"] = {"completeness": 1.0, "discovery": 0.6, "progress": 0.1}
    elif result["playstyle"] == "explorer":
        result["weights"] = {"discovery": 1.0, "completeness": 0.4, "progress": 0.2}
    elif result["playstyle"] == "story_runner":
        result["weights"] = {"progress": 1.0, "mastery": 0.3, "safety": 0.2}

    return result


# --- Parser entry point ---------------------------------------------------


@dataclass
class ConstitutionParseError(Exception):
    detail: str

    def __str__(self) -> str:
        return f"ConstitutionParseError: {self.detail}"


def parse_constitution(
    text: str,
    llm_client: Any = None,
    llm_model: str = "claude-haiku-4-5-20251001",
    fallback_on_error: bool = True,
) -> Constitution:
    """Parse a constitution string into structured fields.

    Args:
        text: designer's free-text constitution (≤ 1 paragraph recommended).
        llm_client: an Anthropic or OpenAI client instance.  If None, uses the
            heuristic fallback.
        llm_model: model name for the LLM call.
        fallback_on_error: if True, fall back to heuristic parsing when the
            LLM call or JSON extraction fails.  If False, raise.

    Returns:
        A Constitution populated with the best available extraction.
    """
    if not text or not text.strip():
        return Constitution(raw_text="")

    parsed: dict[str, Any] | None = None
    if llm_client is not None:
        prompt = _PARSER_PROMPT.format(category_list=_category_list_for_prompt(), text=text.strip())
        try:
            if hasattr(llm_client, "messages"):  # anthropic
                resp = llm_client.messages.create(
                    model=llm_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
            else:  # openai-like
                resp = llm_client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content
            parsed = _extract_json(raw)
        except Exception as e:
            if not fallback_on_error:
                raise ConstitutionParseError(detail=f"LLM call failed: {e}") from e
            logger.warning("Constitution LLM parse failed (%s); using heuristic fallback", e)
            parsed = None

    if parsed is None:
        parsed = _heuristic_parse(text)

    return _constitution_from_parsed(text, parsed)


def _constitution_from_parsed(text: str, parsed: dict[str, Any]) -> Constitution:
    """Build a Constitution from a raw text + parsed-fields dict."""
    playstyle = parsed.get("playstyle")
    if playstyle is not None and not isinstance(playstyle, str):
        playstyle = None

    tb = parsed.get("target_badges")
    target_badges: int | None = None
    if isinstance(tb, int) and 0 <= tb <= 8:
        target_badges = tb
    elif isinstance(tb, str):
        try:
            target_badges = max(0, min(8, int(tb)))
        except ValueError:
            target_badges = None

    notes = parsed.get("notes")
    if not isinstance(notes, str):
        notes = None

    return Constitution(
        raw_text=text,
        playstyle=playstyle,
        target_badges=target_badges,
        notes=notes,
    )


def build_config(
    text: str = "",
    llm_client: Any = None,
    llm_model: str = "claude-haiku-4-5-20251001",
) -> GoalRLConfig:
    """Build a GoalRLConfig from a constitution string.

    High-level entry point: parses the constitution (via LLM or heuristic),
    derives Layer-2 weights and Layer-3 constraints from the parsed structure,
    and assembles the full config.
    """
    if not text or not text.strip():
        return GoalRLConfig(
            constitution=Constitution(raw_text=""),
            layer_one=LayerOne(),
            layer_two=LayerTwo.equal_weights(),
            layer_three=LayerThree(),
        )

    # Run the parser once; capture both the structured dict and the constitution
    parsed: dict[str, Any]
    constitution: Constitution
    if llm_client is not None:
        prompt = _PARSER_PROMPT.format(category_list=_category_list_for_prompt(), text=text.strip())
        try:
            if hasattr(llm_client, "messages"):
                resp = llm_client.messages.create(
                    model=llm_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
            else:
                resp = llm_client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content
            parsed = _extract_json(raw)
        except Exception as e:
            logger.warning("Constitution LLM parse failed (%s); falling back", e)
            parsed = _heuristic_parse(text)
    else:
        parsed = _heuristic_parse(text)

    constitution = _constitution_from_parsed(text, parsed)
    layer_two = LayerTwo.from_dict(parsed.get("weights"))
    layer_three = LayerThree.from_list(parsed.get("constraints"))

    return GoalRLConfig(
        constitution=constitution,
        layer_one=LayerOne(),
        layer_two=layer_two,
        layer_three=layer_three,
    )


__all__ = [
    "GoalCategory",
    "ConstraintKind",
    "CATEGORY_DESCRIPTIONS",
    "LayerOne",
    "LayerTwo",
    "LayerThree",
    "Constraint",
    "Constitution",
    "GoalRLConfig",
    "ConstitutionParseError",
    "parse_constitution",
    "build_config",
]

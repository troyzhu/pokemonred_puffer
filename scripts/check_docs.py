#!/usr/bin/env python3
"""Doc-drift guards for docs/goal_rl/*.

Runs three checks:

1. **Primitives drift.** The primitives library listed in
   `docs/goal_rl/concepts/C03-*.md` must match the keys of
   `pokemonred_puffer.goal_rl.primitives.PRIMITIVES`.

2. **E1 dashboard drift.** The fields listed in
   `docs/goal_rl/concepts/C02-*.md` (the E1 dashboard enumeration) must match
   `pokemonred_puffer.goal_rl.evaluator_frozen.DASHBOARD_FIELDS`.

3. **Cross-reference + file-path integrity.** Every `[C##]`, `[S##]`,
   `[P##]`, `[G##]`, `[R##]`, `[B##]`, `[ADR ####]` reference in the
   goal_rl docs must resolve to a defined ID. Every
   ``pokemonred_puffer/...`` file-path reference must exist on disk.

Exit code 0 on success, 1 on any drift.  Designed to be lean enough for CI
(no deps beyond the stdlib + `omegaconf` + whatever the goal_rl module
itself needs).

Usage:

    python scripts/check_docs.py

    # Or from repo root with explicit failure-only output:
    python scripts/check_docs.py --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- Paths ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
GOAL_RL_DOCS = DOCS_ROOT / "goal_rl"
MAIN_DOC = DOCS_ROOT / "goal_rl.md"

# Ensure the project is importable whether or not `pip install -e .` was run.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _strip_code(text: str) -> str:
    """Remove fenced code blocks and inline code spans before link-matching.

    Doc authors frequently use inline `code` or fenced blocks to *show* what
    a link would look like (e.g. `[C0x](../concepts/C0x-slug.md)` as an
    example of the convention).  Those should not be treated as real links.
    """
    # Drop fenced blocks first (```...```).
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Then inline code spans (`...`).  Avoid chewing backticked shell prompts
    # at line starts by requiring the closing backtick on the same line.
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


# --- Check results --------------------------------------------------------


class Failure(Exception):
    """One drift-check failure."""


def _load_module_symbols():
    """Import the goal_rl module's primitives + dashboard for comparison.

    Kept as a function so the script can still parse docs if the import
    fails (in CI without the full env): in that case we skip the code-vs-doc
    checks and only run the link/path integrity pass.
    """
    try:
        from pokemonred_puffer.goal_rl.evaluator_frozen import DASHBOARD_FIELDS
        from pokemonred_puffer.goal_rl.primitives import PRIMITIVES

        return {
            "primitives": set(PRIMITIVES.keys()),
            "dashboard_fields": tuple(DASHBOARD_FIELDS),
            "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# --- Helpers --------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_goal_rl_docs():
    """Yield all markdown files under docs/goal_rl.md + docs/goal_rl/."""
    if MAIN_DOC.exists():
        yield MAIN_DOC
    if GOAL_RL_DOCS.exists():
        for p in sorted(GOAL_RL_DOCS.rglob("*.md")):
            yield p


# --- Check 1: primitives list matches code --------------------------------


# Matches:  - `fraction` (at the start of a line)
#           or ``fraction(field, max)` ...`` followed by → or just listed.
# The canonical primitive list in C03 is a fenced code block of the form:
#   fraction(field, max)           → min(x / max, 1.0)
PRIMITIVE_BLOCK_RE = re.compile(
    r"```\s*\n((?:[ \t]*[a-z_][a-z_0-9]*\([^\n]*\)[^\n]*\n?)+)\s*```",
    re.MULTILINE,
)
PRIMITIVE_NAME_RE = re.compile(r"^[ \t]*([a-z_][a-z_0-9]*)\(", re.MULTILINE)


def check_primitives(symbols: dict) -> list[str]:
    failures: list[str] = []
    if not symbols.get("available"):
        return [
            "skipping primitives check — could not import goal_rl.primitives "
            f"({symbols.get('error', 'unknown error')})"
        ]

    doc_path = _find_concept_doc("C03")
    if doc_path is None:
        failures.append("could not locate C03 concept doc")
        return failures

    text = _read(doc_path)
    match = PRIMITIVE_BLOCK_RE.search(text)
    if match is None:
        failures.append(
            f"{doc_path}: could not find primitives code block (looked for "
            f"`name(args) → …` fenced block)"
        )
        return failures

    doc_primitives = set(PRIMITIVE_NAME_RE.findall(match.group(1)))
    code_primitives = symbols["primitives"]

    missing_in_doc = code_primitives - doc_primitives
    extra_in_doc = doc_primitives - code_primitives
    if missing_in_doc:
        failures.append(f"{doc_path}: primitives in code but not doc: {sorted(missing_in_doc)}")
    if extra_in_doc:
        failures.append(f"{doc_path}: primitives in doc but not code: {sorted(extra_in_doc)}")
    return failures


# --- Check 2: E1 dashboard fields match code ------------------------------


# Matches the tuple literal of `DASHBOARD_FIELDS` names mentioned in the
# C02 doc — we look for any `` `field_name` `` pattern and compare.
DASHBOARD_INLINE_RE = re.compile(r"`([a-z_][a-z_0-9]*)`")


def check_dashboard_fields(symbols: dict) -> list[str]:
    """Check the E1 dashboard list in C02.

    Two modes depending on how the doc is written:

    - **Canonical-in-doc mode.**  If the doc lists the complete set of
      dashboard fields (every name from `DASHBOARD_FIELDS` appears as
      `` `name` `` in the E1 paragraph), the doc *is* the canonical list
      and we enforce strict equality.
    - **Deferred-to-code mode.**  If the doc contains the phrase
      ``DASHBOARD_FIELDS`` or ``canonical list lives in``, we consider
      the doc is deferring to the code and only check that every
      explicitly-named field in the doc's E1 paragraph is real.

    This gives the doc author the choice: list every field (full
    transparency), or point at the code as the source of truth.
    """
    failures: list[str] = []
    if not symbols.get("available"):
        return []

    doc_path = _find_concept_doc("C02")
    if doc_path is None:
        failures.append("could not locate C02 concept doc")
        return failures

    text = _read(doc_path)

    e1_section = re.search(
        r"^\s*-\s*\*\*E1[^\n]*\n(?:.+\n)+?(?:^\s*-\s*\*\*E2)",
        text,
        re.MULTILINE,
    )
    if e1_section is None:
        failures.append(f"{doc_path}: could not locate E1 bullet in C02 doc")
        return failures

    section_text = e1_section.group(0)
    doc_fields = set(DASHBOARD_INLINE_RE.findall(section_text))
    code_fields = set(symbols["dashboard_fields"])

    deferring = "DASHBOARD_FIELDS" in section_text or "canonical list" in section_text.lower()

    if deferring:
        # Only check that every explicitly-named field is real.
        extras = {f for f in doc_fields - code_fields if _looks_like_field_name(f)}
        if extras:
            failures.append(
                f"{doc_path}: E1 paragraph names fields that aren't in code "
                f"DASHBOARD_FIELDS: {sorted(extras)}"
            )
        return failures

    # Canonical-in-doc mode: strict equality.
    missing_in_doc = code_fields - doc_fields
    extras = {f for f in doc_fields - code_fields if _looks_like_field_name(f)}
    if missing_in_doc:
        failures.append(
            f"{doc_path}: dashboard fields in code but not doc's E1 bullet: "
            f"{sorted(missing_in_doc)}"
        )
    if extras:
        failures.append(f"{doc_path}: dashboard fields in doc but not code: {sorted(extras)}")
    return failures


def _looks_like_field_name(s: str) -> bool:
    """Heuristic: a GameStateSnapshot field looks like `snake_case_name`
    with at least one underscore, or a known short alias.  Used to filter
    prose words that happen to be in backticks (e.g. `E1`, `badges`)."""
    if "_" in s:
        return True
    return s in {"badges"}  # very short but unambiguous


def _find_concept_doc(cid: str) -> Path | None:
    """Locate a C##-*.md file under docs/goal_rl/concepts/."""
    concept_dir = GOAL_RL_DOCS / "concepts"
    if not concept_dir.exists():
        return None
    for p in concept_dir.glob(f"{cid}-*.md"):
        return p
    return None


# --- Check 3: cross-refs + file-path references ---------------------------


# Cross-reference patterns we recognise: bracketed IDs that must resolve.
# The "plain" form is `[C01]`, `[S03]`, etc.  The "linked" form is
# `[C01](path)` — we don't need to validate the path here (the link-target
# check below does that), just that the ID resolves in the taxonomy.
CROSSREF_RE = re.compile(r"\[(C\d{2}|S\d{2}|P\d{2}|G\d{2}|R\d{2}|B\d{2})\](?!\()")
LINKED_CROSSREF_RE = re.compile(r"\[(C\d{2}|S\d{2}|P\d{2}|G\d{2}|R\d{2}|B\d{2})\]\([^)]*\)")
ADR_REF_RE = re.compile(r"\[ADR\s*(\d{4})\]\([^)]+\)")

# File-path references, intentionally limited to the project namespace so
# we don't try to validate every markdown link.
PATH_REF_RE = re.compile(r"(?<![\w/])(pokemonred_puffer/[A-Za-z0-9_./\-]+?\.(?:py|md|yaml|yml))")

# ID-definition patterns per file type (where are IDs introduced):
DEFINITION_RES: dict[str, re.Pattern] = {
    # "### C01 — ..." or "## C01 — ..." in concept/research files.
    "heading": re.compile(r"^#{1,6}\s+([CSPGRB]\d{2})\b", re.MULTILINE),
    # "- **S01** ..." in the sources bibliography.
    "source_bullet": re.compile(r"^[-*]\s+\*\*(S\d{2})\*\*", re.MULTILINE),
}


def _collect_defined_ids() -> set[str]:
    """Scan all goal_rl docs for lines that DEFINE a C##/S##/P##/etc. ID."""
    ids: set[str] = set()
    for path in _iter_goal_rl_docs():
        text = _read(path)
        for pat in DEFINITION_RES.values():
            ids.update(pat.findall(text))
    return ids


def _collect_defined_adrs() -> set[str]:
    adr_dir = GOAL_RL_DOCS / "adr"
    if not adr_dir.exists():
        return set()
    numbers = set()
    for p in adr_dir.glob("*.md"):
        m = re.match(r"(\d{4})-", p.name)
        if m:
            numbers.add(m.group(1))
    return numbers


def check_cross_refs() -> list[str]:
    failures: list[str] = []
    defined = _collect_defined_ids()
    defined_adrs = _collect_defined_adrs()

    for path in _iter_goal_rl_docs():
        text = _strip_code(_read(path))

        # Bare references: [C01], [S03], etc.
        for m in CROSSREF_RE.finditer(text):
            ref = m.group(1)
            if ref not in defined:
                failures.append(
                    f"{_rel(path)}: undefined cross-ref [{ref}] "
                    f"(no heading defines it anywhere in docs/goal_rl/)"
                )

        # Linked references: [C01](...). We still check the ID exists; the
        # link target is validated separately by _check_md_links.
        for m in LINKED_CROSSREF_RE.finditer(text):
            ref = m.group(1)
            if ref not in defined:
                failures.append(f"{_rel(path)}: undefined linked cross-ref [{ref}](…)")

        # ADR references: [ADR 0001](...).
        for m in ADR_REF_RE.finditer(text):
            num = m.group(1)
            if num not in defined_adrs:
                failures.append(f"{_rel(path)}: ADR {num} referenced but no adr/{num}-*.md exists")

    return failures


def check_file_paths() -> list[str]:
    failures: list[str] = []
    for path in _iter_goal_rl_docs():
        text = _strip_code(_read(path))
        for m in PATH_REF_RE.finditer(text):
            ref = m.group(1)
            target = REPO_ROOT / ref
            if not target.exists():
                failures.append(f"{_rel(path)}: file-path reference `{ref}` does not exist")
    return failures


# --- Markdown internal link check -----------------------------------------


MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)")


def check_markdown_links() -> list[str]:
    """Check that every ``[text](relative/path.md)`` in the goal_rl docs
    resolves to a file that exists (headers/anchors not validated — too
    brittle across tools; file existence is the cheap and useful bit)."""
    failures: list[str] = []
    for path in _iter_goal_rl_docs():
        text = _strip_code(_read(path))
        for m in MD_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("/"):
                # Absolute path — not used in this tree; treat as doc bug.
                failures.append(
                    f"{_rel(path)}: absolute markdown link {target!r} (use a relative path)"
                )
                continue
            # Resolve relative to the doc file's directory.
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"{_rel(path)}: link target {target!r} does not exist "
                    f"(resolved to {_rel(resolved)})"
                )
    return failures


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --- CLI ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args(argv)

    symbols = _load_module_symbols()

    all_failures: list[tuple[str, list[str]]] = []
    for name, fn in [
        ("primitives", lambda: check_primitives(symbols)),
        ("dashboard fields", lambda: check_dashboard_fields(symbols)),
        ("cross-references", check_cross_refs),
        ("file-path references", check_file_paths),
        ("markdown link targets", check_markdown_links),
    ]:
        results = fn()
        all_failures.append((name, results))

    any_fatal = False
    for name, results in all_failures:
        # The import-unavailable case returns a "skipping" message; treat as info.
        skipping = any(r.startswith("skipping") for r in results)
        hard_failures = [r for r in results if not r.startswith("skipping")]

        if hard_failures:
            any_fatal = True
            print(f"[FAIL] {name}:", file=sys.stderr)
            for msg in hard_failures:
                print(f"  - {msg}", file=sys.stderr)
        elif skipping and not args.quiet:
            print(f"[SKIP] {name}: {results[0]}")
        elif not args.quiet:
            print(f"[OK]   {name}")

    if any_fatal:
        print(
            "\nDoc drift detected.  Update the docs or the code so they agree, "
            "then re-run `python scripts/check_docs.py`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

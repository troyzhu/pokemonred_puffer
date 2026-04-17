"""Goal-setting RL layer built on top of rubric_rl.

Implements a three-layer value schema (core values / session goals / constraints),
a frozen evaluation protocol (E1 dashboard + E2 frozen scalar + E3 session target),
a constrained LLM revision engine, and hybrid eval triggers (milestone + adaptive-K
safety net).

Design reference: docs/goal_rl.md — concept IDs (C01–C08) in module docstrings
cross-link to sections there.
"""

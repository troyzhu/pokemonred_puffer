# Concept pages

Each concept has a stable ID (`C##`) that is used throughout the code
(`docs/goal_rl.md C07` comments) and the other docs. IDs never change;
a retired concept gets `Status: superseded by Cxx` but keeps its file.

| # | Title | Code |
|---|-------|------|
| [C01](C01-three-layer-value-schema.md) | Three-layer value schema | `goal_rl/schema.py` |
| [C02](C02-frozen-evaluation-protocol.md) | Frozen evaluation protocol (E1/E2/E3) | `goal_rl/evaluator_frozen.py` |
| [C03](C03-constrained-llm-revision-engine.md) | Constrained-LLM revision engine | `goal_rl/revision_engine.py`, `goal_rl/primitives.py` |
| [C04](C04-revision-triggers-and-safety-nets.md) | Revision triggers & safety nets | `goal_rl/triggers.py` |
| [C05](C05-integration-with-rubric-rl.md) | Integration with `rubric_rl` | `goal_rl/goal_grpo.py` |
| [C06](C06-design-tensions.md) | Design tensions | — |
| [C07](C07-learned-milestone-detection.md) | Learned milestone detection | `goal_rl/primitives.py::detect_milestones` |
| [C08](C08-constitution-parsing.md) | Constitution parsing | `goal_rl/schema.py::parse_constitution` |

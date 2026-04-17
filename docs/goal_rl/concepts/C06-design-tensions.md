# C06 — Design Tensions

**Status:** open reminders — these don't resolve, they just get managed.

1. **Stability ↔ responsiveness.** Frequent revisions → non-stationary
   reward; GRPO with a drifting target is fragile. Mitigation:
   [C02 E2 watchdog](C02-frozen-evaluation-protocol.md), cooldown in
   [C04](C04-revision-triggers-and-safety-nets.md), bounded deltas in
   [C03](C03-constrained-llm-revision-engine.md).

2. **Expressiveness ↔ trainability.** Free-form LLM reward code is
   maximally expressive and maximally pathological. Mitigation: the
   primitives library in [C03](C03-constrained-llm-revision-engine.md)
   — enough expressiveness for rebalancing and threshold-tracking,
   not enough to write pathological reward shapes.

3. **Compute.** Each LLM call costs real money. Mitigation: trigger
   sensitivity tuned by compute ceiling; one LLM call per trigger
   fire, not per epoch; `max_llm_revisions` hard cap in config.

4. **Catastrophic forgetting.** A tightened criterion can erase
   learned behavior. Mitigation: bounded deltas in
   [C03](C03-constrained-llm-revision-engine.md) and the
   `remove_criterion` floor that disallows dropping criteria below a
   training-rubric floor. Voyager-style skill preservation remains a
   V2 concern.

5. **Human oversight bottleneck.** Fully-autonomous self-revision
   drifts from designer intent. Mitigation: the free-text constitution
   in [C08](C08-constitution-parsing.md) + audit trail (JSONL per run)
   + eval dashboard visible to the designer each epoch.

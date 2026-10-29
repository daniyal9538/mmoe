# Changelog

## 0.1.0 — 2026-10-29
- Initial repo. Expert-choice M-MoE ported from `Documents/daniyal/mmoemlp` (`src/models/mmoe`, configs
  `mmoe_expert_choice`, `mmoe_ec_cols`, `mmoe_ecs_K*`).
- New: `use_obs_flags`, `capacity`, float `r`, `ec_surrogate`, `ec_noise`, diagnostics `n_orphan`,
  `changed_vs_init`, `routing_margin`; `budget_type` default changed to `concrete`.
- Removed: feature-choice routing, partition/free membership, load-balance loss, dead-expert revival,
  Stage-A matched random control (recorded in README).
- MNIST tutorial notebook, CLI training script, unit tests.

# Branch README: `feature/env-feedback-proj-accel-reordered-exact`

This document tracks the cumulative changes on this branch relative to the earlier
`feature/env-feedback-proj-pg` line, with a focus on env-feedback gradient projection.

## 1) Projection Architecture Changes

- Added projection implementation selector:
  - `actor_rollout_ref.actor.env_feedback_grad_proj_impl`
  - values: `legacy_cpu_offload`, `reordered_exact`
- Default implementation is `reordered_exact`.
- Added projection target selector:
  - `actor_rollout_ref.actor.env_feedback_grad_proj_target`
  - values: `pg`, `pg_plus_other`
  - default: `pg`

### `reordered_exact` (existing on this branch)

- Keeps strict-exact projection math for `target=pg`.
- Reorders backward passes to reduce hot-path CPU offload pressure.
- Includes OOM-safe fallback and runtime disable behavior.

### `pg_plus_other` target (new on this branch)

- Optional mode under `reordered_exact`.
- Uses base gradient direction `g_base = g_pg + g_other` as projection target.
- Implementation performs:
  1. one no-env policy backward to materialize `g_base`
  2. one env backward with hook-based dot/norm accumulation
  3. in-place scale of current grads by `(1 + alpha)`
- In this mode, PG and other losses are **not split into separate backward phases**.

## 2) FSDP Robustness Fixes

- Fixed reordered path where `param.grad` can be unavailable in env hooks under FSDP.
- Added CPU fallback snapshot for reference grads during projection dot calculation.
- Added shard/full alignment helper for safe dot products:
  - handles flattened-full vs local-shard shape mismatch.

## 3) Runtime Safety

- Added/kept projection safety controls:
  - `env_feedback_grad_proj_min_free_gb`
  - `env_feedback_grad_proj_oom_fallback`
- On OOM in projection path:
  - cleanup grads and CUDA cache
  - skip affected microbatch
  - disable projection/entropy runtime paths as configured

## 4) Metrics Added on This Branch

- Existing projection metrics retained:
  - `actor/env_feedback_proj_alpha`
  - `actor/env_feedback_proj_alpha_abs`
  - `actor/env_feedback_proj_dot`
  - `actor/env_feedback_proj_pg_norm_sq`
  - `actor/env_feedback_proj_env_norm_sq`
  - `actor/env_feedback_proj_cosine_before`
  - `actor/env_feedback_proj_cosine_after`
  - `actor/env_feedback_proj_applied`
- Added diagnostics:
  - `actor/env_feedback_proj_pg_grad_present_param_ratio`
  - `actor/env_feedback_proj_pg_grad_present_elem_ratio`
- Added implementation/target tags:
  - `actor/env_feedback_proj_impl` (`0=legacy`, `1=reordered`)
  - `actor/env_feedback_proj_target` (`0=pg`, `1=pg_plus_other`)
- Added alpha sign observability:
  - `actor_step/env_feedback_proj_alpha_pos_ratio`
  - with support counters:
    - `actor_step/env_feedback_proj_alpha_pos_count`
    - `actor_step/env_feedback_proj_alpha_valid_count`

## 5) Training Interface Changes

- Updated `examples/train/AgentGym-RL/webshop_train.sh`:
  - new CLI/env passthrough:
    - `--env_feedback_grad_proj_target`
    - `ENV_FEEDBACK_GRAD_PROJ_TARGET`
- Updated `AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`:
  - added default `env_feedback_grad_proj_target: pg`

## 6) Notes for Repro

- Env-feedback projection is active only when all are true:
  - `project_env_feedback_grad_to_pg=true`
  - `env_feedback_loss_coef != 0`
  - env-feedback token count > 0
- For `target=pg_plus_other`, use:
  - `env_feedback_grad_proj_impl=reordered_exact`
  - `env_feedback_grad_proj_target=pg_plus_other`

## 7) Derived Branch Update: `feature/env-feedback-proj-accel-reordered-exact-alpha-clamp`

- This derived branch adds one behavior change on top of the content above:
  - clamp projection scalar to non-negative before applying it to gradients:
    - `raw_alpha = <g_env, g_target> / ||g_target||^2`
    - `alpha = max(raw_alpha, 0)`
    - apply `g <- (1 + alpha) * g_target`
- Effect:
  - env feedback can increase target-direction update magnitude
  - env feedback can no longer reverse the target direction when raw correlation is negative
- Scope of clamp:
  - applied in active projection path (`_project_env_grad_to_pg`)
  - applied in legacy-compat stat computation (`_compute_env_proj_stats_from_pg_cpu`) for consistency

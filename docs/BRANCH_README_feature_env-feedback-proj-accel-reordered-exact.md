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
- Added projection distribution diagnostics (step-level):
  - `actor_step/env_feedback_proj_alpha_max`
  - `actor_step/env_feedback_proj_alpha_min`
  - `actor_step/env_feedback_proj_alpha_std`
  - `actor_step/env_feedback_proj_cosine_before_max`
  - `actor_step/env_feedback_proj_cosine_before_min`
  - `actor_step/env_feedback_proj_cosine_before_std`
  - `actor_step/env_feedback_proj_cosine_after_max`
  - `actor_step/env_feedback_proj_cosine_after_min`
  - `actor_step/env_feedback_proj_cosine_after_std`
  - `actor_step/env_feedback_proj_cosine_before_pos_ratio`
  - `actor_step/env_feedback_proj_cosine_before_pos_mean`
  - `actor_step/env_feedback_proj_cosine_before_neg_mean`

## 5) Training Interface Changes

- Updated `examples/train/AgentGym-RL/webshop_train.sh`:
  - new CLI/env passthrough:
    - `--env_feedback_grad_proj_target`
    - `ENV_FEEDBACK_GRAD_PROJ_TARGET`
- Updated `AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`:
  - added default `env_feedback_grad_proj_target: pg`
  - added defaults:
    - `pg_plus_other_loss_coef: 1.0`
    - `pg_loss_coef: 1.0`

## 6) PPO Loss Scaling Knobs

- Added two actor-side scalar controls:
  - `actor_rollout_ref.actor.pg_plus_other_loss_coef` (default `1.0`)
  - `actor_rollout_ref.actor.pg_loss_coef` (default `1.0`)
- Effective no-env policy term now follows:
  - `pg_plus_other_loss_coef * (pg_loss_coef * pg_loss + other_loss)`
  - where `other_loss` includes entropy and KL terms.
- Env-feedback weighted loss remains additive and is not multiplied by these two knobs.
- Both knobs are plumbed to train script:
  - `--pg_plus_other_loss_coef` / `PG_PLUS_OTHER_LOSS_COEF`
  - `--pg_loss_coef` / `PG_LOSS_COEF`

## 7) Notes for Repro

- Env-feedback projection is active only when all are true:
  - `project_env_feedback_grad_to_pg=true`
  - `env_feedback_loss_coef != 0`
  - env-feedback token count > 0
- For `target=pg_plus_other`, use:
  - `env_feedback_grad_proj_impl=reordered_exact`
  - `env_feedback_grad_proj_target=pg_plus_other`

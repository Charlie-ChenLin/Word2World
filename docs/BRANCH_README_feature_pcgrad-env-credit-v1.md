# Branch README: `feature/pcgrad-env-credit-v1`

本文档完整记录本分支相对 `pgplusother_alphaclamp` 基线的 **PCGrad 化 env feedback 梯度交互**改动。  
重点回答四件事：

1. 这次到底改了哪些代码与行为；
2. `pcgrad` 与 `alpha_scale` 在实现上有什么本质区别；
3. 如何配置、运行、观测和排障；
4. 当前版本有哪些边界与限制。

---

## 1. 改动目标与设计原则

### 1.1 目标

在 actor 更新阶段，不再只用一个标量 `alpha` 去缩放目标梯度，而是引入显式梯度交互：

- 冲突时（`dot(g_pg, g_env) < 0`）去冲突；
- 对齐时（`dot(g_pg, g_env) > 0`）做有界注入；
- 保留 `other`（entropy + KL）为独立加性项，不参与投影锚点。

### 1.2 设计原则

- **锚点固定为 PG**：首版 `pcgrad` 仅支持 `pg` 锚点；
- **有界注入**：env 正向注入受双重上界约束，避免 env 主导更新；
- **向后兼容**：默认仍走旧逻辑 `alpha_scale`；
- **OOM 可恢复**：保留低显存跳过和 OOM fallback 机制。

---

## 2. 代码改动总览

### 2.1 核心实现

- `AgentGym-RL/verl/workers/agent_actor/dp_actor.py`
  - 新增 `env_feedback_grad_proj_algo` 分支：`alpha_scale` / `pcgrad`
  - 新增 PCGrad 参数读取与校验
  - 新增三路梯度采集 + 合成逻辑（`g_pg / g_other / g_env`）
  - 新增 PCGrad 指标写入（micro-batch 与 actor_step 聚合）

### 2.2 配置

- `AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`
  - 新增：
    - `actor_rollout_ref.actor.env_feedback_grad_proj_algo: alpha_scale`
    - `actor_rollout_ref.actor.env_feedback_pcgrad_lambda_max: 0.2`
    - `actor_rollout_ref.actor.env_feedback_pcgrad_lambda_norm_ratio: 0.3`
    - `actor_rollout_ref.actor.env_feedback_pcgrad_eps: 1e-12`
    - `actor_rollout_ref.actor.env_feedback_pcgrad_log_extra_metrics: True`

### 2.3 训练入口脚本

- `examples/train/AgentGym-RL/webshop_train.sh`
  - 新增 CLI 透传参数：
    - `--env_feedback_grad_proj_algo`
    - `--env_feedback_pcgrad_lambda_max`
    - `--env_feedback_pcgrad_lambda_norm_ratio`
    - `--env_feedback_pcgrad_eps`
    - `--env_feedback_pcgrad_log_extra_metrics`
  - 额外增加：
    - 强制 `PYTHONPATH=${REPO_ROOT}/AgentGym-RL` 优先，防止 editable 安装导到旧仓库
    - 启动前打印 `verl` 导入来源用于自检

---

## 3. 算法说明

> 下文的所有梯度均是 micro-batch 粒度、全参数展开空间上的向量概念。

### 3.1 损失分解

代码中显式拆分三路梯度：

- `g_pg`: 仅来自 `pg_loss_scaled`
- `g_other`: 仅来自 `(- entropy * entropy_coeff + kl * kl_coef) * scale_factor`
- `g_env`: 仅来自 `env_feedback_loss_weighted_scaled`

其中：

- `env_feedback_loss = -masked_mean(log_prob, env_feedback_mask)`
- `env_feedback_loss_weighted = env_feedback_loss * env_feedback_loss_coef`

### 3.2 旧算法 `alpha_scale`（保留）

核心是标量缩放（在 `alpha_clamp` 分支上为非负）：

- `alpha = clamp_min(<g_target, g_env> / ||g_target||^2, 0)`
- 目标梯度按 `(1 + alpha)` 缩放

其中 `g_target` 可是：

- `pg`（默认）
- `pg_plus_other`（仅 `alpha_scale` 下支持）

### 3.3 新算法 `pcgrad`（本分支新增）

当前实现固定锚点 `pg`，并在代码中强制：

- 若配置 `impl != reordered_exact`，回退到 `reordered_exact`
- 若配置 `target != pg`，回退到 `pg`

定义：

- `d = <g_pg, g_env>`
- `||g_pg||^2 = n_pg^2`
- `||g_env||^2 = n_env^2`

#### (1) 冲突消解（PG-anchored）

- `neg = max(0, -d)`
- `conflict_scale = neg / (n_env^2 + eps)`
- `g_pg_pc = g_pg + conflict_scale * g_env`

解释：当 `d < 0` 时，加入 `g_env` 的一部分以抵消负向分量，使 `g_pg_pc` 与 `g_env` 至少不再负相关。

#### (2) 正向有界融合

- `pos = max(0, d)`
- `env_pos_scale = pos / (n_pg^2 + eps)`
- `g_env^+ = env_pos_scale * g_pg`  （env 在 PG 方向的正分量）
- `lambda_from_norm = lambda_norm_ratio * ||g_pg|| / (||g_env^+|| + eps)`
- `lambda = min(lambda_max, lambda_from_norm)`
- 若 `pos <= 0` 或范数过小则 `lambda = 0`

最终注入项：

- `lambda * g_env^+`

#### (3) 最终合成

实现上等价于：

- `g_final = g_pg_pc + g_other + lambda * g_env^+`

代码里用 scale 形式回填：

- `pg_scale = 1 + lambda * env_pos_scale`
- 先加 `pg_scale * g_pg`
- 再加 `g_other`
- 若冲突存在，再加 `conflict_scale * g_env`

---

## 4. 实现细节（与数学对应）

### 4.1 梯度采集机制

采用 hook + CPU snapshot：

- `_capture_loss_grads_to_cpu_without_accum(...)`
  - 反传某个 loss
  - 抓取每个参数 grad 到 CPU dict
  - 同时将当前 grad 清零，避免污染
- `_compute_grad_dict_pair_stats(...)`
  - 计算两组 grad dict 的 dot / norm / 覆盖率
- `_accumulate_offloaded_grads_from_cpu(...)`
  - 将 CPU grad 按给定 scale 回填到 `param.grad`

### 4.2 FSDP 兼容

由于 FSDP 下可能出现 full tensor 与 local shard 形状不一致，代码通过 `_align_env_pg_for_dot(...)` 做对齐后再计算 dot/norm。

### 4.3 低内存与 OOM 行为

- `env_feedback_grad_proj_min_free_gb`：低于阈值直接跳过投影
- `env_feedback_grad_proj_oom_fallback=True`：projection 相关 OOM 后跳过该 micro-batch 并清理缓存，不让整个训练崩掉

---

## 5. 配置项说明

### 5.1 总开关（沿用）

- `actor_rollout_ref.actor.project_env_feedback_grad_to_pg`
  - `False`：完全不启用投影/PCGrad
  - `True`：按 `env_feedback_grad_proj_algo` 执行

### 5.2 算法选择

- `actor_rollout_ref.actor.env_feedback_grad_proj_algo`
  - `alpha_scale`（默认，旧行为）
  - `pcgrad`（新行为）

### 5.3 PCGrad 参数

- `actor_rollout_ref.actor.env_feedback_pcgrad_lambda_max`（默认 `0.2`）
- `actor_rollout_ref.actor.env_feedback_pcgrad_lambda_norm_ratio`（默认 `0.3`）
- `actor_rollout_ref.actor.env_feedback_pcgrad_eps`（默认 `1e-12`）
- `actor_rollout_ref.actor.env_feedback_pcgrad_log_extra_metrics`（默认 `True`）

### 5.4 兼容/限制参数

- `actor_rollout_ref.actor.env_feedback_grad_proj_impl`
  - `pcgrad` 下仅支持 `reordered_exact`
- `actor_rollout_ref.actor.env_feedback_grad_proj_target`
  - `pcgrad` 下仅支持 `pg`
  - 若设置成 `pg_plus_other` 会打印 warning 并回退到 `pg`

---

## 6. 指标体系

### 6.1 旧指标（保留）

- `actor/env_feedback_proj_alpha`
- `actor/env_feedback_proj_dot`
- `actor/env_feedback_proj_cosine_before`
- `actor/env_feedback_proj_cosine_after`
- 以及对应 `actor_step/*` 聚合指标

### 6.2 PCGrad 新增核心指标

- `actor/env_feedback_pcgrad_dot_pg_env_before`
- `actor/env_feedback_pcgrad_cos_pg_env_before`
- `actor/env_feedback_pcgrad_dot_pg_env_after_conflict_fix`
- `actor/env_feedback_pcgrad_lambda`
- `actor/env_feedback_pcgrad_norm_pg`
- `actor/env_feedback_pcgrad_norm_other`
- `actor/env_feedback_pcgrad_norm_env`
- `actor/env_feedback_pcgrad_norm_env_pos`
- `actor/env_feedback_pcgrad_norm_final`
- `actor/env_feedback_pcgrad_final_vs_pg_cosine`
- `actor/env_feedback_pcgrad_conflict_applied`
- `actor/env_feedback_pcgrad_env_inject_ratio`

并同步映射到 `actor_step/*`。

### 6.3 分布统计

会自动计算部分指标的 `max/min/std`，包括：

- `env_feedback_proj_alpha`
- `env_feedback_proj_cosine_before`
- `env_feedback_proj_cosine_after`
- `env_feedback_pcgrad_lambda`
- `env_feedback_pcgrad_final_vs_pg_cosine`
- `env_feedback_pcgrad_env_inject_ratio`

此外保留：

- `actor_step/env_feedback_proj_cosine_before_pos_ratio`
- `actor_step/env_feedback_proj_cosine_before_pos_mean`
- `actor_step/env_feedback_proj_cosine_before_neg_mean`

---

## 7. 运行方式

### 7.1 直接脚本（推荐先小步验证）

```bash
bash examples/train/AgentGym-RL/webshop_train.sh \
  --env_feedback_grad_proj_to_pg true \
  --env_feedback_grad_proj_algo pcgrad \
  --env_feedback_grad_proj_impl reordered_exact \
  --env_feedback_grad_proj_target pg \
  --env_feedback_pcgrad_lambda_max 0.2 \
  --env_feedback_pcgrad_lambda_norm_ratio 0.3 \
  --env_feedback_pcgrad_eps 1e-12 \
  --env_feedback_pcgrad_log_extra_metrics true \
  --env_feedback_loss_coef 1
```

### 7.2 Slurm 示例

本分支可直接使用：

- `slurm/webshop-verl-agent-low-kl-small-minibatch-a800-apptainer-envcoef1-nosched-gn10-pcgrad-pg.slurm`

该脚本已透传全部 PCGrad 参数，并依赖 `webshop_train.sh` 完成最终 Hydra 覆盖。

---

## 8. 启动自检与排障

### 8.1 必看自检日志

`train.log` 开头应看到：

- `PYTHONPATH=.../Word2World_run_proj_pgplusother_alphaclamp/AgentGym-RL`
- `verl origin=.../Word2World_run_proj_pgplusother_alphaclamp/AgentGym-RL/verl/__init__.py`

如果这里指向旧仓库，后续配置和代码都会跑偏。

### 8.2 配置是否生效

在 worker 初始化日志中确认：

- `Actor project_env_feedback_grad_to_pg=True`
- `env_feedback_grad_proj_algo=pcgrad`
- `env_feedback_grad_proj_target=pg`

### 8.3 常见问题

1. `Key 'env_feedback_grad_proj_algo' is not in struct`
   - 原因：导入了旧仓库 `verl`
   - 处理：确认 `PYTHONPATH` 优先级和 `verl origin`

2. `pcgrad` 没有生效（指标全零）
   - 检查：
     - `project_env_feedback_grad_to_pg=true`
     - `env_feedback_loss_coef != 0`
     - 当前 batch `env_feedback_mask` 非空

3. 低显存下投影被跳过
   - 看 `actor/env_feedback_proj_skipped_low_mem`
   - 调小 `env_feedback_grad_proj_min_free_gb`

---

## 9. 与旧行为的兼容关系

### 9.1 默认不变

默认 `env_feedback_grad_proj_algo=alpha_scale`，因此未显式切换时行为与旧分支一致。

### 9.2 回退策略

- 配置非法值时自动回退并打印 warning：
  - `algo` 非法 -> `alpha_scale`
  - `pcgrad + impl!=reordered_exact` -> `reordered_exact`
  - `pcgrad + target!=pg` -> `pg`

---

## 10. 已知限制（当前版本）

1. `pcgrad` 首版仅支持 `target=pg`，不支持 `target=pg_plus_other`。
2. `other` 梯度不参与冲突投影，仅作为加性项保留。
3. `env_feedback_pcgrad_log_extra_metrics` 当前仅影响部分额外统计（非算法主路径）。
4. 尚未提供独立单元测试脚本，当前依赖短跑训练日志验证。

---

## 11. 最小验收建议

跑一个 `5-10 step` 短实验，至少确认：

1. 无 Hydra 配置错误、无 shape mismatch、无 OOM crash；
2. `actor_step/env_feedback_pcgrad_lambda` 有值且不超上界；
3. `actor_step/env_feedback_pcgrad_conflict_applied` 在冲突批次可非零；
4. `actor_step/env_feedback_pcgrad_final_vs_pg_cosine` 可稳定记录。

---

## 12. 关键文件索引

- `AgentGym-RL/verl/workers/agent_actor/dp_actor.py`
- `AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`
- `examples/train/AgentGym-RL/webshop_train.sh`
- `slurm/webshop-verl-agent-low-kl-small-minibatch-a800-apptainer-envcoef1-nosched-gn10-pcgrad-pg.slurm`


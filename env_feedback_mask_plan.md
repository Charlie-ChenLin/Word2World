# `env_feedback_mask` 生成与接入计划（与 `response_mask` 互补）

> 目标：在当前 rollout -> optimizer 的 trajectory-level 数据里，新增一条与 `responses` 对齐的 `env_feedback_mask`，用于标记 **环境反馈（obs）内容 token**。

## 1. 需求定义（将你的约束逐条落地）

我们希望新增的 `env_feedback_mask` 满足：

1) **初始 prompt 全 0**：与当前 `response_mask` 一样，prompt 侧不参与（mask=0）。

2) **chat template / 特殊格式 token 全 0**：例如 Qwen chat template 的 `\n<|im_start|>user\n`、`<|im_end|>` 等格式 token，不管属于 user 还是 assistant，都 mask=0。

3) **环境反馈追加时置 1**：在 rollout 过程中，每次把环境 observation 作为 user 消息追加时，
   - *只*对 observation 的 **content token** 置 1
   - 对 user 的 prefix/suffix token 仍置 0

4) **rollout 结束后同样切 response-side mask**：像 `response_mask` 一样，最终导出的 `env_feedback_mask` 只覆盖 `responses` 对应的 response 部分（shape `[bs, response_length]`），并且会 pad 到固定长度。

> 对齐约定：`env_feedback_mask` 必须与 `responses` 一一对齐（同样的 slice/pad 规则），否则后续在 actor/critic/aux-loss 里会错位。

## 2. 先确认现状：`response_mask` 的完整生成链路（作为参照）

当前 `response_mask` 的生成流程是：

- **(A) rollout 过程中维护全序列 `loss_mask`**
  - `RolloutHandler.add_user_message()`：user 段追加，`loss_mask` 全 0
  - `RolloutHandler.add_assistant_message()`：assistant 段追加，`loss_mask` 主要对内容 token 置 1
  - 文件：`AgentGym-RL/verl/workers/rollout/schemas.py`

- **(B) rollout 结束时切 response-side mask**
  - `RolloutHandler.truncate_output_ids()`：
    - `response_ids = input_ids[len(prompt_ids):]`
    - `response_loss_mask = loss_mask[len(prompt_loss_mask):]`
  - 文件：`AgentGym-RL/verl/workers/rollout/schemas.py`

- **(C) pad 并导出到 DataProto**
  - `vllm_rollout.py` 收集每条轨迹的 `response_loss_mask`，pad 到 `config.response_length`
  - `response_mask = response_loss_mask`
  - 写入 `DataProto.batch['response_mask']`
  - 文件：`AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`

## 3. 设计：并行维护一条 `env_mask`（全序列），再切出 `response_env_mask`

实现思路：**不要去“反转 response_mask”**，而是新增一条并行 mask 流：

- `loss_mask`：用于 RL/GRPO 的 response（主要 assistant 段）
- `env_mask`：用于 env feedback（user 段的 obs content）

这样可以同时做到：
- prompt=0
- template token=0
- assistant=0
- env obs content=1

## 4. 逐步改动清单（按文件/函数定位）

### Step 1：扩展 `RolloutHandler` 数据结构（`schemas.py`）

修改 `AgentGym-RL/verl/workers/rollout/schemas.py`：

- 在 `RolloutHandler.__init__` 新增字段（与现有 `loss_mask/prompt_loss_mask/response_loss_mask` 对称）：
  - `env_mask: List[int]`（全序列）
  - `prompt_env_mask: List[int]`（prompt 侧全 0）
  - `response_env_mask: List[int]`（response-side slice）

- 在长度一致性断言里把 `env_mask` 也纳入，保证 append 时不会漏。

### Step 2：初始化 prompt 的 env_mask 全 0（`vllm_rollout.py`）

修改 `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` 里构造 `RolloutHandler(...)` 的位置：

- 新增初始化：
  - `env_mask=[0] * len(input_ids)`
  - `prompt_env_mask=[0] * len(input_ids)`
  - `response_env_mask=[]`

### Step 3：`add_user_message()`：只对 obs content token 置 1（`schemas.py`）

修改 `RolloutHandler.add_user_message()`：

- 保持现有 `_loss_mask` 逻辑不变（仍为全 0）。
- 新增 `_env_mask` 逻辑（与 append_token_ids 严格同长度）：
  - 如果追加时需要补 `user_prefix_msg`：prefix 部分 `0`，content 部分 `1`
  - 如果不需要补 prefix：content 全 `1`
  - 最后追加 `user_suffix_msg`：suffix 全 `0`
- `self.env_mask += _env_mask`

> 这一步确保你要的“特殊 token 全 mask 掉（0）”，同时 env 反馈内容为 1。

### Step 4：`add_assistant_message()`：env_mask 全 0（`schemas.py`）

修改 `RolloutHandler.add_assistant_message()`：

- 保持现有 `_loss_mask` 逻辑不变。
- 新增 `_env_mask = [0] * len(append_token_ids)`（包括 prefix/suffix/内容）
- `self.env_mask += _env_mask`

### Step 5：rollout 结束切片 `response_env_mask`（`schemas.py`）

修改 `RolloutHandler.truncate_output_ids()`：

- 先像其他字段一样截断：
  - `self.env_mask = self.env_mask[: self.max_model_len]`

- 再切 response-side env mask（必须与 `responses` 对齐）：
  - `self.response_env_mask = self.env_mask[len(self.prompt_env_mask):][: self.max_response_len]`

### Step 6：pad 并写入 `DataProto.batch['env_feedback_mask']`（`vllm_rollout.py`）

修改 `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`：

- 在收集 `response_loss_mask` 的同时，收集 `rollout_handler.response_env_mask`。
- `pad_sequence(..., padding_value=0)`
- 不足 `config.response_length` 时 pad 到固定长度（同 response_mask）。
- 在最终 `TensorDict` 里新增：
  - `env_feedback_mask: env_feedback_mask`

## 5. 最小验证（建议加断言/日志，避免错位）

在 `vllm_rollout.py` 构造 batch 后建议做：

- 形状一致：
  - `env_feedback_mask.shape == response_mask.shape == responses.shape`

- 互斥性（允许都为 0 的位置：模板 token / padding）：
  - `(env_feedback_mask * response_mask).sum() == 0`

- padding 位为 0：
  - `response_attention_mask == 0` 的位置，两种 mask 都应为 0

## 6. 结果（你最终会得到什么）

rollout 输出的 `DataProto.batch` 里将同时包含：

- `responses`：包含 assistant(action) + user(obs) 的 token 流（response-side）
- `response_mask`：只标记需要用于 RL/GRPO 梯度的 token（通常是 assistant 段）
- `env_feedback_mask`：只标记 env obs 的 **content token**（user 段），并且 prompt=0、模板=0、padding=0

后续你就可以在 actor/critic 或额外 loss 中用 `env_feedback_mask` 做 mask（例如只对 env token 做 WM-style 的辅助目标）。

## 7. 后续：把 `env_feedback_mask` 用到 loss（训练侧需要额外改动）

> 这部分不是“生成 mask”必须的，但你后面要在算 loss 时用它，所以需要提前规划训练侧的接入点。

### 7.1 让 mask 进入 actor 的更新逻辑

修改 `AgentGym-RL/verl/workers/agent_actor/dp_actor.py`：

- 在 `update_policy()` 的 `select_keys` 里加入 `env_feedback_mask`
- 在 micro-batch 里取出：`env_mask = data['env_feedback_mask']`

### 7.2 定义 env-feedback 的辅助目标（两种常见做法）

**A) WM-SFT / PPL 风格（直接做 NLL）**

- actor forward 已经会得到 `log_prob`（shape `[bs, response_length]`，每个 response token 的 log p）
- 用 `env_mask` 做 masked mean：
  - `env_nll = -masked_mean(log_prob, env_mask)`
- 合并：`total_loss = policy_loss + env_aux_coef * env_nll`

**B) PPO/GRPO-style 的 “SFT” 目标（对 old policy 做 trust region）**

- 利用现成的 `old_log_probs`（同样是 `[bs, response_length]`，当前实现里 env token 也包含）
- 构造常数优势（例如全 1），并用 `env_mask` 作为 `eos_mask`：
  - `env_pg_loss, _, env_ppo_kl = compute_policy_loss(old_log_prob, log_prob, advantages=ones, eos_mask=env_mask, cliprange=env_clip_ratio)`
- 合并：`total_loss = policy_loss + env_aux_coef * env_pg_loss`

**（可选）再加 reference policy 的约束（对 ref 做 trust region）**

- 当前 pipeline 里 `ref_log_prob` 也是对所有 response token 计算的（env token 也有），可直接复用。
- 对 env token 计算 KL 并加权：
  - `env_kl = masked_mean(kl_penalty(log_prob, ref_log_prob, ...), env_mask)`
  - `total_loss += env_kl_coef * env_kl`

### 7.3 训练侧注意事项（避免“融入训练”时踩坑）

- `env_feedback_mask` 与 `response_mask` **不需要严格互补**：模板 token / padding token 通常两者都为 0 是正常的；但建议保证两者 **不重叠**（内容 token 层面）。
- 建议先把 env 辅助项做成 **纯 actor 的 auxiliary loss**，不要改 `apply_kl_penalty()` / `compute_advantage()`（它们目前完全基于 `response_mask`），避免把 env token 意外当作 RL 奖励序列的一部分。
- env observation 往往很长，aux loss 可能压过 RL：
  - 用 `masked_mean`（按 token 数归一化）
  - 控制 `env_aux_coef`（从很小开始）
  - 必要时对 obs 做截断或只挑选关键信息

### 7.4 建议新增/暴露的超参

- `use_env_aux`：是否启用 env 辅助项
- `env_aux_coef`：aux loss 系数
- `env_clip_ratio`：PPO-style aux 的 clip
- `env_kl_coef`：对 ref 的 KL 系数（可选）

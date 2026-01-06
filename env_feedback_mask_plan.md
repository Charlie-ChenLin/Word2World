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

## 8. 测试脚本设计：可视化单条轨迹的 token `log_prob` + `env_feedback_mask` / `response_mask`

> 目标：在完成 `env_feedback_mask` 相关改动后，提供一个“可复现 + 可视化”的 debug 脚本：
> - 跑出 **一条真实 rollout 轨迹**（包含多轮 env feedback + assistant action）
> - 计算这条轨迹在 **actor policy 下**的逐 token `log_prob`（与优化器使用的 token-level `log_prob` 对齐）
> - 同时拿到对应的 `env_feedback_mask` 与 `response_mask`
> - 画一张图：y=每个 token 的 `log_prob`，x=每个 token decode 后的字符串；用不同颜色标注 `env_feedback_mask==1` 和 `response_mask==1`

### 8.1 输入/输出约定

建议脚本放在：`scripts/debug/plot_env_feedback_mask_logprob.py`（debug 工具，不参与训练逻辑）。

**输入参数（建议）**

- `--mode`：`pipeline|transformers`（默认 `pipeline`；`pipeline` 会尽量复用训练的 Ray + vLLM + log_prob 计算）
- `--model_path`：本地 HF 模型目录（训练/评测用的 actor）
- `--env_server`：如 `http://127.0.0.1:36001`
- `--item_id`：从 `data/train/alfworld_train.json` 里选一个 `item_id`（`<task_type>_<task_id>` 格式），脚本内部用 AgentGym mappings 转成 env 所需的 int id
- `--max_rounds`：轨迹最大 turn（建议默认 5~10，避免图太长）
- `--max_new_tokens_per_turn`：单 turn 最大生成（对齐 `actor_rollout_ref.rollout.max_tokens`，例如 200）
- `--max_prompt_length` / `--max_response_length`：对齐训练配置（用于最终 truncate，与训练一致）
- `--plot_max_tokens`：最多画多少个 response token（例如默认 256；否则 x 轴不可读）
- `--out_png`：输出图路径
- （可选）`--dump_jsonl/--dump_pt`：把 token、log_prob、mask dump 下来便于复查

> 备注：若 `--mode pipeline`，更推荐脚本本身做成 hydra entrypoint（类似 `AgentGym-RL/verl/agent_trainer/main_generation.py`），直接复用训练的 config/override；此时 `--model_path/--env_server/...` 可由 hydra 配置接管，只在你做 `--mode transformers` 时需要显式传参。

**输出**

- `out_png`：一张图（token-level log_prob + mask 颜色）
- （可选）`dump`：保存用于画图的中间数据（含 token_id、token_str、log_prob、mask）

### 8.2 轨迹采样（如何得到“一条 rollout 轨迹”）

为了尽可能 **复用现有训练/推理 pipeline**，建议脚本提供两种模式（通过 `--mode {pipeline,transformers}` 切换）：

#### 8.2.1 `--mode pipeline`（推荐）：复用训练的 Ray + `ActorRolloutRefWorker` + vLLM rollout

目标：复用训练时的 **同一套 token 拼接 / mask 逻辑 / log_prob 计算**，避免自己手写一套导致对齐偏差。

实现要点（参考 `AgentGym-RL/verl/agent_trainer/main_generation.py` 的写法）：

1) 用与训练一致的 hydra 配置初始化 worker group（建议直接复用 `examples/train/AgentGym-RL/alfworld_train.sh` 的参数），并把 batch 相关参数缩到最小：
   - `trainer.n_gpus_per_node=1`（debug 用一张卡即可）
   - batch size 相关配置设为 1（不同 entrypoint 可能是 `data.train_batch_size=1` 或 `data.batch_size=1`）
   - `actor_rollout_ref.rollout.n=1`
   - `actor_rollout_ref.rollout.max_tokens=<max_new_tokens_per_turn>`
2) 构造一条样本的 `DataProto`（与训练一致）：
   - `data.non_tensor_batch["item_id"] = np.array([<item_id>], dtype=object)`（ALFWorld 用 `<task_type>_<task_id>`）
   - `data.non_tensor_batch["raw_prompt"] = np.array([<messages>], dtype=object)`（用 `init_env_client(config.agentgym).conversation_start` 生成两条固定开场）
   - `data.meta_info["max_rounds"] = <max_rounds>`
   - （可选）若 `dp_size>1` 且 `batch_size % dp_size != 0`，按 `main_generation.py` 的方式补齐 dummy data，避免分布式拆分时报错
3) rollout：
   - `output = wg.generate_sequences(data)`  
   - 期待 `output.batch` 里有：`responses`、`response_mask`、`env_feedback_mask`（完成接入后新增）、`prompts`、`attention_mask`、`input_ids`
4) 计算“传给优化器”的 token-level 概率（与训练一致）：
   - `lp = wg.compute_log_prob(output).batch["old_log_probs"]`
   - 这就是 PPO/GRPO 里用于 policy loss 的 `old_log_probs`（与训练对齐，无需在脚本里手写 gather）
5) 只画有效的 response token（排除 padding）：
   - `prompt_len = output.batch["prompts"].shape[-1]`
   - `valid_len = output.batch["attention_mask"][:, prompt_len:].sum(dim=-1).item()`
   - 取 `responses[0, :valid_len]`、`old_log_probs[0, :valid_len]`、`response_mask[0, :valid_len]`、`env_feedback_mask[0, :valid_len]`

#### 8.2.2 `--mode transformers`（fallback）：纯 `transformers` 最小 rollout

当 Ray/vLLM 初始化太重，或你希望在本地快速验证时，可用纯 `transformers` 做一个最小 rollout（逻辑与训练 rollout 一致即可）：

1) 用 `item_id` 找到 env 的 int index：
   - 读取 `AgentGym/agentenv-alfworld/configs/mappings_{train,test,valid_*}.json`
   - key 为 `task_type + "_" + task_id`（与你 `data/train/alfworld_train.json` 的 `item_id` 一致）
2) 初始化 env client：`AlfWorldEnvClient(env_server_base=..., timeout=...)`，`reset(game=<int_id>)`
3) 初始化会话：
   - 复用 `AlfWorldAdapter.conversation_start_dict[ActionFormat.REACT]` 的两条固定开场（system/user + assistant ack）
4) 循环 `t in [1..max_rounds]`：
   - 用 tokenizer 的 chat template 把当前 messages 转为 `input_ids`
   - 调用 `model.generate(max_new_tokens=max_new_tokens_per_turn, ...)` 生成本轮 assistant 文本
   - 追加 assistant message，并从 env 拿到 feedback：`env.step(action_text)`，再追加 user message
   - done 则提前结束

> 注意：测试的关键不是 “生成质量”，而是 “token 拼接 + mask 对齐 + log_prob 计算一致”。因此只要和训练使用同一个 tokenizer/chat_template，并复用同样的 `add_user_message/add_assistant_message` 规则即可。

### 8.3 对齐训练：生成 `response_mask` 与 `env_feedback_mask`

建议 **不要** 在测试脚本里再手写一套 mask 规则：

- `--mode pipeline`：直接使用 rollout 输出的 `response_mask` 与 `env_feedback_mask`（这两条 mask 必须与 `responses` 同步 slice/pad）。
- `--mode transformers`：复用/调用你已经改造过的 `RolloutHandler` 逻辑：

- 构建一个 `RolloutHandler`，每次追加消息都走：
  - `add_assistant_message()`：维护 `loss_mask`（用于 `response_mask`）
  - `add_user_message()`：维护 `env_mask`（用于 `env_feedback_mask`）
- 结束后调用 `truncate_output_ids()`：
  - 得到 `response_ids`（response-side token）
  - 得到 `response_loss_mask`（=`response_mask`）
  - 得到 `response_env_mask`（=`env_feedback_mask`）

### 8.4 计算 token-level `log_prob`（“传给优化器”的那条概率）

对齐 PPO/GRPO 里 actor 侧的 token log_prob 定义：

- `--mode pipeline`：使用 `wg.compute_log_prob(output)` 得到 `old_log_probs`，这是训练里真正用来做 policy loss 的那条概率。
- （可选）若你想在 **FSDP 重算侧** 对比 `log_probs` vs `old_log_probs`（排查 train/eval 差异），可以对同一条 batch：
  - 正常调用一次 `compute_log_prob`（得到 `old_log_probs`）
  - 再设置 `batch.meta_info['log_prob_train_mode']=True` 调一次 `compute_log_prob`，并把输出 rename 成 `log_probs`
- `--mode transformers`：按下述方式手动计算（可作为 pipeline 的 cross-check）。

- 用模型对 **完整序列**前向：`full_input_ids = prompt_ids + response_ids`（包含 env feedback token）
- 取 `logits` 做 `log_softmax`，对每个位置 gather 下一 token 的 log prob：
  - 对 response 第 j 个 token（full index = `prompt_len + j`），使用 `logits[prompt_len + j - 1]`
  - 得到 `log_prob[j]`，shape `[response_len]`
- 用 `response_mask` / `env_feedback_mask` 作为 overlay（不改变 log_prob 本身，只影响标注/统计）

（可选）同时输出几个 sanity check：

- `assert (response_mask & env_feedback_mask).sum() == 0`（内容 token 层面应互斥；模板/pad 允许都为 0）
- 打印：
  - `masked_mean(log_prob, response_mask)`
  - `masked_mean(log_prob, env_feedback_mask)`

### 8.5 画图（x=token 字符串，y=log_prob，用颜色标注 mask）

建议用 `matplotlib`：

- x 轴：token index（绘图坐标），tick label：`tokenizer.decode([token_id], clean_up_tokenization_spaces=False)` 的字符串
- y 轴：对应 token 的 `log_prob`
- 颜色规则（优先级）：
  - `env_feedback_mask==1`：红色
  - `response_mask==1`：蓝色
  - 其他（模板 token / padding / 未参与）：灰色
  - 若出现 overlap（不应出现）：紫色并 raise/报警
- 为避免图不可读：
  - 默认只画前 `--plot_max_tokens` 个 response token
  - 或支持 `--plot_start` / `--plot_end` 选一个窗口

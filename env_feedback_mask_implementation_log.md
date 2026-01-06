# `env_feedback_mask` 落地实现记录（Rollout → Optimizer 对齐）

> 目标：在 **训练 pipeline 实际传给优化器** 的 trajectory 数据里，新增并导出一条与 `responses` token 对齐的 `env_feedback_mask`，用于标记 **环境反馈（user/obs）内容 token**；并提供一个脚本在 ALFWorld 上复用训练 pipeline rollout + `old_log_probs` 计算，画出 token-level `log_prob` 并用 mask 标注。

---

## 0) 设计文档更新（便于复用训练 pipeline）

**修改文件**
- `env_feedback_mask_plan.md`

**修改点**
- 在“测试脚本设计”中补充了 `--mode pipeline`（推荐）方案：直接复用训练的 `RayPPOTrainer.fit()` 中那条 `generate_sequences → compute_log_prob` 路径拿到：
  - rollout 输出的 `responses/response_mask/env_feedback_mask`
  - 训练一致的 `old_log_probs`

---

## 1) `RolloutHandler`：新增 `env_mask` 并与 token 拼接同步

**修改文件**
- `AgentGym-RL/verl/workers/rollout/schemas.py`

**修改点**
- `RolloutHandler.__init__`：
  - 新增 `env_mask / prompt_env_mask / response_env_mask` 三个字段（与 `loss_mask` 的组织方式一致）。
  - 初始化时 `env_mask` 默认全 0（prompt 部分不标记 env）。
- `RolloutHandler.add_user_message()`：
  - 新增参数 `mark_env_feedback: bool=True`，用于控制本条 user message 的 **content token** 是否计入 env feedback：
    - `mark_env_feedback=True`：**仅 content token** 标记为 1
    - `mark_env_feedback=False`：content token 标记为 0（用于 **初始任务 prompt / S0**，避免把“问题本身”算作 env feedback）
    - `user_prefix_msg` / `user_suffix_msg`（模板 token）标记为 0
  - 这样 `env_mask` 精确对应 “环境反馈文本内容 token”，而不是整段 user message 的模板/分隔符。
- `RolloutHandler.add_assistant_message()`：
  - 对 assistant 追加的所有 token（含模板 prefix/suffix/content）都维护 `_env_mask=0`。
- `RolloutHandler.truncate_output_ids()`：
  - 对齐现有 `response_*` 的截断方式，新增：
    - `self.response_env_mask = self.env_mask[len(self.prompt_env_mask):][: self.max_response_len]`

**关键对齐约定**
- `env_mask` 与 `input_ids` **逐 token 同步生长**，并保证：
  - `len(input_ids) == len(attention_mask) == len(position_ids) == len(loss_mask) == len(env_mask)`
- `response_env_mask` 与 `response_ids/response_loss_mask` 的 slice 逻辑一致，因此能保证 response-side 对齐。

**测试方法（单元级，不依赖 env server）**
- 目标：验证 `add_user_message/add_assistant_message` 维护的 `env_mask` 长度与内容标注是否符合预期。
- 方法（建议在有本地 tokenizer 的环境下执行）：
  1) 随便找一个可用 tokenizer（与训练一致更好）
  2) 构造一个 `RolloutHandler`（prompt 部分随意，只要能触发 prefix/suffix 分支）
  3) 调用：
     - `add_user_message(tokenizer, "OBS_TEXT")` → 期望 env_mask 对应 `OBS_TEXT` 的 token 为 1
     - `add_user_message(tokenizer, "S0_TEXT", mark_env_feedback=False)` → 期望 env_mask 对应 `S0_TEXT` 的 token 全为 0
     - `add_assistant_message(tokenizer, "ACTION_TEXT")` → 期望 env_mask 全 0
  4) 调 `truncate_output_ids()` 检查 `response_env_mask` 与 `response_ids` 同长

---

## 2) vLLM rollout：把 `response_env_mask` pad 后写入 `DataProto.batch['env_feedback_mask']`

**修改文件**
- `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`

**修改点**
- `preprocess_prompt_to_rollout_handler()`：
  - 构造 `RolloutHandler` 时补齐初始化参数：
    - `env_mask=[0]*len(input_ids)`
    - `prompt_env_mask=[0]*len(input_ids)`
    - `response_env_mask=[]`
- `generate_sequences()`：
  - env reset 后第一次 `observe()` 得到的 `task` 视为 **初始任务 prompt（S0）**：
    - 调 `rollout_handler.add_user_message(..., mark_env_feedback=False)`，避免把“问题本身”标成 env feedback
  - 后续每轮 `env.step()` 得到的 `state` 仍用默认 `mark_env_feedback=True` 标成 env feedback
  - 在收集 `response_ids/response_loss_mask` 的同时，收集 `rollout_handler.response_env_mask`
  - 使用同样的 `pad_sequence` + `pad_sequence_to_length(..., self.config.response_length, 0)` pad 到固定长度
  - 最终写入：
    - `env_feedback_mask = response_env_mask`
    - `batch['env_feedback_mask'] = env_feedback_mask`

**关键对齐约定**
- `env_feedback_mask.shape == response_mask.shape == responses.shape == (bs, response_length)`
- response-side 的 valid length 可用 `attention_mask[:, prompt_len:].sum(-1)` 来取非 padding token 范围（debug/可视化脚本使用此规则）。

**测试方法（集成级，依赖 env server + 模型 + GPU）**
- 目标：确认 pipeline rollout 输出包含 `env_feedback_mask` 且与 `responses/old_log_probs` 对齐。
- 方法：
  - 直接运行下面第 3 节脚本（脚本内置 shape/assert/overlap 检查）：
    - 断言 `env_feedback_mask` 存在
    - 断言 `env_feedback_mask` 与 `response_mask` 不重叠
    - 断言与 `old_log_probs` token 维度一致

---

## 3) 可视化脚本：复用训练 pipeline 生成轨迹 + `old_log_probs`

**新增文件**
- `scripts/debug/plot_env_feedback_mask_logprob.py`

**复用点（尽量贴近训练）**
- 使用与训练一致的 driver / worker 组织方式（复用 `main_ppo.py` 的资源池与 `RayPPOTrainer` 初始化流程）：
  - `RayPPOTrainer.init_workers()` 创建 `actor_rollout_wg`（以及必要时的 ref/critic）
  - 复刻 `RayPPOTrainer.fit()` 的关键 dataflow：`generate_sequences → compute_log_prob`
  - 得到 rollout trajectory（`responses/response_mask/env_feedback_mask/...`）与 **训练里优化器用的** `old_log_probs`
- prompt 构造复用 `RLHFDataset._build_messages()` 的逻辑（system prompt + 两条固定开场）；`--item_id` 模式下直接按同样规则手工构造一条样本。

**补充：在 FSDP 重算侧同时拿到 `log_probs` 与 `old_log_probs`**

为了在同一条轨迹上对比：
- `old_log_probs`（训练里存到 batch 里的“旧策略概率”）
- `log_probs`（同样走 FSDP 重算，但强制用 train mode，便于排查 train/eval 行为差异）

新增了一个轻量开关（默认不影响训练）：

- `AgentGym-RL/verl/workers/agent_actor/dp_actor.py`
  - `DataParallelPPOActor.compute_log_prob(..., train_mode: bool=False)`：支持在 train/eval 两种 mode 下重算 log_prob，并在结束后恢复原 mode
- `AgentGym-RL/verl/workers/agent_fsdp_workers.py`
  - `ActorRolloutRefWorker.compute_log_prob()`：读取 `data.meta_info['log_prob_train_mode']`，透传到 `train_mode` 参数

脚本里会：
1) 正常调用一次 `compute_log_prob` 产出 `old_log_probs`
2) 设置 `batch.meta_info['log_prob_train_mode']=True` 再调一次 `compute_log_prob`，并把输出 rename 成 `log_probs`
3) 画图时分别输出两张图：`<stem>_old_log_probs.png` 与 `<stem>_log_probs_train.png`（`--out_png` 传 base），并支持用 `--tokens_per_row` 将完整轨迹自动换行绘制

**脚本输出**
- `--out_png`：token-level `log_prob` 图（会输出两张 png）
  - y 轴：每个 token 的 `log_prob`
  - x 轴：逐 token decode 字符串（过长时建议用 `--plot_max_tokens` 控制）
  - 红色点：`env_feedback_mask==1`
  - 蓝色点：`response_mask==1`
  - 灰色点：`response_mask==0 && env_feedback_mask==0`（主要是模板/分隔符 token，以及 S0 的 prompt token）
  - 当 `log_prob < -1` 时，对应 x 轴 token label 会染成其 mask 颜色（红/蓝/灰），便于定位低概率 token
- （可选）`--dump_jsonl`：逐 token dump（token_id/token_str/log_prob/masks），便于二次分析

**补充：如何控制每轮生成长度**
- 脚本参数 `--max_new_tokens_per_turn` 会直接透传到 vLLM 的 `SamplingParams.max_tokens`（等价于训练里的 `actor_rollout_ref.rollout.max_tokens`）。

**测试方法（推荐：端到端跑一次）**
1) 确保 ALFWorld env server 已启动（例如 `http://127.0.0.1:36001`）
2) 选择本地模型目录，并确保离线可用
3) 单卡最小配置运行（示例）：

```bash
python scripts/debug/plot_env_feedback_mask_logprob.py \
  --override actor_rollout_ref.agentgym.task_name=alfworld \
  --override actor_rollout_ref.agentgym.env_addr=http://127.0.0.1:36001 \
  --override actor_rollout_ref.model.path=/ABS/PATH/TO/LOCAL/HF/MODEL \
  --override data.train_file=data/train/alfworld_train.json \
  --override data.train_batch_size=1 \
  --override data.shuffle=false \
  --override data.max_prompt_length=1024 \
  --override data.max_response_length=8192 \
  --override trainer.nnodes=1 \
  --override trainer.n_gpus_per_node=1 \
  --override actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  --override actor_rollout_ref.rollout.n=1 \
  --override actor_rollout_ref.rollout.max_tokens=200 \
  --plot_max_tokens 256 \
  --out_png ./debug/env_feedback_mask_logprob.png \
  --dump_jsonl ./debug/env_feedback_mask_logprob.jsonl
```

若脚本成功：
- 会生成 `./debug/env_feedback_mask_logprob_old_log_probs.png` 与 `./debug/env_feedback_mask_logprob_log_probs_train.png`
- 并在终端打印保存路径
- 若 mask/shape 对齐失败，会直接 `assert` 报错（用于快速定位对齐问题）

---

## 4) 其他改动（与 `env_feedback_mask` 无直接耦合）

**修改文件**
- `examples/train/AgentGym-RL/alfworld_train.sh`

**修改点**
- 将示例脚本里的默认 `train_batch_size` 从 16 调整为 8（更保守的默认值，便于在资源较小的机器上先跑通）。

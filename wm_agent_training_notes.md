# World2World：WM 与 Agent 训练要点记录（2026-01-15）

本文件记录我们这次对话中关于数据生成、prompt 结构与潜在泄漏风险的关键结论。

## 1) 各数据目录用途

- `data/llama_factory/*`：世界模型（WM）SFT 训练数据，LLaMA-Factory 的 chat 格式。
  - 每条样本是完整/较长轨迹：`system`（环境信息）+ `user`（动作）+ `assistant`（环境反馈）。
  - 通过 `scripts/download_data/download_data.py` 从 HF 数据集下载得到，本仓库不生成。
- `data/init_contexts/*`：初始化上下文，用于**评估/交互**，不是 WM 训练集。
  - `agent_instruct_*.json`：固定 3 条消息（`user`、`assistant`、`user`）。
  - `wm_instruct_*.json`：仅 1 条 `system`，用于 WM 起始状态。

## 2) `data/init_contexts` 的生成方式

### Agent 初始 prompt

由 `scripts/collect_init_context/collect_agent_instruct.py` 生成（经 `collect_agent_instruct.sh` 调用）。

对每个任务实例：
1) `conversation_start[0]` -> `user`
2) `conversation_start[1]` -> `assistant`
3) `observe()` -> `user`

结果：`agent_instruct_*` **没有 `system` 角色**。

### WM 初始 prompt（以 ALFWorld 为例）

由 `scripts/collect_init_context/collect_wm_instruct_alfworld.py` 生成。

对每个 `data_idx`：
1) 用 `AgentGym/agentenv-alfworld/configs/mappings_*.json` 将 `data_idx` 映射到 `task_type/task_id`。
2) 读取 `~/.cache/alfworld/json_2.1.1/<split>/<task_type>/<task_id>/traj_data.json` 及同目录下的 `initial_state.pddl`。
3) 从 `initial_state.pddl` 解析完整环境描述。
4) 拼接为 system prompt：

```
# Environment Information (Only visible to Assistant)
{env_description}

# User Environment Information (Displayed to User)
{agent_instruct[2].content}
```

### 为什么 WM prompt 更“重”

- WM prompt 包含**完整状态**（来自 `initial_state.pddl`）。
- Agent prompt 只包含**可见观测**（`observe()` 返回文本）。

## 3) “Objects on Receptacles” 从哪来（ALFWorld）

在 `scripts/collect_init_context/collect_wm_instruct_alfworld.py` 中：

- `parse_relations()` 解析 `initial_state.pddl` 中的 `inReceptacle(obj, recep)`。
- 构造 `contained_pairs` 并渲染为：
  - `On the <receptacle>, you see a <obj>, ...`
  - 若为可开合容器，带 opened/closed 语句。

所以该段**源自 `initial_state.pddl`**，不是 agent 的 prompt。

## 4) Agent 训练（ALFWorld RL）的 system prompt

`slurm/alfworld-l40s.slurm` 里训练调用：

```
bash examples/train/AgentGym-RL/alfworld_train.sh
```

`AgentGym-RL/verl/utils/agent_dataset/rl_dataset.py` 中：
- 只用 `conversation_start` 生成 `user/assistant`，不注入环境状态。
- **固定**加一段 Qwen 系统头（chat template）：

```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

结论：**RL 训练有固定 system header，但不包含环境特权信息。**

## 5) WM SFT 是否会用这个 Qwen system prompt

README 的 WM 训练配置使用 `data/llama_factory/*_train*.json`（`template: qwen`）。

WM 数据本身就有 `system`（环境信息），因此不会使用 RL 里那个固定 Qwen system header。

## 6) 数据泄漏风险核对（基于仓库内文件）

潜在泄漏 = WM-SFT 训练实例与评测实例重叠，或 agent 训练误用 WM 的特权 prompt。

已做的交叉检查：

- **ALFWorld**：`mappings_train.json` vs `mappings_test/valid_*`  
  -> `(task_type, task_id)` 与 `item_id` 均 **0 重叠**。
- **TextWorld**：`data/llama_factory/textworld_train_58805.json` vs `data/eval/textworld_test.json`  
  -> **0 重叠**。
- **Webshop**：`data/llama_factory/webshop_train_70790.json` vs `data/eval/webshop_test.json`  
  -> **0 重叠**。
- **SciWorld**：`data/train/sciworld_train.json` vs `data/eval/sciworld_test.json`  
  -> **存在 61 个 id 重叠**。WM-SFT 文件无 id，需进一步确认。

结论：
- README 的 WM-SFT 配置在 ALFWorld/TextWorld/Webshop 上看起来安全。
- SciWorld 需要额外核查 split 是否严格互斥。

## 7) 关键参考文件

- WM 训练数据：`data/llama_factory/*.json`
- 初始化上下文：`data/init_contexts/*`
- WM prompt 构建：
  - `scripts/collect_init_context/collect_wm_instruct_alfworld.py`
  - `scripts/collect_init_context/collect_wm_instruct_sciworld.py`
  - `scripts/collect_init_context/collect_wm_instruct_textworld.py`
  - `scripts/collect_init_context/collect_wm_instruct_webshop.py`
- Agent prompt 构建：`scripts/collect_init_context/collect_agent_instruct.py`
- ALFWorld 映射表：`AgentGym/agentenv-alfworld/configs/mappings_*.json`
- RL 训练入口：`examples/train/AgentGym-RL/alfworld_train.sh`
- RL prompt 模板：`AgentGym-RL/verl/utils/agent_dataset/rl_dataset.py`


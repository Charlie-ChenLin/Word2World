# Multi-turn RL 训练快速指南（ALFWorld）

目标：用 `scripts/env_server/start_alfworld.sh` 启动环境，用 `examples/train/AgentGym-RL/alfworld_train.sh` 启动训练。

## 必须理解的 3 件事

1) **环境 server 和训练是两个进程**：训练脚本不会自动启动 server。  
2) **通常是两个 venv**：`uv_alfworld` 跑 server；`uv_agentgym_rl` 跑训练。  
3) **默认连本机端口**：训练脚本默认 `env_addr=http://127.0.0.1:36001`，所以 server 与训练需在**同一台机器/同一 Pod**（除非你改地址）。

## 路径检查（跑之前先对齐）

### 1) venv 位置（脚本默认在 repo 根目录找 `uv_*`）

需要满足：

- `uv_alfworld/bin/activate`
- `uv_agentgym_rl/bin/activate`

如果你的 venv 在 `../uv_alfworld`、`../uv_agentgym_rl`，在 repo 根目录建软链：

```bash
ln -s ../uv_alfworld uv_alfworld
ln -s ../uv_agentgym_rl uv_agentgym_rl
```

### 2) ALFWorld 环境数据（server 读取）

ALFWorld server 默认读取：`$HOME/.cache/alfworld`。检查：

```bash
ls "$HOME/.cache/alfworld/json_2.1.1/train" 1>/dev/null
```

如果你的 `~/` 目录空间不够，推荐把数据放到仓库内的 `./.cache/alfworld`，再用软链把 `$HOME/.cache/alfworld` 指过去（无需改任何代码/脚本）：

```bash
cd /path/to/Word2World
mkdir -p .cache ~/.cache

# 方式 A：已存在 ~/.cache/alfworld（把数据挪到仓库内）
mv ~/.cache/alfworld "$PWD/.cache/alfworld"
ln -sfn "$PWD/.cache/alfworld" ~/.cache/alfworld

# 方式 B：从 data/alfworld.zip 解压到仓库内（如果你是自己手动准备 zip）
# unzip -o data/alfworld.zip -d .cache
# rm -rf ~/.cache/alfworld
# ln -sfn "$PWD/.cache/alfworld" ~/.cache/alfworld

# 验证（应显示为一个 symlink）
# ls -ld ~/.cache/alfworld
```

> 注意：`scripts/download_data/download_data.py` 会 `rm -rf ~/.cache/alfworld`（会把这个软链删掉），如果你用它重新下载/解压，需要再执行一次 `ln -sfn ...`。

#### 软链排查（本次踩坑记录）

- **期望现象**：`ls -ld ~/.cache/alfworld` 显示 `lrwx... ~/.cache/alfworld -> /path/to/Word2World/.cache/alfworld`
- **错误现象**：`ls -ld ~/.cache/alfworld` 显示 `drwx...`，且 `ls -l ~/.cache/alfworld` 里出现了 `alfworld -> .../.cache/alfworld`（说明你原来就有 `~/.cache/alfworld` 目录，`ln` 把软链创建到了目录里面）
- **修复方式（保留备份）**：

```bash
cd /path/to/Word2World
mv ~/.cache/alfworld ~/.cache/alfworld.bak
ln -s "$PWD/.cache/alfworld" ~/.cache/alfworld
ls -ld ~/.cache/alfworld
```

确认软链生效且数据正常后，可删除备份释放空间：`rm -rf ~/.cache/alfworld.bak`

### 3) 训练数据（trainer 读取）

训练脚本读取：`${DATA_ROOT:-data}/AgentItemId/alfworld_train.json`。检查/设置：

```bash
export DATA_ROOT=/path/to/data_root
ls "$DATA_ROOT/AgentItemId/alfworld_train.json"
```

### 4) 模型路径（本地目录）

训练脚本默认：`models/Qwen2.5-7B-Instruct`，建议用环境变量覆盖：

```bash
export AGENT_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
ls "$AGENT_MODEL_PATH/config.json"
```

## 两个 terminal 启动（推荐）

### Terminal A：启动 ALFWorld server（端口 36001）

```bash
cd /path/to/Word2World
bash scripts/env_server/start_alfworld.sh
```

验证（可选）：

```bash
curl -s http://127.0.0.1:36001/
```

### Terminal B：启动训练

```bash
cd /path/to/Word2World
export DATA_ROOT=/path/to/data_root
export AGENT_MODEL_PATH=/path/to/model
export N_GPUS_PER_NODE=1
bash examples/train/AgentGym-RL/alfworld_train.sh
```

> 训练启动时 dataloader 也会连接环境 server，所以 server 要先起来。

## K8s 备注（只保留关键差异）

- **同一个 Pod**：两次 `kubectl exec -it <pod> -- bash` 进入同一 Pod，分别跑 server/训练即可；`127.0.0.1:36001` 仍然有效。
- **不同 Pod**：不能用 `127.0.0.1`，需要给 server 建 Service，然后把训练脚本里的 `env_server_url` 改成 `http://<service>:36001`（例如 `http://alfworld-env:36001`）。

## 常见报错快速定位

- `Connection refused`：server 没起 / 端口不对 / server 与训练不在同一 Pod（但还在用 `127.0.0.1`）。
- `uv_alfworld/bin/activate: No such file`：venv 不在 repo 根目录，按上面建软链或改脚本路径。
- 找不到 `~/.cache/alfworld` 或 `alfworld_train.json`：数据路径没准备好或 `DATA_ROOT` 未设置正确。

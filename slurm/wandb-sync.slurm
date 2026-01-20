#!/bin/bash

#SBATCH --job-name=wandb_sync_to_name
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32                 # 每个任务分配 CPU 数
#SBATCH --mem=32G
#SBATCH --mail-type=all
#SBATCH --mail-user=chenlin114514@sjtu.edu.cn # 邮件地址
#SBATCH --output=/home/lchen/log/%j.out
#SBATCH --error=/home/lchen/log/%j.err
#SBATCH --time=3-00:00:00

set -euo pipefail

# ------------------------ 环境初始化 ------------------------
# 加载 Conda 环境
source /home/lchen/home/uv_agentgym_rl/bin/activate


# 代理（如不需要可注释）
export https_proxy=http://192.168.102.101:7890
export http_proxy=http://192.168.102.101:7890
export all_proxy=socks://192.168.102.101:7890

# W&B 凭证（已在脚本内设置，避免依赖外部登录）
export WANDB_API_KEY="2e3f0cdfb6089a1e52ee117856ca6dfcf331858b"

# # 可选：按需加载 module（与 conda 共存时通常不必须）
# module load python/3.8 || true

# ------------------------ 配置区（按需修改） ------------------------
# 这里支持多组配置；每一行是 "WANDB_DIR|TARGET_PROJECT"
# 示例：
#   CONFIGS=(
#     "/home/user/proj1/wandb|Project-One"
#     "/home/user/proj2/wandb|Project-Two"
#   )
CONFIGS=(
    "/home/lchen/home/Word2World/wandb|hybrid-rl-alfworld"
)

# 目标 entity（可选；留空则使用当前凭证默认 entity）
ENTITY="1969773923-shanghai-jiao-tong-university"

# ------------------------ 逻辑区：查找与同步 ------------------------
# 检查 wandb CLI 是否可用
if ! command -v wandb >/dev/null 2>&1; then
  echo "错误：未找到 'wandb' 命令。请确认当前环境已安装：pip install wandb"
  exit 2
fi

JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}   # 并发同步数，可通过环境变量覆盖

for entry in "${CONFIGS[@]}"; do
  IFS="|" read -r WANDB_DIR TARGET_PROJECT <<< "$entry"

  if [[ -z "${WANDB_DIR:-}" || -z "${TARGET_PROJECT:-}" ]]; then
    echo "跳过无效配置：$entry"
    continue
  fi

  if [[ ! -d "$WANDB_DIR" ]]; then
    echo "未找到目录：$WANDB_DIR，已跳过（项目：$TARGET_PROJECT）"
    continue
  fi

  echo "---------------- 配置 ----------------"
  echo "root=$WANDB_DIR"
  echo "project=$TARGET_PROJECT"
  echo "entity=${ENTITY:-<default>}"
  echo "并发=$JOBS"
  echo "--------------------------------------"

  echo "正在递归查找离线运行目录..."
  # 仅匹配 offline-run-202511*，限制在各 wandb 目录下一层，避免深层重复匹配
  mapfile -t WDIRS < <(find "$WANDB_DIR" -type d -name "wandb")
  RUN_DIRS=()
  for w in "${WDIRS[@]}"; do
    while IFS= read -r -d '' d; do
      RUN_DIRS+=("$d")
    # done < <(find "$w" -maxdepth 1 -mindepth 1 -type d -name "offline-run-202511*" -print0)
    done < <(find "$w" -maxdepth 1 -mindepth 1 -type d -name "offline-run*" -print0)
  done

  COUNT=${#RUN_DIRS[@]}
  echo "找到 $COUNT 个离线运行目录。"
  if [[ $COUNT -eq 0 ]]; then
    echo "没有需要同步的内容，跳过该配置。"
    continue
  fi
  TOTAL=$COUNT

  # 并发同步；--no-include-online 确保只同步离线 run；强制指定 project；可选 entity
  export TARGET_PROJECT ENTITY TOTAL
  # 为每个 run 编号，格式 idx|path
  INPUTS=()
  idx=0
  for d in "${RUN_DIRS[@]}"; do
    idx=$((idx + 1))
    INPUTS+=("$idx|$d")
  done

  printf '%s\0' "${INPUTS[@]}" | xargs -0 -n1 -P "$JOBS" bash -c '
    IFS="|" read -r idx d <<< "$1"
    prefix="[$idx/$TOTAL]"
    if [[ -n "$ENTITY" ]]; then
      echo "$prefix 同步：$d → project=$TARGET_PROJECT, entity=$ENTITY"
      if wandb sync --no-include-online --project "$TARGET_PROJECT" --entity "$ENTITY" "$d"; then
        echo "$prefix 成功同步：$d"
      else
        echo "$prefix 同步失败（已跳过）：$d"
      fi
    else
      echo "$prefix 同步：$d → project=$TARGET_PROJECT"
      if wandb sync --no-include-online --project "$TARGET_PROJECT" "$d"; then
        echo "$prefix 成功同步：$d"
      else
        echo "$prefix 同步失败（已跳过）：$d"
      fi
    fi
  ' _ 

  echo "完成该配置：已处理 $COUNT 个离线运行目录。"
done

echo "全部配置处理完成。"

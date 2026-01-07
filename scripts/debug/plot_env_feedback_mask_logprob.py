#!/usr/bin/env python3
"""
Roll out a single multi-turn trajectory (AgentGym / ALFWorld), recompute token-level old_log_probs
with the same pipeline used by training, and visualize per-token log_prob with env/assistant masks.

This script computes and visualizes both:
- old_log_probs: the "old policy" log probs (compute_log_prob in eval mode, as in training)
- log_probs: log probs computed in train mode (same FSDP recompute path; useful to spot eval/train differences)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "AgentGym-RL", REPO_ROOT / "AgentGym"):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.agent_trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker, CriticWorker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml",
        help="Base OmegaConf yaml (ppo_trainer.yaml).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf dotlist override, can be repeated. Example: actor_rollout_ref.agentgym.task_name=alfworld",
    )
    parser.add_argument(
        "--item_id",
        default=None,
        help="AgentGym item_id (for ALFWorld: '<task_type>_<task_id>'). If omitted, use the first batch from the train dataloader.",
    )
    parser.add_argument("--max_rounds", type=int, default=None, help="Max env interaction rounds (turns).")
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=None,
        help="How many rollouts to sample for the same prompt (sets actor_rollout_ref.rollout.n). If omitted, keep config value.",
    )
    parser.add_argument(
        "--max_new_tokens_per_turn",
        type=int,
        default=None,
        help="Max new tokens per turn (align with actor_rollout_ref.rollout.max_tokens).",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample in the (possibly repeated) batch to plot.")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Output directory under which to save executer_logs and plots. Default: ./debug/<timestamp>.",
    )
    parser.add_argument(
        "--verl_ckpt",
        default=None,
        help=(
            "Optional VERL checkpoint path to load before rollout/logprob. "
            "Accepts either a `global_step_*/` folder (containing `actor/`) or an `actor/` folder."
        ),
    )
    parser.add_argument(
        "--plot_max_tokens",
        type=int,
        default=0,
        help="Only plot first N valid response tokens. Use 0 to plot all tokens.",
    )
    parser.add_argument(
        "--tokens_per_row",
        type=int,
        default=200,
        help="Wrap plots into multiple rows; each row shows at most N tokens (use 0 to disable wrapping).",
    )
    parser.add_argument(
        "--lp_threshold",
        type=float,
        default=-1.0,
        help="Log-prob threshold for highlighting (log_prob < threshold), legend stats, and y-floor plots.",
    )
    parser.add_argument(
        "--out_png",
        default="env_feedback_mask_logprob.png",
        help=(
            "Output png path base. Two files will be written per rollout: "
            "'<stem>_log_probs_train.png', '<stem>_log_probs_train_clip_<thr>.png'."
        ),
    )
    parser.add_argument("--dump_jsonl", default=None, help="Optional dump jsonl path for token/logprob/masks.")
    return parser.parse_args()


def _load_cfg(config_path: str, overrides: list[str]) -> Any:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _decode_token(tokenizer, token_id: int) -> str:
    s = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    return s.replace("\n", "\\n").replace("\t", "\\t")


def _derive_png_path(base_png: Path, suffix: str) -> Path:
    if base_png.suffix.lower() != ".png":
        raise ValueError(f"--out_png must end with .png, got: {base_png}")
    return base_png.with_name(f"{base_png.stem}{suffix}{base_png.suffix}")


def _derive_per_rollout_base(base_png: Path, *, rollout_idx: int) -> Path:
    if base_png.suffix.lower() != ".png":
        raise ValueError(f"--out_png must end with .png, got: {base_png}")
    return base_png.with_name(f"{base_png.stem}_rollout{rollout_idx}{base_png.suffix}")


def _sanitize_for_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s))


def _compute_mask_segment_stats(
    mask: np.ndarray, log_probs: np.ndarray, token_strs: list[str] | None = None
) -> list[dict[str, Any]]:
    """Return list of contiguous segments where mask==1 with start/end and mean log_prob."""
    stats: list[dict[str, Any]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            end = idx - 1
            seg_lp = float(log_probs[start : end + 1].mean().item())
            stats.append({"start": start, "end": end, "mean": seg_lp, "len": end - start + 1})
            start = None
    if start is not None:
        end = len(mask) - 1
        seg_lp = float(log_probs[start : end + 1].mean().item())
        stats.append({"start": start, "end": end, "mean": seg_lp, "len": end - start + 1})

    if token_strs is not None:
        for seg in stats:
            seg_tokens = "".join(token_strs[seg["start"] : seg["end"] + 1]).strip().lower()
            seg["is_nothing_happens"] = "nothing happens" in seg_tokens
    return stats
    return stats


def _resolve_verl_ckpt_actor_path(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--verl_ckpt not found: {p}")

    # Accept either .../global_step_x/actor or .../global_step_x
    if p.is_dir() and p.name == "actor":
        return str(p)
    actor = p / "actor"
    if actor.is_dir():
        return str(actor)
    raise ValueError(f"--verl_ckpt must point to `.../global_step_*/` or `.../global_step_*/actor/`, got: {p}")


def _plot_wrapped_token_logprobs(
    *,
    plt,
    token_strs: list[str],
    y: np.ndarray,
    raw_y: np.ndarray | None,
    response_mask: np.ndarray,
    env_feedback_mask: np.ndarray,
    tokens_per_row: int,
    low_logprob_threshold: float,
    title: str,
    ylabel: str,
    out_path: Path,
    env_segment_stats: list[dict[str, Any]] | None = None,
    env_overall_mean: float | None = None,
    env_overall_mean_no_nothing: float | None = None,
    resp_segment_stats: list[dict[str, Any]] | None = None,
    resp_overall_mean: float | None = None,
    footnote_text: str | None = None,
) -> None:
    if raw_y is None:
        raw_y = y
    below_threshold = raw_y < low_logprob_threshold

    env_feedback_mask = env_feedback_mask.astype(bool)
    response_mask = response_mask.astype(bool)
    env_total = int(env_feedback_mask.sum())
    resp_total = int(response_mask.sum())
    env_below = int((env_feedback_mask & below_threshold).sum())
    resp_below = int((response_mask & below_threshold).sum())

    if env_total > 0:
        env_ratio = env_below / env_total
        env_label = (
            f"env_feedback_mask=1 (lp<{low_logprob_threshold:g}: {env_below}/{env_total}={env_ratio:.1%})"
        )
    else:
        env_label = f"env_feedback_mask=1 (lp<{low_logprob_threshold:g}: n/a)"

    if resp_total > 0:
        resp_ratio = resp_below / resp_total
        resp_label = f"response_mask=1 (lp<{low_logprob_threshold:g}: {resp_below}/{resp_total}={resp_ratio:.1%})"
    else:
        resp_label = f"response_mask=1 (lp<{low_logprob_threshold:g}: n/a)"
    n_tokens = len(token_strs)
    if tokens_per_row <= 0:
        tokens_per_row = n_tokens
    n_rows = max(1, math.ceil(n_tokens / tokens_per_row))

    fig_w = max(12.0, tokens_per_row * 0.08)
    fig_h = max(4.0, n_rows * 2.8)
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharey=True, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = [axes]

    for row_idx in range(n_rows):
        start = row_idx * tokens_per_row
        end = min(n_tokens, (row_idx + 1) * tokens_per_row)
        x = np.arange(end - start)

        row_y = y[start:end]
        row_y_raw = raw_y[start:end]
        row_token_strs = token_strs[start:end]
        row_env = env_feedback_mask[start:end]
        row_resp = response_mask[start:end]
        row_neither = ~(row_env | row_resp)

        ax = axes[row_idx]
        ax.plot(x, row_y, color="0.6", linewidth=1.0, alpha=0.8)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.2)
        ax.grid(True, axis="x", color="0.9", linewidth=0.5, alpha=0.6)

        if row_neither.any():
            neither_idx = np.nonzero(row_neither)[0]
            ax.scatter(neither_idx, row_y[neither_idx], s=10, color="0.5", label="mask=0" if row_idx == 0 else None)
        if row_env.any():
            env_idx = np.nonzero(row_env)[0]
            ax.scatter(env_idx, row_y[env_idx], s=10, color="red", label=env_label if row_idx == 0 else None)
        if row_resp.any():
            resp_idx = np.nonzero(row_resp)[0]
            ax.scatter(resp_idx, row_y[resp_idx], s=10, color="blue", label=resp_label if row_idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels(row_token_strs, rotation=90, fontsize=6)
        for i, label in enumerate(ax.get_xticklabels()):
            if row_y_raw[i] >= low_logprob_threshold:
                continue
            if row_resp[i]:
                label.set_color("blue")
            elif row_env[i]:
                label.set_color("red")
            else:
                label.set_color("0.5")

        # Annotate per-turn averages for env and response masks.
        y_min = float(row_y.min())
        y_max = float(row_y.max())
        y_span = y_max - y_min if y_max > y_min else 1.0
        # place annotations near the x-axis and stack upward to reduce overlap
        y_base_band = y_min + 0.05 * y_span
        annotate_gap = 0.10 * y_span

        band_idx = [0]  # shared stack counter per subplot row (env + resp)

        def _annotate_segments(stats, overall_mean, color, label_prefix, nn_mean=None):
            if not stats or overall_mean in (None, 0):
                return
            for seg in stats:
                if seg["end"] < start or seg["start"] >= end:
                    continue
                local_start = max(seg["start"], start)
                local_end = min(seg["end"], end - 1)
                mid_abs = (local_start + local_end) // 2
                mid = mid_abs - start
                if mid < 0 or mid >= len(row_y):
                    continue
                ratio = seg["mean"] / overall_mean if overall_mean else float("nan")
                ratio_nn = seg["mean"] / nn_mean if nn_mean not in (None, 0) else None
                y_pos = y_base_band + annotate_gap * band_idx[0]
                band_idx[0] += 1
                ax.text(
                    mid,
                    y_pos,
                    f"{label_prefix} mu={seg['mean']:.3f} r={ratio:.2f}x",
                    color=color,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="1.0", alpha=0.7, edgecolor="none"),
                )
                if ratio_nn is not None:
                    y_pos_nn = y_base_band + annotate_gap * band_idx[0]
                    band_idx[0] += 1
                    ax.text(
                        mid,
                        y_pos_nn,
                        f"{label_prefix} rNN={ratio_nn:.2f}x",
                        color=color,
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        bbox=dict(facecolor="1.0", alpha=0.7, edgecolor="none"),
                    )

        _annotate_segments(env_segment_stats, env_overall_mean, "darkred", "env", nn_mean=env_overall_mean_no_nothing)
        _annotate_segments(resp_segment_stats, resp_overall_mean, "navy", "resp", nn_mean=None)

        if row_idx == 0:
            # Ensure the legend shows all mask categories even if the first row contains none of them.
            if env_total > 0 and not row_env.any():
                ax.scatter([], [], s=10, color="red", label=env_label)
            if resp_total > 0 and not row_resp.any():
                ax.scatter([], [], s=10, color="blue", label=resp_label)
            ax.legend(loc="best")
            ax.set_ylabel(ylabel)
        if row_idx == n_rows - 1:
            ax.set_xlabel("decoded token (per-token)")

        ax.set_title(f"tokens {start}..{end - 1} ({end - start} tokens)")

    fig.suptitle(title, y=1.0)
    if footnote_text:
        fig.text(0.995, 0.01, footnote_text, ha="right", va="bottom", fontsize=7, color="0.25")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config, args.override)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (REPO_ROOT / "debug" / run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save rollout conversations the same way as training does (vllm_rollout writes json when meta_info['global_steps'] is set).
    cfg.actor_rollout_ref.rollout.rollout_log_dir = str(run_dir / "executer_logs")

    # Number of rollouts for the same prompt.
    if args.n_rollouts is not None:
        cfg.actor_rollout_ref.rollout.n = int(args.n_rollouts)

    if args.max_new_tokens_per_turn is not None:
        cfg.actor_rollout_ref.rollout.max_tokens = int(args.max_new_tokens_per_turn)

    # Allow the script to run without explicitly passing common env vars.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    # Tokenizer must match training (same as main_ppo.py).
    local_model_path = copy_local_path_from_hdfs(cfg.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(
        local_model_path, trust_remote_code=cfg.actor_rollout_ref.model.get("trust_remote_code", False)
    )
    tokenizer.padding_side = "left"

    # Build trainer and worker groups the same way as main_ppo.py (but we only run one batch).
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
    )
    trainer.init_workers()
    if args.verl_ckpt:
        actor_ckpt = _resolve_verl_ckpt_actor_path(args.verl_ckpt)
        print(f"[plot_env_feedback_mask_logprob] Loading VERL actor checkpoint: {actor_ckpt}")
        trainer.actor_rollout_wg.load_checkpoint(actor_ckpt, del_local_after_load=False)

    # Prepare one batch of prompts (either from dataloader, or a user-provided item_id).
    if args.item_id is None:
        batch_dict = next(iter(trainer.train_dataloader))
        batch: DataProto = DataProto.from_single_dict(batch_dict)
    else:
        # Build a single-sample prompt exactly like RLHFDataset._build_messages().
        env_client = trainer.train_dataset.env_client
        conversation_start_user = env_client.conversation_start[0]["value"]
        conversation_start_assistant = env_client.conversation_start[1]["value"]
        messages = [
            {"role": "user", "content": conversation_start_user},
            {"role": "assistant", "content": conversation_start_assistant},
        ]
        prompt_with_chat_template = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            + conversation_start_user
            + "<|im_end|>\n<|im_start|>assistant\n"
            + conversation_start_assistant
            + "<|im_end|>"
        )

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=cfg.data.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=cfg.data.get("truncation", "error"),
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        batch = DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            non_tensors={"item_id": [args.item_id], "raw_prompt": [messages]},
        )

    if "raw_prompt" not in batch.non_tensor_batch:
        raise KeyError("Missing `raw_prompt` in dataloader output. Set `data.return_raw_chat=True`.")

    # Mimic one iteration in RayPPOTrainer.fit(): pop prompt, rollout, then recompute log probs.
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=["item_id", "raw_prompt"],
    )
    gen_batch.meta_info["global_steps"] = run_tag
    if args.max_rounds is not None:
        gen_batch.meta_info["max_rounds"] = int(args.max_rounds)
    else:
        try:
            gen_batch.meta_info["max_rounds"] = int(cfg.algorithm.rounds_ctrl.rounds)
        except Exception:
            gen_batch.meta_info["max_rounds"] = int(cfg.actor_rollout_ref.agentgym.get("max_rounds", 10))

    gen_batch_output = trainer.actor_rollout_wg.generate_sequences(gen_batch)

    # Keep `item_id` on the driver side for plotting/debugging (training pops it into `gen_batch`).
    if "item_id" in gen_batch.non_tensor_batch:
        batch.non_tensor_batch["item_id"] = gen_batch.non_tensor_batch["item_id"]

    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch = batch.repeat(repeat_times=cfg.actor_rollout_ref.rollout.n, interleave=True)
    batch = batch.union(gen_batch_output)

    # Recompute log_probs in train mode (same FSDP path; useful to detect train/eval mismatches)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    batch.meta_info["log_prob_train_mode"] = True
    lp_train_mode = trainer.actor_rollout_wg.compute_log_prob(batch).rename(
        old_keys="old_log_probs", new_keys="log_probs"
    )
    batch.meta_info.pop("log_prob_train_mode", None)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install it or use --dump_jsonl.") from e

    out_base_cli = Path(args.out_png)
    # Always place figure outputs under the timestamped run_dir unless the user provided an absolute path.
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Note: callers (e.g., bash scripts) often pass an absolute path like `${REPO_ROOT}/debug/foo.png`.
    # We still force figures into `run_dir/figures` for easy run isolation; only the filename is respected.
    out_base = figures_dir / out_base_cli.name

    n_samples = int(batch.batch["responses"].shape[0])
    if args.sample_idx is not None and args.sample_idx >= 0:
        sample_indices = [int(args.sample_idx)]
    else:
        sample_indices = list(range(n_samples))

    print(f"[plot_env_feedback_mask_logprob] run_dir: {run_dir}")
    print(f"[plot_env_feedback_mask_logprob] executer_logs: {run_dir / 'executer_logs'}")
    print(f"[plot_env_feedback_mask_logprob] plotting {len(sample_indices)} rollout(s) (batch_size={n_samples})")

    for rollout_i, sample_idx in enumerate(sample_indices):
        prompt_len = batch.batch["prompts"].shape[-1]
        valid_len = int(batch.batch["attention_mask"][sample_idx, prompt_len:].sum().item())
        item_id = batch.non_tensor_batch["item_id"][sample_idx]
        reward = float(batch.batch["scores"][sample_idx, :valid_len].sum().item())

        responses = batch.batch["responses"][sample_idx, :valid_len].cpu()
        response_mask = batch.batch["response_mask"][sample_idx, :valid_len].to(torch.bool).cpu()
        env_feedback_mask = batch.batch["env_feedback_mask"][sample_idx, :valid_len].to(torch.bool).cpu()
        log_probs = lp_train_mode.batch["log_probs"][sample_idx, :valid_len].cpu()

        if args.plot_max_tokens is not None and args.plot_max_tokens > 0 and valid_len > args.plot_max_tokens:
            responses = responses[: args.plot_max_tokens]
            response_mask = response_mask[: args.plot_max_tokens]
            env_feedback_mask = env_feedback_mask[: args.plot_max_tokens]
            log_probs = log_probs[: args.plot_max_tokens]

        assert responses.shape == response_mask.shape == env_feedback_mask.shape == log_probs.shape
        overlap = (response_mask & env_feedback_mask)
        assert not overlap.any(), f"env_feedback_mask overlaps response_mask at {overlap.nonzero().flatten().tolist()}"

        token_ids = responses.tolist()
        token_strs = [_decode_token(tokenizer, tid) for tid in token_ids]

        env_segment_stats = _compute_mask_segment_stats(
            env_feedback_mask.numpy(), log_probs.numpy(), token_strs=token_strs
        )
        env_overall_mean = float(np.mean([s["mean"] for s in env_segment_stats])) if env_segment_stats else None
        env_overall_mean_no_nothing = (
            float(np.mean([s["mean"] for s in env_segment_stats if not s.get("is_nothing_happens", False)]))
            if env_segment_stats
            else None
        )
        resp_segment_stats = _compute_mask_segment_stats(response_mask.numpy(), log_probs.numpy(), token_strs=token_strs)
        resp_overall_mean = float(np.mean([s["mean"] for s in resp_segment_stats])) if resp_segment_stats else None

        dump_jsonl = None
        if args.dump_jsonl:
            dump_root = run_dir / "dumps"
            dump_root.mkdir(parents=True, exist_ok=True)
            dump_jsonl = dump_root / f"{out_base.stem}_rollout{rollout_i}_{_sanitize_for_filename(item_id)}.jsonl"
            with dump_jsonl.open("w", encoding="utf-8") as f:
                for i, (tid, tstr, lp_new, rm, em) in enumerate(
                    zip(
                        token_ids,
                        token_strs,
                        log_probs.tolist(),
                        response_mask.tolist(),
                        env_feedback_mask.tolist(),
                    )
                ):
                    f.write(
                        json.dumps(
                            {
                                "i": i,
                                "token_id": tid,
                                "token_str": tstr,
                                "log_prob": lp_new,
                                "response_mask": int(rm),
                                "env_feedback_mask": int(em),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        out_base_i = _derive_per_rollout_base(out_base, rollout_idx=rollout_i)
        out_train_path = _derive_png_path(out_base_i, "_log_probs_train")

        title = (
            f"{cfg.actor_rollout_ref.agentgym.task_name} | item_id={item_id} | rollout={rollout_i} | "
            f"reward={reward:g} | valid_tokens={valid_len} (prompt excluded)"
        )
        if env_overall_mean is not None:
            title += f" | env_turn_avg={env_overall_mean:.3f}"
        if env_overall_mean_no_nothing is not None:
            title += f" | env_turn_avg_noNothing={env_overall_mean_no_nothing:.3f}"
        if resp_overall_mean is not None:
            title += f" | resp_turn_avg={resp_overall_mean:.3f}"
        footnote_parts = [
            "mu=seg mean logp (mask==1)",
            "r=seg mu / overall mu (same mask)",
            "rNN=seg mu / env mu without 'Nothing happens.' turns",
            "env_mu_noNH excludes 'Nothing happens.' env turns",
        ]
        avg_summary = []
        if env_overall_mean is not None:
            avg_summary.append(f"env_mu={env_overall_mean:.3f}")
        if env_overall_mean_no_nothing is not None:
            avg_summary.append(f"env_mu_noNH={env_overall_mean_no_nothing:.3f}")
        if resp_overall_mean is not None:
            avg_summary.append(f"resp_mu={resp_overall_mean:.3f}")
        if avg_summary:
            footnote_parts.append(" | ".join(avg_summary))
        footnote = "; ".join(footnote_parts)
        _plot_wrapped_token_logprobs(
            plt=plt,
            token_strs=token_strs,
            y=log_probs.numpy(),
            raw_y=None,
            response_mask=response_mask.numpy(),
            env_feedback_mask=env_feedback_mask.numpy(),
            tokens_per_row=int(args.tokens_per_row),
            low_logprob_threshold=float(args.lp_threshold),
            title=title,
            ylabel="log_probs (train mode)",
            out_path=out_train_path,
            env_segment_stats=env_segment_stats,
            env_overall_mean=env_overall_mean,
            env_overall_mean_no_nothing=env_overall_mean_no_nothing,
            resp_segment_stats=resp_segment_stats,
            resp_overall_mean=resp_overall_mean,
            footnote_text=footnote,
        )

        thr_tag = f"{float(args.lp_threshold):g}"
        out_train_floor_path = _derive_png_path(out_base_i, f"_log_probs_train_clip_{thr_tag}")
        _plot_wrapped_token_logprobs(
            plt=plt,
            token_strs=token_strs,
            y=np.maximum(log_probs.numpy(), float(args.lp_threshold)),
            raw_y=log_probs.numpy(),
            response_mask=response_mask.numpy(),
            env_feedback_mask=env_feedback_mask.numpy(),
            tokens_per_row=int(args.tokens_per_row),
            low_logprob_threshold=float(args.lp_threshold),
            title=f"{title} | y_floored@{thr_tag}",
            ylabel=f"log_probs (train mode, floored @ {thr_tag})",
            out_path=out_train_floor_path,
            env_segment_stats=env_segment_stats,
            env_overall_mean=env_overall_mean,
            env_overall_mean_no_nothing=env_overall_mean_no_nothing,
            resp_segment_stats=resp_segment_stats,
            resp_overall_mean=resp_overall_mean,
            footnote_text=footnote,
        )

        print(f"[plot_env_feedback_mask_logprob] Saved log_probs (train mode) plot to: {out_train_path}")
        print(f"[plot_env_feedback_mask_logprob] Saved floored log_probs (train mode) plot to: {out_train_floor_path}")
        if dump_jsonl is not None:
            print(f"[plot_env_feedback_mask_logprob] Saved dump to: {dump_jsonl}")


if __name__ == "__main__":
    main()

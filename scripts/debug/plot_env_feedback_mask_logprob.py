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
import os
import sys
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
        "--max_new_tokens_per_turn",
        type=int,
        default=None,
        help="Max new tokens per turn (align with actor_rollout_ref.rollout.max_tokens).",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample in the (possibly repeated) batch to plot.")
    parser.add_argument("--plot_max_tokens", type=int, default=256, help="Only plot first N valid response tokens.")
    parser.add_argument("--out_png", default="env_feedback_mask_logprob.png", help="Output png path.")
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


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config, args.override)

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
    gen_batch.meta_info["global_steps"] = "debug_single"
    if args.max_rounds is not None:
        gen_batch.meta_info["max_rounds"] = int(args.max_rounds)
    else:
        try:
            gen_batch.meta_info["max_rounds"] = int(cfg.algorithm.rounds_ctrl.rounds)
        except Exception:
            gen_batch.meta_info["max_rounds"] = int(cfg.actor_rollout_ref.agentgym.get("max_rounds", 10))

    gen_batch_output = trainer.actor_rollout_wg.generate_sequences(gen_batch)

    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch = batch.repeat(repeat_times=cfg.actor_rollout_ref.rollout.n, interleave=True)
    batch = batch.union(gen_batch_output)

    # recompute old_log_probs (eval mode, as in training)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    old_lp = trainer.actor_rollout_wg.compute_log_prob(batch)

    # recompute log_probs in train mode (same FSDP path; useful to detect train/eval mismatches)
    batch.meta_info["log_prob_train_mode"] = True
    lp_train_mode = trainer.actor_rollout_wg.compute_log_prob(batch).rename(
        old_keys="old_log_probs", new_keys="log_probs"
    )
    batch.meta_info.pop("log_prob_train_mode", None)

    # Extract valid response window.
    sample_idx = int(args.sample_idx)
    if sample_idx < 0 or sample_idx >= len(batch):
        raise ValueError(f"--sample_idx out of range: {sample_idx} (batch_size={len(batch)})")

    prompt_len = batch.batch["prompts"].shape[-1]
    valid_len = int(batch.batch["attention_mask"][sample_idx, prompt_len:].sum().item())

    item_id = batch.non_tensor_batch["item_id"][sample_idx]
    responses = batch.batch["responses"][sample_idx, :valid_len].cpu()
    response_mask = batch.batch["response_mask"][sample_idx, :valid_len].to(torch.bool).cpu()
    if "env_feedback_mask" not in batch.batch.keys():
        raise KeyError("Missing `env_feedback_mask` in rollout output. Did you finish wiring it into vLLM rollout?")
    env_feedback_mask = batch.batch["env_feedback_mask"][sample_idx, :valid_len].to(torch.bool).cpu()

    old_log_probs = old_lp.batch["old_log_probs"][sample_idx, :valid_len].cpu()
    log_probs = lp_train_mode.batch["log_probs"][sample_idx, :valid_len].cpu()

    if args.plot_max_tokens is not None and valid_len > args.plot_max_tokens:
        responses = responses[: args.plot_max_tokens]
        response_mask = response_mask[: args.plot_max_tokens]
        env_feedback_mask = env_feedback_mask[: args.plot_max_tokens]
        old_log_probs = old_log_probs[: args.plot_max_tokens]
        log_probs = log_probs[: args.plot_max_tokens]

    assert responses.shape == response_mask.shape == env_feedback_mask.shape == old_log_probs.shape == log_probs.shape
    overlap = (response_mask & env_feedback_mask)
    assert not overlap.any(), f"env_feedback_mask overlaps response_mask at {overlap.nonzero().flatten().tolist()}"

    token_ids = responses.tolist()
    token_strs = [_decode_token(tokenizer, tid) for tid in token_ids]

    if args.dump_jsonl:
        dump_path = Path(args.dump_jsonl)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            for i, (tid, tstr, lp_old, lp_new, rm, em) in enumerate(
                zip(
                    token_ids,
                    token_strs,
                    old_log_probs.tolist(),
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
                            "old_log_prob": lp_old,
                            "log_prob": lp_new,
                            "response_mask": int(rm),
                            "env_feedback_mask": int(em),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install it or use --dump_jsonl.") from e

    x = np.arange(len(token_ids))
    y_old = old_log_probs.numpy()
    y = log_probs.numpy()
    fig_w = max(12.0, len(token_ids) * 0.08)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(fig_w, 7.5))

    axes[0].plot(x, y_old, color="0.6", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("old_log_probs")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].set_title(f"{cfg.actor_rollout_ref.agentgym.task_name} | item_id={item_id} | valid_tokens={valid_len}")

    axes[1].plot(x, y, color="0.6", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("log_probs (train mode)")
    axes[1].set_xlabel("decoded token (per-token)")
    axes[1].grid(True, axis="y", alpha=0.2)

    if env_feedback_mask.any():
        env_idx = env_feedback_mask.nonzero().flatten().numpy()
        axes[0].scatter(env_idx, y_old[env_idx], s=10, color="red", label="env_feedback_mask=1")
        axes[1].scatter(env_idx, y[env_idx], s=10, color="red", label="env_feedback_mask=1")
    if response_mask.any():
        resp_idx = response_mask.nonzero().flatten().numpy()
        axes[0].scatter(resp_idx, y_old[resp_idx], s=10, color="blue", label="response_mask=1")
        axes[1].scatter(resp_idx, y[resp_idx], s=10, color="blue", label="response_mask=1")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(token_strs, rotation=90, fontsize=6)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[plot_env_feedback_mask_logprob] Saved plot to: {out_path}")
    if args.dump_jsonl:
        print(f"[plot_env_feedback_mask_logprob] Saved dump to: {args.dump_jsonl}")


if __name__ == "__main__":
    main()

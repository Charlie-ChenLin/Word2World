#!/usr/bin/env python3
"""Pure rollout for WebShop with vLLM (no training updates).

Generates trajectories and writes executer_logs via vllm_rollout.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import ray
from omegaconf import OmegaConf

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker


def _load_item_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "step_item_ids" in data:
        # Flatten list of list, keep order
        flat = []
        for batch in data["step_item_ids"]:
            flat.extend(batch)
        return flat
    if isinstance(data, list):
        # Either list of item_ids or list of list
        if len(data) > 0 and isinstance(data[0], list):
            flat = []
            for batch in data:
                flat.extend(batch)
            return flat
        return data
    raise ValueError("Unsupported item_id JSON format")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to generation.yaml")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--env_server_url", required=True)
    parser.add_argument("--item_ids_json", required=True)
    parser.add_argument("--rollout_log_dir", required=True)
    parser.add_argument("--global_step", default="1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_response_length", type=int, default=12288)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_rounds", type=int, default=15)
    parser.add_argument("--rollout_tp_size", type=int, default=1)
    parser.add_argument("--rollout_gpu_mem_util", type=float, default=0.6)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    cfg.model.path = args.model_path
    cfg.agentgym.task_name = "webshop"
    cfg.agentgym.env_addr = args.env_server_url
    cfg.agentgym.max_rounds = args.max_rounds
    cfg.data.max_prompt_length = args.max_prompt_length
    cfg.data.max_response_length = args.max_response_length
    cfg.data.batch_size = args.batch_size
    cfg.data.n_samples = 1
    cfg.rollout.n = args.rollout_n
    cfg.rollout.max_model_len = args.max_model_len
    cfg.rollout.max_tokens = args.max_tokens
    cfg.rollout.tensor_model_parallel_size = args.rollout_tp_size
    cfg.rollout.gpu_memory_utilization = args.rollout_gpu_mem_util
    cfg.rollout.rollout_log_dir = args.rollout_log_dir

    nnodes = int(os.environ.get("NNODES", cfg.trainer.nnodes))
    env_gpus = int(os.environ.get("N_GPUS_PER_NODE", "0"))
    if env_gpus <= 0:
        detected = torch.cuda.device_count()
        env_gpus = detected if detected > 0 else cfg.trainer.n_gpus_per_node
    cfg.trainer.nnodes = nnodes
    cfg.trainer.n_gpus_per_node = env_gpus

    item_ids = _load_item_ids(args.item_ids_json)
    if len(item_ids) == 0:
        raise ValueError("No item_ids found")

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    tokenizer = hf_tokenizer(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=cfg, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    env_client = init_env_client(cfg.agentgym)
    conv_user = env_client.conversation_start[0]["value"]
    conv_assistant = env_client.conversation_start[1]["value"]
    prompt_with_chat_template = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + conv_user
        + "<|im_end|>\n<|im_start|>assistant\n"
        + conv_assistant
        + "<|im_end|>"
    )

    num_items = len(item_ids)
    batch_size = args.batch_size
    num_batches = (num_items + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(num_items, start + batch_size)
        batch_item_ids = item_ids[start:end]
        if not batch_item_ids:
            continue

        prompts = [prompt_with_chat_template for _ in range(len(batch_item_ids))]
        messages = [
            [
                {"role": "user", "content": conv_user},
                {"role": "assistant", "content": conv_assistant},
            ]
            for _ in range(len(batch_item_ids))
        ]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompts,
            tokenizer=tokenizer,
            max_length=args.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        data = DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            non_tensors={"item_id": np.array(batch_item_ids, dtype=object), "raw_prompt": np.array(messages, dtype=object)},
        )
        data.meta_info["global_steps"] = str(args.global_step)
        data.meta_info["max_rounds"] = args.max_rounds

        real_batch_size = data.batch["input_ids"].shape[0]
        dp_size = wg.world_size // cfg.rollout.tensor_model_parallel_size
        if real_batch_size % dp_size != 0:
            dummy_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_size]
            data = DataProto.concat([data, dummy_data])

        _ = wg.generate_sequences(data)

    print(f"Finished pure rollout for {num_items} item_ids. Logs at: {args.rollout_log_dir}")


if __name__ == "__main__":
    main()

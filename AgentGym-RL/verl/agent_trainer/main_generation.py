# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
from collections import defaultdict
import json
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_json(os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json"))
    item_ids = dataset[config.data.prompt_key].tolist()
    # load sub category test file
    category_files = os.listdir(config.data.path)
    category_files = [f for f in category_files if not f.startswith(f"{config.agentgym.task_name}_test")]
    category_map = {}
    for category_file in category_files:
        path = os.path.join(config.data.path, category_file)
        with open(path, "r") as f:
            datas = json.load(f)
            for data in datas:
                category_map[data["item_id"]] = category_file.split(".")[0]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    # NOTE: Avoid creating an extra empty batch when `total_samples % batch_size == 0`.
    # The previous `(total_samples // batch_size) + 1` leads to `batch_item_ids=[]` on the last
    # iteration, and `tokenizer([])` crashes with `IndexError: list index out of range`.
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size
    output_lst = [[] for _ in range(config.data.n_samples)]
    raw_output_lst = [[] for _ in range(config.data.n_samples)]
    rule_output_lst = [[] for _ in range(config.data.n_samples)]
    has_raw_scores = False
    has_rule_scores = False
    env_client = init_env_client(config.agentgym)

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx: end_idx]
        if len(batch_item_ids) == 0:
            # Defensive guard in case `total_samples` changes unexpectedly or for any edge case.
            continue
        prompt_with_chat_template = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + env_client.conversation_start[1]["value"] + "<|im_end|>" for _ in range(len(batch_item_ids))]
        messages = [[{"role": "user", "content": env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": env_client.conversation_start[1]["value"]}] for _ in range(len(batch_item_ids))]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data.meta_info['global_steps'] = 'test_batch_' + str(batch_idx)
        data.meta_info['max_rounds'] = config.agentgym.max_rounds
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')

        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]

            output_lst[i].extend(output.batch['task_scores'].sum(dim=-1).tolist())
            if 'raw_task_scores' in output.batch:
                has_raw_scores = True
                raw_output_lst[i].extend(output.batch['raw_task_scores'].sum(dim=-1).tolist())
            if 'rule_task_scores' in output.batch:
                has_rule_scores = True
                rule_output_lst[i].extend(output.batch['rule_task_scores'].sum(dim=-1).tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_np = np.array(output_lst, dtype=object)
    output_np = np.transpose(output_np, axes=(1, 0))
    output_lst = output_np.tolist()

    print("============Total Task Evaluation============")
    scores = np.array(output_np, dtype=np.float32)
    print(f"Avg@{config.data.n_samples}: {np.mean(scores)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(scores, axis=-1) > 0)}")

    def _transpose_scores(score_list):
        score_np = np.array(score_list, dtype=object)
        score_np = np.transpose(score_np, axes=(1, 0))
        return np.array(score_np, dtype=np.float32)

    eps = 1e-8
    if has_raw_scores:
        raw_scores = _transpose_scores(raw_output_lst)
        print(f"RawAvg@{config.data.n_samples}: {np.mean(raw_scores)}")
        print(f"RawPass@{config.data.n_samples}: {np.mean(np.max(raw_scores, axis=-1) > 0)}")
        print(f"RawSuccessRate (reward==1.0): {np.mean(raw_scores >= 1.0 - eps)}")

    if has_rule_scores:
        rule_scores = _transpose_scores(rule_output_lst)
        print(f"RuleAvg@{config.data.n_samples}: {np.mean(rule_scores)}")
        print(f"RulePass@{config.data.n_samples}: {np.mean(np.max(rule_scores, axis=-1) > 0)}")
        print(f"RuleSuccessRate (reward>0): {np.mean(rule_scores > 0)}")

    # Verl-agent style metrics (WebShop): success_rate (reward==1.0) and rule-based test_score (10/0).
    # If rule reward is already enabled in the env, scores will be 10/0; warn and adapt.
    max_score = float(np.max(scores)) if scores.size else 0.0
    print("============Verl-Agent Style Evaluation============")
    if max_score > 1.0 + eps:
        # Likely already rule-based reward; raw continuous WebShop score is not available.
        success_rate = np.mean(scores > 0)
        test_score = np.mean(scores)
        print("Note: detected rule-based rewards (max>1). Raw WebShop task_score is not available.")
        print(f"SuccessRate (reward>0): {success_rate}")
        print(f"TestScore (rule_reward mean): {test_score}")
        print("WebshopTaskScore (raw mean): N/A")
    else:
        success_rate = np.mean(scores >= 1.0 - eps)
        test_score = success_rate * 10.0
        print(f"WebshopTaskScore (raw mean): {np.mean(scores)}")
        print(f"SuccessRate (reward==1.0): {success_rate}")
        print(f"TestScore (rule_reward 10/0 mean): {test_score}")
    print("============Sub Task Evaluation============")
    
    category_success_bucket = defaultdict(list)
    for item_id, score in zip(item_ids, output_lst):
        category = category_map[item_id]
        category_success_bucket[category].append(score)
    for category_file in category_files:
        category = category_file.split(".")[0]
        print(f"Category: {category}")
        print(f"Avg@{config.data.n_samples}: {np.mean(np.array(category_success_bucket[category]))}")
        print(f"Pass@{config.data.n_samples}: {np.mean(np.max(np.array(category_success_bucket[category]), axis=-1) > 0)}")



if __name__ == '__main__':
    main()

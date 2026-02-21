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
"""Pre-tokenized dataset with explicit loss masks."""

import json
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask


def _read_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def _truncate(seq: Iterable[int], max_length: int, truncation: str) -> list[int]:
    seq = list(seq)
    if len(seq) <= max_length:
        return seq
    if truncation == "left":
        return seq[-max_length:]
    if truncation == "right":
        return seq[:max_length]
    raise NotImplementedError(f"Unknown truncation method: {truncation}")


class PreTokenizedDataset(Dataset):
    """Dataset that reads pre-tokenized input_ids + loss_mask from JSON/JSONL."""

    def __init__(
        self,
        json_file: str,
        tokenizer: PreTrainedTokenizer,
        input_ids_key: str = "input_ids",
        loss_mask_key: str = "loss_mask",
        max_length: int = 4096,
        truncation: str = "right",
    ):
        assert truncation in ["error", "left", "right"], "Invalid truncation mode"
        self.tokenizer = tokenizer
        self.input_ids_key = input_ids_key
        self.loss_mask_key = loss_mask_key
        self.max_length = max_length
        self.truncation = truncation
        self.records = _read_json_records(json_file)

        if self.tokenizer.pad_token_id is None:
            # Keep consistent with training defaults.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        record = self.records[item]
        input_ids = record[self.input_ids_key]
        loss_mask = record[self.loss_mask_key]
        if len(input_ids) != len(loss_mask):
            raise ValueError(
                f"input_ids and loss_mask length mismatch: {len(input_ids)} vs {len(loss_mask)}"
            )

        if self.truncation == "error" and len(input_ids) > self.max_length:
            raise NotImplementedError(
                f"sequence length {len(input_ids)} exceeds max_length {self.max_length}"
            )

        input_ids = _truncate(input_ids, self.max_length, self.truncation)
        loss_mask = _truncate(loss_mask, self.max_length, self.truncation)

        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

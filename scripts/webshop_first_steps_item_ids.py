#!/usr/bin/env python3
"""Compute item_ids used in the first N training steps.

This matches the RandomSampler + drop_last behavior used by RayPPOTrainer.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import RandomSampler


def load_item_ids(train_file: str) -> list[str]:
    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [row["item_id"] for row in data]


def iter_epoch_indices(num_items: int, seed: int) -> list[int]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = RandomSampler(range(num_items), generator=gen, replacement=False)
    return list(iter(sampler))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Path to webshop_train.json")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--unique_out", default=None, help="Optional output path for unique item_ids")
    args = parser.parse_args()

    item_ids = load_item_ids(args.train_file)
    num_items = len(item_ids)
    batch_size = args.train_batch_size
    steps_needed = args.steps

    if batch_size <= 0:
        raise ValueError("train_batch_size must be > 0")
    if steps_needed <= 0:
        raise ValueError("steps must be > 0")

    all_steps: list[list[str]] = []
    remaining = steps_needed
    epoch = 0
    seed = args.seed

    while remaining > 0:
        indices = iter_epoch_indices(num_items, seed + epoch)
        # drop_last to match DataLoader(drop_last=True)
        usable = (len(indices) // batch_size) * batch_size
        indices = indices[:usable]
        step_indices = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        if len(step_indices) == 0:
            raise RuntimeError("No batches formed; check dataset size and batch size.")

        take = min(remaining, len(step_indices))
        for i in range(take):
            all_steps.append([item_ids[idx] for idx in step_indices[i]])
        remaining -= take
        epoch += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_file": args.train_file,
                "steps": args.steps,
                "train_batch_size": args.train_batch_size,
                "seed": args.seed,
                "step_item_ids": all_steps,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    if args.unique_out:
        seen = []
        seen_set = set()
        for batch in all_steps:
            for item_id in batch:
                if item_id not in seen_set:
                    seen.append(item_id)
                    seen_set.add(item_id)
        unique_path = Path(args.unique_out)
        unique_path.parent.mkdir(parents=True, exist_ok=True)
        with unique_path.open("w", encoding="utf-8") as f:
            json.dump(seen, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()

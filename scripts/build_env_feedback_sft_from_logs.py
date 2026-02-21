#!/usr/bin/env python3
"""Build pre-tokenized SFT data from rollout executer_logs.

Output JSONL lines with {input_ids, loss_mask, item_id}.
loss_mask is aligned for next-token loss (shifted left by 1).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from verl.utils import hf_tokenizer
from verl.workers.rollout.schemas import RolloutHandler, Message


def build_handler(messages, tokenizer, max_model_len, max_response_len):
    if len(messages) < 2:
        raise ValueError("Need at least 2 messages for conversation_start.")

    user0 = messages[0]["content"]
    assistant0 = messages[1]["content"]
    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + user0
        + "<|im_end|>\n<|im_start|>assistant\n"
        + assistant0
        + "<|im_end|>"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    zeros = [0] * len(input_ids)

    handler = RolloutHandler(
        messages=[Message(role=m["role"], content=m["content"]) for m in messages[:2]],
        task_name="webshop",
        item_id=0,
        score=0.0,
        done=False,
        input_ids=list(input_ids),
        prompt_ids=list(input_ids),
        response_ids=[],
        attention_mask=list(attention_mask),
        prompt_attention_mask=list(attention_mask),
        response_attention_mask=[],
        position_ids=list(position_ids),
        prompt_position_ids=list(position_ids),
        response_position_ids=[],
        loss_mask=list(zeros),
        prompt_loss_mask=list(zeros),
        response_loss_mask=[],
        env_mask=list(zeros),
        prompt_env_mask=list(zeros),
        response_env_mask=[],
        max_response_len=max_response_len,
        max_model_len=max_model_len,
    )
    return handler


def iter_log_files(executer_logs_dir: Path, max_step: Optional[int]):
    for step_dir in sorted(executer_logs_dir.glob("step*")):
        step_name = step_dir.name
        if not step_name.startswith("step"):
            continue
        try:
            step_num = int(step_name[4:])
        except ValueError:
            continue
        if max_step is not None and step_num > max_step:
            continue
        for path in sorted(step_dir.glob("*.json")):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executer_logs_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_response_len", type=int, default=32768)
    parser.add_argument(
        "--filter_overlong",
        action="store_true",
        help="Skip samples whose response length exceeds max_response_len.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logs_dir = Path(args.executer_logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(logs_dir)

    tokenizer = hf_tokenizer(args.model_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for log_path in iter_log_files(logs_dir, args.max_step):
            with log_path.open("r", encoding="utf-8") as f_in:
                records = json.load(f_in)
            for record in records:
                messages = record.get("conversations", [])
                if len(messages) < 2:
                    continue
                handler = build_handler(
                    messages,
                    tokenizer,
                    max_model_len=args.max_model_len,
                    max_response_len=args.max_response_len,
                )
                user_seen_after_start = 0
                for msg in messages[2:]:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "assistant":
                        handler.add_assistant_message(tokenizer, content)
                    elif role == "user":
                        user_seen_after_start += 1
                        mark_env = user_seen_after_start > 1
                        handler.add_user_message(tokenizer, content, mark_env_feedback=mark_env)
                    else:
                        # Skip unknown roles
                        continue
                response_len = len(handler.input_ids) - len(handler.prompt_ids)
                if args.filter_overlong and response_len > args.max_response_len:
                    skipped += 1
                    continue
                handler.truncate_output_ids()

                env_mask = handler.env_mask
                input_ids = handler.input_ids
                if len(env_mask) != len(input_ids):
                    raise RuntimeError("env_mask length mismatch")

                # Shift left so loss_mask aligns with next-token loss.
                if len(env_mask) > 0:
                    loss_mask = env_mask[1:] + [0]
                else:
                    loss_mask = env_mask

                out_record = {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "item_id": record.get("item_id"),
                }
                f_out.write(json.dumps(out_record, ensure_ascii=True) + "\n")
                written += 1
                if args.limit is not None and written >= args.limit:
                    print(f"Reached limit {args.limit}")
                    return

    print(f"Wrote {written} samples to {out_path}")
    if args.filter_overlong:
        print(
            f"Skipped {skipped} overlong samples (> {args.max_response_len} tokens)"
        )


if __name__ == "__main__":
    main()

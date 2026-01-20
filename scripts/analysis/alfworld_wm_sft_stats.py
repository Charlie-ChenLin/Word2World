#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


TASK_KEY = "Your task is to:"
USER_INFO_KEY = "# User Environment Information (Displayed to User)"
SUCCESS_MARK = "[SUCCESS]"
FAIL_MARKS = ("[FAIL]", "[FAILED]")


def extract_task(system_text: str) -> str | None:
    if TASK_KEY not in system_text:
        return None
    after = system_text.split(TASK_KEY, 1)[1]
    for sep in ("AVAILABLE ACTIONS:", "Available actions:", "AVAILABLE ACTIONS"):
        if sep in after:
            after = after.split(sep, 1)[0]
            break
    task = after.strip()
    if task.endswith("."):
        task = task[:-1].strip()
    return task or None


def extract_user_env_info(system_text: str) -> str | None:
    if USER_INFO_KEY not in system_text:
        return None
    after = system_text.split(USER_INFO_KEY, 1)[1]
    return after.strip() or None


def find_system_message(messages: list[dict]) -> str | None:
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return None


def find_last_assistant_message(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count trajectories per task for ALFWorld WM SFT data."
    )
    parser.add_argument(
        "--input",
        default="data/llama_factory/alfworld_train_with_env_54006.json",
        help="Input JSON file (default: data/llama_factory/alfworld_train_with_env_54006.json)",
    )
    parser.add_argument(
        "--agent_instruct",
        default="data/init_contexts/alfworld/agent_instruct_train.json",
        help="Agent init file used to align question text to data_idx.",
    )
    parser.add_argument(
        "--output",
        default="outputs/alfworld_wm_sft_task_stats.json",
        help="Output JSON file path (default: outputs/alfworld_wm_sft_task_stats.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    agent_instruct_path = Path(args.agent_instruct)
    agent_data = json.loads(agent_instruct_path.read_text())

    text_to_data_idx: dict[str, list[int]] = {}
    data_idx_to_text: dict[int, str] = {}
    for item in agent_data:
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue
        user_text = messages[2].get("content", "").strip()
        if not user_text:
            continue
        data_idx = int(item["data_idx"])
        text_to_data_idx.setdefault(user_text, []).append(data_idx)
        data_idx_to_text[data_idx] = user_text

    stats_by_idx: dict[int, dict[str, int]] = {}
    missing_user_env_info = 0
    no_match = 0
    ambiguous = 0
    ambiguous_examples: list[dict[str, object]] = []

    for item in data:
        messages = item.get("messages", [])
        system_text = find_system_message(messages) or ""
        user_env_info = extract_user_env_info(system_text)
        if not user_env_info:
            missing_user_env_info += 1
            continue
        data_idx_list = text_to_data_idx.get(user_env_info, [])
        if not data_idx_list:
            no_match += 1
            continue
        if len(data_idx_list) > 1:
            ambiguous += 1
            if len(ambiguous_examples) < 5:
                ambiguous_examples.append(
                    {"data_idx": data_idx_list, "question": user_env_info}
                )
            continue
        data_idx = data_idx_list[0]

        last_assistant = find_last_assistant_message(messages) or ""
        success = SUCCESS_MARK in last_assistant
        explicit_fail = any(mark in last_assistant for mark in FAIL_MARKS)

        entry = stats_by_idx.setdefault(
            data_idx, {"total": 0, "success": 0, "fail": 0, "explicit_fail": 0}
        )
        entry["total"] += 1
        if success:
            entry["success"] += 1
        else:
            entry["fail"] += 1
        if explicit_fail:
            entry["explicit_fail"] += 1

    sorted_items = sorted(stats_by_idx.items(), key=lambda kv: kv[0])
    summary = {
        "input": str(input_path),
        "agent_instruct": str(agent_instruct_path),
        "note": "fail = total - success; explicit_fail counts [FAIL]/[FAILED] markers if present",
        "missing_user_env_info": missing_user_env_info,
        "no_match_user_env_info": no_match,
        "ambiguous_user_env_info": ambiguous,
        "ambiguous_examples": ambiguous_examples,
        "data_idx_stats": [
            {
                "data_idx": data_idx,
                "question": data_idx_to_text.get(data_idx, ""),
                "total": counts["total"],
                "success": counts["success"],
                "fail": counts["fail"],
                "explicit_fail": counts["explicit_fail"],
                "success_rate": (
                    counts["success"] / counts["total"] if counts["total"] else 0.0
                ),
            }
            for data_idx, counts in sorted_items
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote stats to {out_path} (tasks={len(summary['tasks'])})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quickly scan all AlfWorld games and find ones whose reset returns an error
or misses required fields (observation / available_actions).

Usage (from repo root):
  source ../uv_alfworld/bin/activate
  python scripts/debug/scan_alfworld_games.py [--start 0] [--end 3001]

This calls the server-side wrapper directly (no HTTP), so you see the real
Python exceptions and stack traces from env reset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Iterable

# Ensure the AlfWorld wrapper is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "AgentGym/agentenv-alfworld")

from agentenv_alfworld.env_wrapper import ALFWorld_Wrapper  # type: ignore


MAPPING_FILES = [
    "AgentGym/agentenv-alfworld/configs/mappings_train.json",
    "AgentGym/agentenv-alfworld/configs/mappings_test.json",
    "AgentGym/agentenv-alfworld/configs/mappings_valid_seen.json",
    "AgentGym/agentenv-alfworld/configs/mappings_valid_unseen.json",
]


def load_game_ids(files: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for path in files:
        with open(path) as f:
            mapping = json.load(f)
        ids.extend(int(m["item_id"]) for m in mapping)
    return sorted(ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan AlfWorld games to find ones whose reset fails or misses keys",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma/space separated list of game ids to scan; overrides start/end",
    )
    parser.add_argument("--start", type=int, default=0, help="lowest game id to scan")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="highest game id to scan (inclusive); default scans to max id",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.expanduser("~/.cache/alfworld"),
        help="path to AlfWorld data (matches ALFWORLD_DATA)",
    )
    parser.add_argument(
        "--config_path",
        default="AgentGym/agentenv-alfworld/configs/base_config.yaml",
        help="path to AlfWorld base_config.yaml",
    )
    parser.add_argument(
        "--raise_on_error",
        action="store_true",
        help="when reset returns error, rerun without catch to show full traceback",
    )
    args = parser.parse_args()

    if args.ids:
        import re

        tokens = [t for t in re.split(r"[,\s]+", args.ids.strip()) if t]
        game_ids = [int(t) for t in tokens]
    else:
        game_ids = load_game_ids(MAPPING_FILES)
        if args.end is not None:
            game_ids = [gid for gid in game_ids if args.start <= gid <= args.end]
        else:
            game_ids = [gid for gid in game_ids if gid >= args.start]

    if args.ids:
        print(f"Scanning explicit ids: {len(game_ids)} items -> {game_ids}")
    else:
        print(
            f"Scanning {len(game_ids)} games from id>={args.start}"
            + (f" to {args.end}" if args.end is not None else "")
        )

    wrapper = ALFWorld_Wrapper(data_path=args.data_path, config_path=args.config_path)
    env_id = wrapper.create()["id"]

    def reset_raw(idx: int, game: int, world_type: str = "Text"):
        """Call reset logic without the try/except in env_wrapper to expose traceback."""
        wrapper._check_id(idx, True)
        wrapper.env[idx].game_files = [wrapper.games[game]]
        wrapper.env[idx].num_games = 1
        wrapper.env_init[idx] = wrapper.env[idx].init_env(batch_size=1)
        ob, info = wrapper.env_init[idx].reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        available_actions = info.get("admissible_commands", [[]])[0]
        payload = {
            "id": idx,
            "observation": ob,
            "available_actions": available_actions,
            "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
        }
        wrapper.info[idx] = {
            "world_type": world_type,
            "game": game,
            "observation": ob,
            "available_actions": available_actions,
            "done": False,
            "reward": 0,
            "deleted": False,
        }
        return payload

    bad: list[tuple[int, str]] = []
    try:
        for gid in game_ids:
            try:
                resp = wrapper.reset(env_id, gid, "Text")
                if "error" in resp:
                    msg = f"error response: {resp}"
                    if args.raise_on_error:
                        try:
                            reset_raw(env_id, gid, "Text")
                        except Exception as ee:  # noqa: BLE001
                            tb2 = traceback.format_exc()
                            msg = f"{msg} | raw traceback: {ee}\\n{tb2}"
                    bad.append((gid, msg))
                    print(f"[BAD ] game_id={gid}: {msg}", flush=True)
                elif "observation" not in resp or "available_actions" not in resp:
                    msg = f"missing keys in response: {resp}"
                    bad.append((gid, msg))
                    print(f"[BAD ] game_id={gid}: {msg}", flush=True)
                else:
                    print(f"[OK  ] game_id={gid}", flush=True)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                bad.append((gid, f"exception: {e}\n{tb}"))
                print(f"[BAD ] game_id={gid}: exception: {e}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user; partial results below...")
    finally:
        try:
            wrapper.close(env_id)
        except Exception:
            pass

    print(f"Finished. bad_count={len(bad)}")
    for gid, msg in bad:
        print(f"[BAD] game_id={gid}: {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

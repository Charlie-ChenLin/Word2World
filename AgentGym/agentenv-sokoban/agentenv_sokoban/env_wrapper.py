"""Sokoban environment wrapper with verl-agent-aligned train/val seed split."""

import os
import re
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .sokoban import SokobanEnv


def _normalize_split(split: Optional[str], default_split: str = "train") -> str:
    if split is None:
        return default_split
    split = str(split).strip().lower()
    if split in ("val", "valid", "validation", "test", "eval", "evaluation"):
        return "val"
    return "train"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_action(
    response: str,
    *,
    strict_match: bool = True,
    require_think_tag: bool = True,
) -> Tuple[int, bool, str]:
    # 0: still/invalid, 1: up, 2: down, 3: left, 4: right
    strict_action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }
    relaxed_action_pools = dict(strict_action_pools)
    relaxed_action_pools.update(
        {
            "none": 0,
            "stay": 0,
            "noop": 0,
        }
    )

    if not isinstance(response, str):
        return 0, False, "non_string_response"

    text = response.lower()

    if strict_match:
        if require_think_tag and not re.search(
            r"<think>\s*.*?\s*</think>", text, flags=re.DOTALL
        ):
            return 0, False, "missing_think_tag"

        tag_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.DOTALL)
        if not tag_match:
            return 0, False, "missing_action_tag"

        candidate = tag_match.group(1).strip()
        for key, val in strict_action_pools.items():
            if re.search(rf"\b{re.escape(key)}\b", candidate):
                return val, True, "ok"
        return 0, False, "unknown_action_keyword"

    candidates = []
    tag_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.DOTALL)
    if tag_match:
        candidates.append(tag_match.group(1).strip())

    line_match = re.search(r"action\s*:\s*(.*?)(?:\n|$)", text, flags=re.DOTALL)
    if line_match:
        candidates.append(line_match.group(1).strip())

    candidates.append(text.strip())

    for candidate in candidates:
        for key, val in relaxed_action_pools.items():
            if re.search(rf"\b{re.escape(key)}\b", candidate):
                return val, True, "ok_relaxed"

    return 0, False, "no_action_keyword"


class NotInitializedError(Exception):
    pass


class SokobanEnvServer:
    """HTTP-backed multi-session Sokoban server."""

    def __init__(self) -> None:
        self._max_id = 0
        self._lock = threading.Lock()

        self.mode = os.environ.get("SOKOBAN_MODE", "tiny_rgb_array")
        self.default_split = _normalize_split(os.environ.get("SOKOBAN_DEFAULT_SPLIT", "train"))
        self.dim_x = int(os.environ.get("SOKOBAN_DIM_X", "6"))
        self.dim_y = int(os.environ.get("SOKOBAN_DIM_Y", "6"))
        self.num_boxes = int(os.environ.get("SOKOBAN_NUM_BOXES", "1"))
        self.max_steps = int(os.environ.get("SOKOBAN_MAX_STEPS", "50"))
        self.search_depth = int(os.environ.get("SOKOBAN_SEARCH_DEPTH", "30"))
        self.seed = int(os.environ.get("SOKOBAN_SEED", "0"))
        self.strict_action_match = _env_flag("SOKOBAN_STRICT_ACTION_MATCH", True)
        self.require_think_tag = _env_flag("SOKOBAN_REQUIRE_THINK_TAG", True)
        try:
            self.invalid_action_penalty = float(
                os.environ.get("SOKOBAN_INVALID_ACTION_PENALTY", "0.1")
            )
        except ValueError:
            self.invalid_action_penalty = 0.1
        self.invalid_action_penalty = max(0.0, float(self.invalid_action_penalty))

        self._rng = np.random.default_rng(self.seed)
        self.env: Dict[int, SokobanEnv] = {}
        self.meta: Dict[int, Dict[str, Any]] = {}

    def _build_env(self) -> SokobanEnv:
        return SokobanEnv(
            mode=self.mode,
            dim_room=(self.dim_x, self.dim_y),
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth,
        )

    def _check_env_idx(self, env_idx: int) -> None:
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _sample_seed(self, split: str) -> int:
        # Match verl-agent semantics:
        # train: randint(0, 2**16 - 1), val: randint(2**16, 2**32 - 1)
        if split == "val":
            return int(self._rng.integers(2**16, 2**32 - 1))
        return int(self._rng.integers(0, 2**16 - 1))

    def create(self, _: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1
            self.env[env_idx] = self._build_env()
            self.meta[env_idx] = {
                "split": self.default_split,
                "seed": None,
                "observation": "",
                "score": 0.0,
                "reward": 0.0,
                "done": False,
                "won": False,
                "invalid_action_count": 0,
                "invalid_action_penalty_total": 0.0,
            }
        return env_idx

    def reset(
        self,
        env_idx: int,
        item_id: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        del item_id  # item_id is not used in verl-agent sokoban split logic.
        with self._lock:
            self._check_env_idx(env_idx)
            resolved_split = _normalize_split(split, self.default_split)
            resolved_seed = int(seed) if seed is not None else self._sample_seed(resolved_split)

            obs, info = self.env[env_idx].reset(seed=resolved_seed)
            serialized_obs = obs if isinstance(obs, str) else str(obs)
            info = dict(info or {})
            info.setdefault("won", False)
            info["seed"] = resolved_seed
            info["split"] = resolved_split

            self.meta[env_idx] = {
                "split": resolved_split,
                "seed": resolved_seed,
                "observation": serialized_obs,
                "score": 0.0,
                "reward": 0.0,
                "done": False,
                "won": bool(info.get("won", False)),
                "invalid_action_count": 0,
                "invalid_action_penalty_total": 0.0,
            }

            return {
                "observation": serialized_obs,
                "reward": 0.0,
                "score": 0.0,
                "done": False,
                "info": info,
            }

    def step(self, env_idx: int, response: str) -> Dict[str, Any]:
        with self._lock:
            self._check_env_idx(env_idx)

            action, is_action_valid, parse_reason = _parse_action(
                response,
                strict_match=self.strict_action_match,
                require_think_tag=self.require_think_tag,
            )
            obs, reward, done, info = self.env[env_idx].step(action)
            serialized_obs = obs if isinstance(obs, str) else str(obs)
            info = dict(info or {})
            info["action_idx"] = action
            info["is_action_valid"] = bool(is_action_valid)
            info["action_parse_reason"] = parse_reason

            penalty_applied = 0.0
            if (not is_action_valid) and self.invalid_action_penalty > 0:
                penalty_applied = float(self.invalid_action_penalty)
                reward = float(reward) - penalty_applied
            else:
                reward = float(reward)

            invalid_action_count = int(self.meta[env_idx].get("invalid_action_count", 0))
            invalid_action_penalty_total = float(
                self.meta[env_idx].get("invalid_action_penalty_total", 0.0)
            )
            if not is_action_valid:
                invalid_action_count += 1
                invalid_action_penalty_total += penalty_applied

            info["invalid_action_penalty"] = penalty_applied
            info["invalid_action_count"] = invalid_action_count
            info["invalid_action_penalty_total"] = invalid_action_penalty_total

            score = float(self.meta[env_idx].get("score", 0.0)) + float(reward)
            info["seed"] = self.meta[env_idx].get("seed")
            info["split"] = self.meta[env_idx].get("split")

            self.meta[env_idx].update(
                {
                    "observation": serialized_obs,
                    "reward": float(reward),
                    "score": float(score),
                    "done": bool(done),
                    "won": bool(info.get("won", False)),
                    "invalid_action_count": invalid_action_count,
                    "invalid_action_penalty_total": invalid_action_penalty_total,
                }
            )

            return {
                "observation": serialized_obs,
                "reward": float(reward),
                "score": float(score),
                "done": bool(done),
                "info": info,
            }

    def observation(self, env_idx: int) -> str:
        with self._lock:
            self._check_env_idx(env_idx)
            return str(self.meta[env_idx].get("observation", ""))

    def close(self, env_idx: int) -> bool:
        with self._lock:
            if env_idx in self.env:
                del self.env[env_idx]
            if env_idx in self.meta:
                del self.meta[env_idx]
        return True


sokoban_env_server = SokobanEnvServer()

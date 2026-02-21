"""
Rule-based reward conversion for WebShop.

This mirrors the verl-agent WebShop wrapper: if an episode is done and the
raw WebShop reward equals 1.0, emit a fixed success reward (default: 10.0),
otherwise emit a fixed failure reward (default: 0.0).
"""

from typing import Optional, Tuple
import os


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def rule_reward_enabled() -> bool:
    return _is_truthy(os.getenv("WEBSHOP_RULE_REWARD"))


def rule_reward_values() -> Tuple[float, float]:
    success = os.getenv("WEBSHOP_RULE_REWARD_SUCCESS", "10")
    failure = os.getenv("WEBSHOP_RULE_REWARD_FAIL", "0")
    try:
        return float(success), float(failure)
    except ValueError:
        return 10.0, 0.0


def apply_rule_reward(
    reward: float,
    done: bool,
    success_reward: float,
    failure_reward: float,
) -> Tuple[float, bool]:
    if done and reward == 1.0:
        return success_reward, True
    return failure_reward, False

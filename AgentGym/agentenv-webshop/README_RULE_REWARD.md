# WebShop rule-based reward (0/1 -> fixed reward)

This repo's WebShop environment (AgentGym) uses the native WebShop reward from
`web_agent_site.engine.goal.get_reward`, which is a dense score in `[0, 1]`.
In `projects/multi-turn-rl/verl-agent`, the WebShop wrapper converts that into a
rule-based terminal reward: **success -> 10, failure -> 0**.

## What I changed

- Added `agentenv_webshop/rule_reward.py` to mirror the verl-agent rule-based
  conversion.
- Hooked it into `agentenv_webshop/environment.py` so the conversion is applied
  **only when enabled**.

## How to enable

Set the environment variable before starting the WebShop server:

```
export WEBSHOP_RULE_REWARD=1
```

Optional overrides:

```
export WEBSHOP_RULE_REWARD_SUCCESS=10
export WEBSHOP_RULE_REWARD_FAIL=0
```

If `WEBSHOP_RULE_REWARD` is unset, the behavior is unchanged and the native
WebShop dense reward is returned.

## Where it maps to verl-agent

The logic mirrors:

- `projects/multi-turn-rl/verl-agent/agent_system/environments/env_package/webshop/envs.py`

Specifically:

```
if done and reward == 1.0:
    reward = 10.0
else:
    reward = 0
```

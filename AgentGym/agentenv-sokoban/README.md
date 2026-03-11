# AgentEnv Sokoban

Sokoban environment server for Word2World AgentGym-RL.

## Install

```bash
cd AgentGym/agentenv-sokoban
pip install -e .
```

Or use the project's `uv + apptainer` setup script:

```bash
bash scripts/env_setup/uv_sokoban_apptainer.sh
```

## Launch

```bash
sokoban --host 0.0.0.0 --port 37001
```

## Split behavior

This implementation follows `verl-agent` seed split semantics:

- `train`: seed sampled from `[0, 2^16 - 1)`
- `val/test`: seed sampled from `[2^16, 2^32 - 1)`

Set default split by env var:

```bash
export SOKOBAN_DEFAULT_SPLIT=train   # or val
```

## Action Parsing And Invalid Penalty

By default, the server now uses a stricter action matching rule aligned with
`verl-agent` Sokoban prompts:

- Require `<action>...</action>` to parse action.
- Require `<think>...</think>` for the action to be considered valid.
- Apply extra invalid-action penalty on top of env reward.

Configuration:

```bash
export SOKOBAN_STRICT_ACTION_MATCH=1      # 1: strict tag-based parsing, 0: relaxed fallback parsing
export SOKOBAN_REQUIRE_THINK_TAG=1        # 1: missing <think>...</think> counts as invalid
export SOKOBAN_INVALID_ACTION_PENALTY=0.1 # extra penalty per invalid action
```

To disable the additional invalid-action penalty:

```bash
export SOKOBAN_INVALID_ACTION_PENALTY=0
```

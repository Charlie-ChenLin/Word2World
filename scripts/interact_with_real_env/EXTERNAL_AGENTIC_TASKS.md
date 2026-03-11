# Integrating External Agentic Benchmarks into Word2World

This note explains how to integrate external tasks such as `GAIA`, `tau-bench`, `BFCL`, and `AppWorld` into the current Word2World evaluation loop without rewriting the core runner.

## 1. Current Integration Boundary in Word2World

The real-environment evaluation path is:

1. `scripts/interact_with_real_env/run_*.sh`
2. `scripts/interact_with_real_env/run.py`
3. `agentenv.controller.Evaluator`
4. `Task(BaseTask) -> EnvClient(BaseEnvClient) -> HTTP env server`

So the stable extension point is:

- implement or provide a `Task` class (internal or external import path)
- expose an HTTP environment adapter (proxy) that the `EnvClient` can call

## 2. What Was Added in This Repo

To make external tasks pluggable:

1. `run.py` now supports:
   - `--task_class_path` for dynamic task class loading
   - `--task_client_args_json` for extra env client args
2. `run_openai.sh`, `run_azure.sh`, `run_vllm.sh` now support:
   - `ENV_START_CMD` (custom env launcher)
   - `ENV_SERVER_BASE` / `ENV_PORT`
   - `INFERENCE_FILE` override
   - optional forwarding of `TASK_CLASS_PATH` and `TASK_CLIENT_ARGS_JSON`
3. Added generic task skeleton:
   - `agentenv.envs.external_http:ExternalHTTPTask`
   - a parameterized HTTP client for proxy-style integrations

## 3. Recommended Mapping for GAIA / tau-bench / BFCL / AppWorld

Use two integration modes.

1. `Interactive mode` (best for `tau-bench`, `AppWorld`, some GAIA setups):
   - build a proxy server exposing:
     - `POST /create`
     - `POST /reset`
     - `POST /step`
     - `GET /observation`
     - optional `POST /close`
   - run with `ExternalHTTPTask` or a custom `Task` class

2. `Static-eval mode` (best for `BFCL`, some GAIA subsets):
   - no multi-step environment needed
   - either:
     - wrap each sample as a one-step env in the proxy, or
     - keep a dedicated evaluator script and only reuse model serving utilities

## 4. Benchmark-Specific Notes

### GAIA

- Nature: open-ended, multi-hop tasks, often with files/tools/web context.
- Suggestion:
  - start with proxy-based integration (`Interactive mode`) if you want unified success-rate reporting.
  - use a custom task class if you need rich observation formatting (attachments/tool traces).
- Key point:
  - ensure `reset(idx)` maps deterministically to a GAIA sample.

### tau-bench

- Nature: conversational, tool/API interaction and policy constraints.
- Suggestion:
  - map simulator state to `observation`
  - map tool/API call text to `action`
  - return `reward in {0,1}` and `done`
- Usually the easiest to fit the existing AgentGym-style loop.

### BFCL

- Nature: function-calling correctness benchmark (often single-turn or short-turn).
- Suggestion:
  - fastest path: one-step proxy env (`done=True` after one prediction).
  - alternative: standalone evaluator and only share model invocation utilities.

### AppWorld

- Nature: long-horizon app/API/tool workflows; closer to embodied tool use.
- Suggestion:
  - implement proxy server with explicit state snapshots.
  - if actions are structured (JSON/function call), prefer a custom adapter parser in a custom `Task` class.

## 5. Minimal Example: Running an External Task via Proxy

Assume your proxy server serves benchmark `tau_bench` and follows the HTTP contract above.

```bash
export INFERENCE_FILE="data/eval/tau_bench_test.json"
export ENV_PORT=36031
export ENV_START_CMD="source ~/uv_envs/uv_tau_bench/bin/activate && python -m tau_proxy.server --host 0.0.0.0 --port ${ENV_PORT}"

export TASK_CLASS_PATH="agentenv.envs.external_http:ExternalHTTPTask"
export TASK_CLIENT_ARGS_JSON='{
  "create_path": "create",
  "reset_path": "reset",
  "step_path": "step",
  "observation_path": "observation",
  "close_path": "close",
  "id_body_key": "id",
  "id_query_key": "id",
  "reset_index_key": "idx",
  "action_key": "action",
  "observation_key": "observation",
  "available_actions_key": "available_actions",
  "reward_key": "reward",
  "done_key": "done",
  "parse_action_with_react": true
}'

bash scripts/interact_with_real_env/run_openai.sh \
  tau_bench 0 "$OPENAI_API_KEY" "$OPENAI_BASE_URL" \
  gpt-4o 64 40 -1 test outputs
```

## 6. Data File Contract (`INFERENCE_FILE`)

`run.py` now supports multiple index schemas, including:

1. list of `{"item_id": "..._<number>"}` (existing format)
2. list of `{"data_idx": <int>}` / `{"idx": <int>}` / `{"index": <int>}`
3. fallback to row index when IDs are non-numeric

For external tasks, simplest is:

```json
[
  {"idx": 0},
  {"idx": 1},
  {"idx": 2}
]
```

## 7. Practical Rollout Plan

1. First connect `tau-bench` with `ExternalHTTPTask` as a reference integration.
2. Reuse the same proxy scaffolding for `AppWorld`.
3. For `BFCL`, decide whether to keep one-step proxy (unified metrics) or standalone scorer.
4. For `GAIA`, start with a constrained subset (no heavy external tools), then expand tool coverage.

This sequence minimizes risk while keeping your training/eval pipeline unified.

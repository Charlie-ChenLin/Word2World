from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class ExternalHTTPEnvClient(BaseEnvClient):
    """A generic AgentGym-compatible env client for HTTP-backed external tasks.

    This is useful when integrating third-party benchmarks by adding a thin
    proxy server that exposes create/reset/step/observation endpoints.
    """

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are an autonomous agent interacting with an environment.\n"
                    "Respond in ReAct format:\n"
                    "Thought:\n<brief reasoning>\n\n"
                    "Action:\n<one action>"
                ),
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "Understood. I will output one action per turn.",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        create_path: str = "create",
        reset_path: str = "reset",
        step_path: str = "step",
        observation_path: str = "observation",
        close_path: str = "close",
        id_query_key: str = "id",
        id_body_key: str = "id",
        create_payload: dict[str, Any] | None = None,
        reset_index_key: str = "idx",
        action_key: str = "action",
        observation_key: str = "observation",
        available_actions_key: str = "available_actions",
        reward_key: str = "reward",
        done_key: str = "done",
        parse_action_with_react: bool = True,
        append_available_actions: bool = True,
        conversation_start_prompt: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        self.create_path = create_path
        self.reset_path = reset_path
        self.step_path = step_path
        self.observation_path = observation_path
        self.close_path = close_path
        self.id_query_key = id_query_key
        self.id_body_key = id_body_key
        self.reset_index_key = reset_index_key
        self.action_key = action_key
        self.observation_key = observation_key
        self.available_actions_key = available_actions_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.parse_action_with_react = parse_action_with_react
        self.append_available_actions = append_available_actions

        if conversation_start_prompt:
            self.conversation_start = (
                ConversationMessage(
                    {
                        "from": "human",
                        "loss": None,
                        "value": conversation_start_prompt,
                    }
                ),
                ConversationMessage(
                    {
                        "from": "gpt",
                        "loss": False,
                        "value": "Understood.",
                    }
                ),
            )

        payload = create_payload or {}
        ok = requests.post(
            f"{self.env_server_base}/{self.create_path}",
            timeout=self.timeout,
            json=payload,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = self._parse_env_id(ok.json())
        self.info: dict[str, Any] = {}

    def __len__(self):
        return self.data_len

    @staticmethod
    def _parse_env_id(payload: Any):
        if isinstance(payload, dict):
            for key in ("id", "env_id", "env_idx"):
                if key in payload:
                    return payload[key]
            raise RequestException(f"Invalid /create payload (missing env id): {payload}")
        # Some servers return plain scalar IDs.
        if isinstance(payload, (int, str)):
            return payload
        raise RequestException(f"Invalid /create payload: {payload}")

    def _post(self, path: str, data: dict[str, Any] | None = None) -> Any:
        payload = dict(data or {})
        payload[self.id_body_key] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=payload,
            timeout=self.timeout,
        )
        assert res.status_code == 200, f"POST {path} failed: {res.text}"
        return res.json()

    def _get(self, path: str) -> Any:
        res = requests.get(
            f"{self.env_server_base}/{path}",
            params={self.id_query_key: self.env_id},
            timeout=self.timeout,
        )
        assert res.status_code == 200, f"GET {path} failed: {res.text}"
        return res.json()

    def _extract_observation(self, response: Any) -> str:
        if isinstance(response, dict):
            return str(response.get(self.observation_key, response))
        return str(response)

    def _extract_actions(self, response: Any) -> list[str]:
        if not isinstance(response, dict):
            return []
        actions = response.get(self.available_actions_key, []) or []
        if isinstance(actions, list):
            return [str(a) for a in actions]
        return []

    def observe(self) -> str:
        response = self._get(self.observation_path)
        observation = self._extract_observation(response)
        actions = self._extract_actions(response)
        if self.append_available_actions and actions:
            return f"{observation}\nAVAILABLE ACTIONS: {', '.join(actions)}"
        return observation

    def _parse_action(self, action: str) -> str:
        if not self.parse_action_with_react:
            return action
        # Lightweight ReAct parser: keep text after the last "Action:".
        if "Action:" in action:
            return action.rsplit("Action:", 1)[-1].strip()
        return action.strip()

    def step(self, action: str) -> StepOutput:
        parsed_action = self._parse_action(action)
        if not parsed_action:
            return StepOutput(
                state="Invalid Action: empty action.",
                reward=0.0,
                done=False,
            )

        response = self._post(self.step_path, {self.action_key: parsed_action})
        self.info = response if isinstance(response, dict) else {"raw_response": response}

        if isinstance(response, dict):
            state = self._extract_observation(response)
            reward = float(response.get(self.reward_key, 0.0) or 0.0)
            done_raw = response.get(self.done_key, response.get("terminated", False))
            done = bool(done_raw)
        else:
            state, reward, done = str(response), 0.0, False

        return StepOutput(
            state=state,
            reward=reward,
            done=done,
        )

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post(self.reset_path, {self.reset_index_key: idx})
        if isinstance(response, dict):
            self.info = response
            return response
        self.info = {"raw_response": response}
        return {"raw_response": response}

    def close(self):
        if not self.close_path:
            return {"closed": False}
        try:
            return self._post(self.close_path, {})
        except Exception:
            return {"closed": False}


class ExternalHTTPTask(BaseTask):
    env_client_cls = ExternalHTTPEnvClient
    env_name = "ExternalHTTP"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        *args,
        n_clients: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)

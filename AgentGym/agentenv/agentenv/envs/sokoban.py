from typing import Any, Dict, Mapping, Optional

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class SokobanEnvClient(BaseEnvClient):
    """Sokoban HTTP client with verl-agent-aligned train/val split mode."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are an expert agent operating in the Sokoban environment. "
                    "Your goal is to push all boxes onto all targets. "
                    "Symbols: wall=#, floor=_, target=O, box=X, player=P, box_on_target=√, player_on_target=S. "
                    "First reason in <think>...</think>, then output exactly one action in "
                    "<action>up/down/left/right/still</action>."
                ),
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I will reason first and then output one action tag.",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        split: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self._split = self._normalize_split(split)

        create_body = {"id": 0}
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=create_body,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        payload = ok.json()
        if isinstance(payload, dict):
            self.env_id = payload.get("env_idx", payload.get("id"))
        else:
            self.env_id = payload

        if self.env_id is None:
            raise RuntimeError(f"Invalid /create payload: {payload}")

    def __len__(self):
        return self.data_len

    @staticmethod
    def _normalize_split(split: Optional[str]) -> str:
        if split is None:
            return "train"
        split = str(split).strip().lower()
        if split in ("val", "valid", "validation", "test", "eval", "evaluation"):
            return "val"
        return "train"

    def set_validation_mode(self, is_validate: bool) -> None:
        self._split = "val" if bool(is_validate) else "train"

    def set_rollout_split(self, split: str) -> None:
        self._split = self._normalize_split(split)

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        response = self._get("observation")
        if isinstance(response, dict):
            return str(response.get("observation", ""))
        return str(response)

    def step(self, action: str) -> StepOutput:
        response = self._post("step", {"action": action})
        # Keep the same convention as other env clients: use cumulative score as task reward.
        score = response.get("score", response.get("reward", 0.0))
        return StepOutput(
            state=response.get("observation", ""),
            reward=float(score),
            done=bool(response.get("done", False)),
            info=response.get("info", None),
        )

    def reset(self, id: int = 0) -> Dict[str, Any]:
        return self._post("reset", {"id": id, "split": self._split})

    def close(self):
        return self._post("close", {})


class SokobanTask(BaseTask):
    env_client_cls = SokobanEnvClient
    env_name = "Sokoban"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        *args,
        n_clients: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)

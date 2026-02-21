# Environment Change Log

## 2026-02-04
- WebShop env (`~/uv_envs/uv_webshop`): downgraded `tokenizers` to `0.21.4` (from `0.22.2`) to satisfy `transformers` requirement used by `pyserini` and avoid WebShop env server import errors.
- WebShop env (`~/uv_envs/uv_webshop`): `huggingface_hub` changed to `0.36.1` (from `1.3.7`) as a dependency of the tokenizers downgrade.
- Command: `python -m pip install "tokenizers>=0.21,<0.22"`

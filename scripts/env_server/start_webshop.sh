source "$HOME/uv_envs/uv_webshop/bin/activate"
WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"
webshop --host 0.0.0.0 --port "$WEBSHOP_PORT"

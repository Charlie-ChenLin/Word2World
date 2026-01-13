import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import light_hf_proxy  # noqa: F401  # optional lightweight HF proxy
    _LIGHT_HF_PROXY_READY = True
except ImportError:
    _LIGHT_HF_PROXY_READY = False


# Default to mainland China mirror to speed up HF access
HF_MIRROR = "https://hf-mirror.com"


def configure_hf_endpoint(endpoint: str = HF_MIRROR) -> None:
    """Force huggingface_hub to use the given base URL (mirror)."""
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_HUB_BASE_URL"] = endpoint
    print(f"Using Hugging Face mirror: {endpoint}")
    if _LIGHT_HF_PROXY_READY:
        print("light_hf_proxy active (import succeeded).")
    else:
        print("light_hf_proxy not installed; continuing without it.")


def ensure_huggingface_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`."
        ) from exc


def download_agent_eval(output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "X1AOX1A/LLMasWorldModels"
    print(f"Downloading dataset '{repo_id}' (repo_type=dataset, revision=main)...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        local_files_only=False,
    )

    src_root = Path(snapshot_path)
    print(f"Snapshot downloaded to cache: {src_root}")
    print(f"Copying files into: {output_dir}")

    # Copy all files while preserving relative paths
    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dest_root = output_dir if rel == "." else output_dir / rel
        Path(dest_root).mkdir(parents=True, exist_ok=True)
        for name in files:
            src_file = Path(root) / name
            dest_file = Path(dest_root) / name
            # Overwrite existing files if present
            shutil.copy2(src_file, dest_file)



def main(argv=None):
    parser = argparse.ArgumentParser(description="Download AgentGym/AgentEval dataset into a folder.")
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Path to output directory (default: data)",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=HF_MIRROR,
        help="Hugging Face base URL to use (default: mainland mirror)",
    )
    args = parser.parse_args(argv)

    ensure_huggingface_hub()
    # Try user-specified endpoint first, then fall back to official if it fails
    endpoints = [args.hf_endpoint]
    if "https://huggingface.co" not in endpoints:
        endpoints.append("https://huggingface.co")

    last_err: Exception | None = None
    output_dir = Path(args.output_dir)
    for ep in endpoints:
        configure_hf_endpoint(ep)
        try:
            download_agent_eval(output_dir)
            break
        except Exception as exc:  # broad catch to allow graceful fallback
            last_err = exc
            print(f"Download failed via {ep}: {exc}\nTrying next endpoint...")
    else:
        raise SystemExit(last_err)

    # rm -rf ~/.cache/alfworld
    # unzip data/alfworld.zip -d ~/.cache
    os.system("rm -rf ~/.cache/alfworld")
    os.system("unzip -o data/alfworld.zip -d ./.cache")
    os.system("""ln -sfn "$PWD/.cache/alfworld" ~/.cache/alfworld""")
    os.system("ls -ld ~/.cache/alfworld")
    # # unzip data/textworld.zip
    os.system("unzip -o data/textworld.zip -d data/textworld/")
    # # unzip data/webshop.zip
    os.system("unzip -o data/webshop.zip -d AgentGym/agentenv-webshop/webshop/")
    # # unzip data/webshop_index.zip
    os.system("unzip -o data/webshop_index.zip -d AgentGym/agentenv-webshop/webshop/")


if __name__ == "__main__":
    sys.exit(main())

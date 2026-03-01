#!/usr/bin/env python3
"""
Debug utility: verify strict-exact projection algebra used by the actor projection update.

This script checks equivalence between two mathematically exact forms:
  legacy  : g_other + g_pg + alpha * g_pg
  reordered: g_other + (1 + alpha) * g_pg

It is a lightweight local sanity check and does not require the full training stack.
"""

import argparse

import torch


def _parse_args():
    parser = argparse.ArgumentParser(description="Check exact projection equivalence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-tensors", type=int, default=16, help="Number of synthetic parameter tensors.")
    parser.add_argument("--min-size", type=int, default=1024, help="Minimum elements per tensor.")
    parser.add_argument("--max-size", type=int, default=16384, help="Maximum elements per tensor.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)

    dtype = _dtype_from_name(args.dtype)
    # Keep accumulation in fp32 to mirror training code's projection stats.
    pg_norm_sq = torch.zeros((), dtype=torch.float32)
    dot = torch.zeros((), dtype=torch.float32)

    g_pg_list = []
    g_env_list = []
    g_other_list = []
    for _ in range(args.num_tensors):
        size = torch.randint(low=args.min_size, high=args.max_size + 1, size=(1,)).item()
        g_pg = torch.randn(size, dtype=dtype)
        g_env = torch.randn(size, dtype=dtype)
        g_other = torch.randn(size, dtype=dtype)
        g_pg_list.append(g_pg)
        g_env_list.append(g_env)
        g_other_list.append(g_other)

        g_pg_fp32 = g_pg.float()
        g_env_fp32 = g_env.float()
        pg_norm_sq.add_(torch.sum(g_pg_fp32 * g_pg_fp32))
        dot.add_(torch.sum(g_env_fp32 * g_pg_fp32))

    alpha = dot / pg_norm_sq.clamp_min(args.eps)

    legacy = []
    reordered = []
    for g_pg, g_other in zip(g_pg_list, g_other_list):
        legacy.append(g_other + g_pg + g_pg * alpha.to(dtype=g_pg.dtype))
        reordered.append(g_other + g_pg * (1.0 + alpha).to(dtype=g_pg.dtype))

    all_ok = True
    max_abs = 0.0
    for g_legacy, g_reordered in zip(legacy, reordered):
        diff = (g_legacy - g_reordered).abs()
        max_abs = max(max_abs, diff.max().item())
        if not torch.allclose(g_legacy, g_reordered, rtol=args.rtol, atol=args.atol):
            all_ok = False
            break

    print(
        f"alpha={alpha.item():.8f} "
        f"pg_norm_sq={pg_norm_sq.item():.8f} "
        f"dot={dot.item():.8f} "
        f"max_abs_diff={max_abs:.8e} "
        f"allclose={all_ok}"
    )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

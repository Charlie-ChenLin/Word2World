"""Entrypoint for the Sokoban agent environment server."""

import argparse

import uvicorn


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=37001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run("agentenv_sokoban:app", host=args.host, port=args.port)

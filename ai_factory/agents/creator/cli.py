#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from ai_factory.agents.creator import parse_recipe_yaml, generate_files_from_recipe, persist_files


def main():
    parser = argparse.ArgumentParser(description="Deterministic Creator - generate agent from YAML recipe")
    parser.add_argument("--recipe", required=True, help="Path to recipe YAML file")
    parser.add_argument("--dest", choices=["local", "blob"], default=os.environ.get("CREATOR_DEST", "local"))
    parser.add_argument("--out", default=os.environ.get("CREATOR_OUT_BASE", "ai"), help="Local output base directory")
    parser.add_argument("--container", default=os.environ.get("BLOB_CONTAINER", "agents"), help="Azure Blob container name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present")
    args = parser.parse_args()

    with open(args.recipe, "r", encoding="utf-8") as f:
        content = f.read()
    # Keep original YAML as-is, but validation uses parsed model
    recipe = parse_recipe_yaml(content)

    files = generate_files_from_recipe(recipe, original_yaml=content)

    dest, paths = persist_files(files, dest=args.dest, out_base=args.out, container=args.container, overwrite=args.overwrite)

    print(f"status=ok name={recipe.name} dest={dest}")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()



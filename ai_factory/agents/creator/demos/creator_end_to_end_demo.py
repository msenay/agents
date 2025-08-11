#!/usr/bin/env python3
"""
Creator End-to-End Demo (LLM-free)

This demo shows two ways to use the deterministic Creator:
  1) Direct invoke: pass recipe YAML/URL as input, get output paths
  2) Redis Pub/Sub: publish a message to creator:recipes:in and read result from creator:recipes:out

Environment (for Blob):
  - AZURE_STORAGE_CONNECTION_STRING  (or AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY)
  - BLOB_CONTAINER=agents (default)

Environment (for Redis Pub/Sub demo):
  - REDIS_URL=redis://:password@localhost:6379/0
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv, find_dotenv

from ai_factory.agents.creator.agent import CreatorAgent

try:
    import redis  # type: ignore
except Exception:
    redis = None


RECIPE_YAML = """
name: testAgent
description: Prints test
system_prompt: |
  You are an assistant that says test.
tools: []
memory:
  enabled: false
azure:
  route: /api/test
  auth_level: function
"""


def demo_direct_invoke():
    print("\n=== 1) Direct invoke demo ===")
    agent = CreatorAgent(enable_pubsub=False)
    payload = {
        "recipe_yaml": RECIPE_YAML,
        # Change to "blob" to upload to Azure Blob (requires env)
        "dest": "local",
        # Local base output directory
        "path": os.environ.get("CREATOR_OUT_BASE", "ai"),
        "overwrite": True,
    }
    result = agent.invoke(json.dumps(payload))
    print("Creator result:", result["messages"][-1].content)


def demo_pubsub():
    if not redis:
        print("\n(Skipping pub/sub demo: 'redis' package not installed)")
        return

    print("\n=== 2) Redis Pub/Sub demo ===")
    if not os.environ.get("REDIS_URL"):
        print("❌ REDIS_URL not set. Example: export REDIS_URL=redis://:password@localhost:6379/0")
        return

    # Start agent with pubsub enabled
    agent = CreatorAgent(enable_pubsub=True)

    client = redis.from_url(os.environ["REDIS_URL"])  # type: ignore
    out = client.pubsub()
    out.subscribe("creator:recipes:out")

    payload = {
        "recipe_yaml": RECIPE_YAML,
        "dest": os.environ.get("CREATOR_DEST", "local"),
        "path": os.environ.get("CREATOR_OUT_BASE", "ai"),
        "container": os.environ.get("BLOB_CONTAINER", "agents"),
        "overwrite": True,
    }
    client.publish("creator:recipes:in", json.dumps(payload))
    print("Published recipe to creator:recipes:in")

    # Wait for result
    start = time.time()
    while time.time() - start < 10:
        msg = out.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if not msg:
            continue
        data = msg.get("data")
        text = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
        try:
            obj = json.loads(text)
        except Exception:
            obj = {"raw": text}
        print("Received on creator:recipes:out:", json.dumps(obj, ensure_ascii=False))
        break


def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)
    demo_direct_invoke()
    demo_pubsub()


if __name__ == "__main__":
    main()



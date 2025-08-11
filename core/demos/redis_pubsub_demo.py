#!/usr/bin/env python3
"""
Redis Pub/Sub Demo for CoreAgent

Starts a CoreAgent with Redis pub/sub enabled. Subscribes to input channels and
publishes outputs to an output channel. Use redis-cli to publish messages:

  redis-cli -u $REDIS_URL PUBLISH agent:in '{"input":"Hello","config":{"configurable":{"thread_id":"demo"}}}'

Environment:
  - REDIS_URL=redis://:password@localhost:6379/0
  - (optional) OPENAI_API_KEY or Azure OpenAI envs to see real LLM responses
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.core_agent import CoreAgent
from core.config import AgentConfig
from core.utils.env import load_env_if_exists

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


def create_model():
    key = os.environ.get("OPENAI_API_KEY")
    if key and ChatOpenAI:
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
    return None


def main():
    load_env_if_exists()
    if not os.environ.get("REDIS_URL"):
        print("❌ REDIS_URL not set. Example: export REDIS_URL=redis://:redis_password@localhost:6379/0")
        return

    config = AgentConfig(
        name="PubSubAgent",
        model=create_model(),
        system_prompt="You are a helpful assistant.",
        enable_pubsub=True,
        pubsub_sub_channels=["agent:in"],
        pubsub_pub_channel="agent:out",
        enable_memory=False,
    )

    agent = CoreAgent(config)
    print("✅ Pub/Sub agent started. Listening on 'agent:in' and publishing to 'agent:out'")
    print("💬 Try: redis-cli -u $REDIS_URL PUBLISH agent:in 'Hello from redis' (or JSON payload)")

    # Keep the demo alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")


if __name__ == "__main__":
    main()




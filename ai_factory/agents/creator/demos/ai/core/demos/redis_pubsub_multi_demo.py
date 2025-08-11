#!/usr/bin/env python3
"""
Redis Pub/Sub Multi-Agent Demo for CoreAgent

Spins up multiple CoreAgents, each with its own Redis input/output channels.
Also starts:
  - A global router: listens on 'agent:in' and routes messages to agents by keyword
  - An output printer: subscribes to all agents' out channels and prints results

Environment:
  - REDIS_URL=redis://:password@localhost:6379/0
  - (optional) OPENAI_API_KEY or Azure OpenAI envs

Try:
  - Publish globally (router decides target):
      redis-cli -u $REDIS_URL PUBLISH agent:in '{"input":"billing issue about invoice"}'
  - Publish to a specific agent:
      redis-cli -u $REDIS_URL PUBLISH agent:sales:in 'need a discount quote'
"""

import os
import threading
import json
import time

from ai_factory.agents.core.core_agent import CoreAgent
from ai_factory.agents.core.config import AgentConfig
from dotenv import load_dotenv, find_dotenv
import redis as redis_lib
from langchain_openai import ChatOpenAI, AzureChatOpenAI



def create_model():
    # Try Azure OpenAI first if creds exist
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if endpoint and api_key:
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt4o")
        print("🟦 Using Azure OpenAI model:", deployment)
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0
        )

    # Fallback: OpenAI if OPENAI_API_KEY is present
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print("🟩 Using OpenAI model:", model_name)
        return ChatOpenAI(model=model_name, temperature=0)

    print("⚠️ No LLM credentials found; running without model")
    return None


def start_output_printer(redis_url: str, out_channels: list[str]) -> threading.Thread:
    def loop():
        client = redis_lib.from_url(redis_url)
        ps = client.pubsub()
        ps.subscribe(*out_channels)
        print(f"🖨️  Output printer subscribed: {out_channels}")
        for item in ps.listen():
            if item and item.get("type") == "message":
                data = item.get("data")
                try:
                    text = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
                    obj = json.loads(text)
                except Exception:
                    obj = {"raw": str(data)}
                channel = item.get("channel")
                if isinstance(channel, (bytes, bytearray)):
                    channel = channel.decode()
                # Store output history in Redis (list per channel)
                try:
                    client.lpush(f"history:{channel}:out", json.dumps(obj, ensure_ascii=False))
                except Exception:
                    pass
                print(f"\n📤 RECEIVED on {channel}: {json.dumps(obj, ensure_ascii=False)}")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_global_router(redis_url: str, routing: dict[str, list[str]], default_in: str) -> threading.Thread:
    """Route messages from 'agent:in' to specific agent input channels based on keyword rules.

    routing: {target_channel: [keyword1, keyword2, ...]}
    default_in: fallback target channel
    """
    def loop():
        client = redis_lib.from_url(redis_url)
        ps = client.pubsub()
        ps.subscribe("agent:in")
        print("🧭 Router subscribed to 'agent:in'")
        for item in ps.listen():
            if item and item.get("type") == "message":
                data = item.get("data")
                text = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
                try:
                    obj = json.loads(text)
                    content = (obj.get("input") or obj.get("message") or "").lower()
                except Exception:
                    obj = None
                    content = text.lower()

                target = default_in
                for channel, keywords in routing.items():
                    if any(k in content for k in keywords):
                        target = channel
                        break

                payload = obj if obj else {"input": content}
                # Store input history
                try:
                    client.lpush("history:agent:in", json.dumps(payload, ensure_ascii=False))
                    client.lpush(f"history:{target}:in", json.dumps(payload, ensure_ascii=False))
                except Exception:
                    pass
                client.publish(target, json.dumps(payload))
                print(f"↪️  ROUTED {json.dumps(payload, ensure_ascii=False)} -> {target}")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_state_dumper(redis_url: str, interval_seconds: int = 10) -> threading.Thread:
    """Periodically prints a snapshot of Redis history lists for the demo."""
    def loop():
        client = redis_lib.from_url(redis_url)
        while True:
            try:
                keys = sorted([k.decode() if isinstance(k, (bytes, bytearray)) else k for k in client.keys("history:*")])
                print("\n🔎 Redis history snapshot:")
                if not keys:
                    print("  (no history keys)")
                for k in keys:
                    try:
                        length = client.llen(k)
                        sample = [
                            (item.decode() if isinstance(item, (bytes, bytearray)) else item)
                            for item in client.lrange(k, 0, min(2, length - 1))
                        ] if length else []
                        print(f"  - {k}: len={length}, head={sample}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"(state dumper error: {e})")
            time.sleep(interval_seconds)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not redis_lib:
        print("❌ redis package not installed. pip install redis")
        return
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("❌ REDIS_URL not set. Example: export REDIS_URL=redis://:redis_password@localhost:6379/0")
        return

    model = create_model()

    # Define two agents: support and sales
    support_cfg = AgentConfig(
        name="SupportAgent",
        model=model,
        system_prompt="You are a helpful support assistant for billing and issues.",
        enable_pubsub=True,
        pubsub_sub_channels=["agent:support:in"],
        pubsub_pub_channel="agent:support:out",
    )
    sales_cfg = AgentConfig(
        name="SalesAgent",
        model=model,
        system_prompt="You are a helpful sales assistant for pricing and quotes.",
        enable_pubsub=True,
        pubsub_sub_channels=["agent:sales:in"],
        pubsub_pub_channel="agent:sales:out",
    )

    support = CoreAgent(support_cfg)
    sales = CoreAgent(sales_cfg)

    # Printer listens to all outputs
    out_printer = start_output_printer(redis_url, ["agent:support:out", "agent:sales:out"])  # noqa: F841

    # Router listens on global input and routes by keywords
    routing = {
        "agent:support:in": ["refund", "invoice", "bug", "error", "issue", "billing"],
        "agent:sales:in": ["price", "quote", "discount", "offer", "plan"],
    }
    router = start_global_router(redis_url, routing, default_in="agent:support:in")  # noqa: F841

    # Periodic state dumper
    dumper = start_state_dumper(redis_url, interval_seconds=10)  # noqa: F841

    print("\n🚀 Multi-agent pub/sub running:")
    print("   - Global IN:   agent:in (router)")
    print("   - Support IN:  agent:support:in  | OUT: agent:support:out")
    print("   - Sales IN:    agent:sales:in    | OUT: agent:sales:out")
    print("\nExamples:")
    print("  redis-cli -u $REDIS_URL PUBLISH agent:in '{\"input\":\"billing issue about invoice\"}'")
    print("  redis-cli -u $REDIS_URL PUBLISH agent:sales:in 'need a discount quote'")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")


if __name__ == "__main__":
    main()



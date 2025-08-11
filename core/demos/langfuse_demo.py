#!/usr/bin/env python3
"""
Langfuse Observability Demo for CoreAgent

This demo enables Langfuse via environment variables and runs a few invocations
so you can verify traces in your Langfuse instance (cloud or self-hosted).

Environment variables expected:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (e.g., https://cloud.langfuse.com or your self-hosted URL)

For model usage (optional, to see model spans):
  - Prefer Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT
  - Or OpenAI: OPENAI_API_KEY (and optional OPENAI_MODEL)
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.core_agent import CoreAgent
from core.config import AgentConfig
from core.utils.env import load_env_if_exists


from langchain_openai import AzureChatOpenAI, ChatOpenAI



def create_model():
    # Try Azure first
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if endpoint and api_key and AzureChatOpenAI:
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

    # Fallback to OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and ChatOpenAI:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print("🟩 Using OpenAI model:", model_name)
        return ChatOpenAI(model=model_name, temperature=0)

    print("⚠️ No LLM credentials found; running without model")
    return None


def run_demo():
    # Load .env first
    load_env_if_exists()

    # Langfuse env probe
    lf_pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    lf_sec = os.environ.get("LANGFUSE_SECRET_KEY")
    lf_host = os.environ.get("LANGFUSE_HOST")
    if not (lf_pub and lf_sec and lf_host):
        print("⚠️ Langfuse env incomplete. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.")
    else:
        print("🔑 Langfuse env detected:")
        print("   - LANGFUSE_PUBLIC_KEY: set")
        print("   - LANGFUSE_SECRET_KEY: set")
        print(f"   - LANGFUSE_HOST: {lf_host}")

    # Configure agent with Langfuse enabled
    config = AgentConfig(
        name="LangfuseCoreAgent",
        model=create_model(),
        system_prompt="You are a concise assistant.",
        enable_langfuse=True,
        langfuse_public_key=lf_pub,
        langfuse_secret_key=lf_sec,
        langfuse_host=lf_host,
    )

    agent = CoreAgent(config)
    print("✅ Agent initialized with Langfuse enabled")

    # Make a couple of calls to produce traces
    messages = [
        "Hello! Please respond with a short greeting.",
        "Name two benefits of observability for LLM apps.",
    ]
    for msg in messages:
        print(f"\n👤 {msg}")
        result = agent.invoke(msg, config={"configurable": {"thread_id": "lf-demo-21"}})
        content = result["messages"][-1].content if result and "messages" in result else "<no response>"
        print(f"🤖 {content}")

    print("\n📡 Check your Langfuse project for new traces.")


if __name__ == "__main__":
    run_demo()




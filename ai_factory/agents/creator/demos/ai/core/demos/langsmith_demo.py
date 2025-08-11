#!/usr/bin/env python3
"""
LangSmith Observability Demo for CoreAgent

This demo shows how to enable LangSmith tracing for a CoreAgent and make a few
invocations so traces appear in your LangSmith project dashboard.

Environment variables required:
  - LANGSMITH_API_KEY
  - (optional) LANGSMITH_PROJECT (defaults to 'CoreAgentDemo' if not set here)
  - (optional) LANGSMITH_ENDPOINT

It uses Azure OpenAI (like redis_memory_demo) if available; otherwise will warn.
"""

import os

from ai_factory.agents.core.core_agent import CoreAgent
from ai_factory.agents.core.config import AgentConfig
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)


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

def run_demo():
    # Load .env automatically if present
    load_dotenv(find_dotenv(usecwd=True), override=False)
    # Quick sanity check for LangSmith env
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("⚠️ LANGSMITH_API_KEY not found in environment. Ensure a .env exists in project root or CWD.")
    else:
        print("🔑 LangSmith API key detected from environment.")
    if not os.environ.get("LANGSMITH_PROJECT"):
        print("ℹ️ LANGSMITH_PROJECT not set; default will be used.")
    # Configure LangSmith via AgentConfig
    config = AgentConfig(
        name="LangSmithCoreAgent",
        model=create_model(),
        system_prompt="You are a concise assistant.",
        enable_langsmith=True,
        langsmith_project=os.environ.get("LANGSMITH_PROJECT", "CoreAgentDemo"),
        default_run_name="my-agent-invoke",
        default_run_tags=["core_agent", "dev"],
        default_run_metadata={"feature": "search", "session_group": "trial"},
    )

    agent = CoreAgent(config)
    print("✅ Agent initialized with LangSmith tracing enabled")

    # Make a couple of calls to produce traces
    messages = [
        "Hello! Please respond with a short greeting.",
        "Now, list 2 benefits of observability in LLM apps.",
    ]
    for msg in messages:
        result = agent.invoke(msg, config={"configurable": {"thread_id": "u123"}})
        content = result["messages"][-1].content if result and "messages" in result else "<no response>"
        print(f"🤖 {content}")

    print("\n📡 Check your LangSmith project for new traces.")


if __name__ == "__main__":
    run_demo()



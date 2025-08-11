#!/usr/bin/env python3
"""
Agent Registry

Single source of truth for constructing agents by name. Both Azure Functions and
FastAPI import this to retrieve agents.
"""

from __future__ import annotations

import os
from typing import Dict

from core.core_agent import CoreAgent
from core.config import AgentConfig


def _default_model():
    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except Exception:
        AzureChatOpenAI = None
        ChatOpenAI = None

    # Prefer Azure if available
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if endpoint and api_key and AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt4o"),
            temperature=0
        )

    # Fallback OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and ChatOpenAI:
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    return None


def build_agent(name: str) -> CoreAgent:
    name = name.lower()
    model = _default_model()

    # Simple routing by name – extend with tools/prompts as needed
    if name in ("support", "billing", "supportagent"):
        cfg = AgentConfig(
            name="SupportAgent",
            model=model,
            system_prompt="You are a helpful support assistant for billing and issues.",
            enable_langsmith=bool(os.environ.get("LANGSMITH_API_KEY")),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "core_agent"),
        )
        return CoreAgent(cfg)

    if name in ("sales", "pricing", "salesagent"):
        cfg = AgentConfig(
            name="SalesAgent",
            model=model,
            system_prompt="You are a helpful sales assistant for pricing and quotes.",
            enable_langsmith=bool(os.environ.get("LANGSMITH_API_KEY")),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "core_agent"),
        )
        return CoreAgent(cfg)

    # Default agent
    cfg = AgentConfig(
        name="DefaultAgent",
        model=model,
        system_prompt="You are a concise assistant.",
        enable_langsmith=bool(os.environ.get("LANGSMITH_API_KEY")),
        langsmith_project=os.environ.get("LANGSMITH_PROJECT", "core_agent"),
    )
    return CoreAgent(cfg)




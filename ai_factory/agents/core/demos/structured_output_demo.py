#!/usr/bin/env python3
"""
Structured Output Demo for CoreAgent

Shows how to force the agent to return a specific Pydantic schema using
AgentConfig.response_format.

Environment (any one works):
  - OpenAI:   OPENAI_API_KEY
  - Azure:    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT
"""

from __future__ import annotations
import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from ai_factory.agents.core.core_agent import CoreAgent
from ai_factory.agents.core.config import AgentConfig
from langchain_openai import ChatOpenAI, AzureChatOpenAI


class ProductInfo(BaseModel):
    name: str
    price: float
    currency: str
    in_stock: bool
    tags: List[str] = []


def create_model():
    # Try Azure first
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if endpoint and api_key and AzureChatOpenAI:
        # Structured outputs require 2024-08-01-preview or later on Azure
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
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

def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)

    model = create_model()
    if not model:
        print("⚠️ No LLM configured. Set OpenAI or Azure envs to run this demo.")
        return

    config = AgentConfig(
        name="StructuredAgent",
        model=model,
        system_prompt="You return only the requested ProductInfo JSON strictly.",
        response_format=ProductInfo,
    )

    agent = CoreAgent(config)

    user_input = (
        "Extract product info from this text: 'I need MacBook Air M3 priced 1299 USD; "
        "tags: laptop, apple. Are they in stock? yes'"
    )

    result = agent.invoke(user_input)
    content = result.get("messages", [])[-1].content if result else None
    print("\nStructured content (last message):\n", content)


if __name__ == "__main__":
    main()



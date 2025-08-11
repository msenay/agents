#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict
import json
import os

import requests
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from core.core_agent import CoreAgent
from core.config import AgentConfig
from core.model import CoreAgentState

from .schema import parse_recipe_yaml
from .generator import generate_files_from_recipe
from .storage import persist_files


class CreatorAgent(CoreAgent):
    """
    LLM-free CoreAgent that deterministically generates agents from YAML recipes.
    - Listens to Redis Pub/Sub if enabled (via CoreAgent's PubSubManager)
    - When invoked with JSON or plain YAML, produces files locally or to Azure Blob
    """

    def __init__(self,
                 *,
                 enable_pubsub: bool = True,
                 sub_channel: str = "creator:recipes:in",
                 pub_channel: str = "creator:recipes:out"):
        cfg = AgentConfig(
            name="CreatorAgent",
            model=None,  # no LLM
            system_prompt="Deterministic agent generator (no LLM).",
            tools=[],
            enable_memory=False,
            enable_pubsub=enable_pubsub,
            pubsub_sub_channels=[sub_channel] if enable_pubsub else [],
            pubsub_pub_channel=pub_channel if enable_pubsub else None,
        )
        super().__init__(cfg)

    def _build_agent(self, strict_mode: bool = False):
        # Override to build a custom single-node graph that performs generation
        graph = StateGraph(CoreAgentState)

        def run_creator(state: CoreAgentState) -> Dict[str, Any]:
            # Extract input text (JSON or YAML) from last human message
            input_text = ""
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    input_text = msg.content or ""
                    break

            # Parse payload
            payload: Dict[str, Any] = {}
            recipe_yaml = None
            try:
                payload = json.loads(input_text) if input_text else {}
            except Exception:
                # Treat entire input as YAML if JSON parse fails
                recipe_yaml = input_text

            try:
                if recipe_yaml is None:
                    recipe_yaml = payload.get("recipe_yaml")
                if not recipe_yaml and payload.get("recipe_url"):
                    resp = requests.get(payload["recipe_url"], timeout=10)
                    resp.raise_for_status()
                    recipe_yaml = resp.text
                if not recipe_yaml:
                    raise ValueError("Missing recipe_yaml or recipe_url in input")

                # Build recipe and artifacts
                recipe = parse_recipe_yaml(recipe_yaml)
                files = generate_files_from_recipe(recipe, original_yaml=recipe_yaml)

                dest = payload.get("dest") or os.environ.get("CREATOR_DEST", "local")
                out_base = payload.get("path") or os.environ.get("CREATOR_OUT_BASE", "ai")
                container = payload.get("container") or os.environ.get("BLOB_CONTAINER", "agents")
                overwrite = bool(payload.get("overwrite", False))

                target, paths = persist_files(files, dest=dest, out_base=out_base, container=container, overwrite=overwrite)

                result = {"status": "ok", "event": "code_ready", "name": recipe.name, "dest": target, "paths": paths}
                state.messages.append(AIMessage(content=json.dumps(result)))
            except Exception as e:
                err = {"status": "error", "message": str(e)}
                state.messages.append(AIMessage(content=json.dumps(err)))

            return {"messages": state.messages}

        graph.add_node("run_creator", run_creator)
        graph.add_edge(START, "run_creator")
        graph.add_edge("run_creator", END)

        # Compile (no memory/checkpointer)
        self.graph = graph
        self.compiled_graph = graph.compile()



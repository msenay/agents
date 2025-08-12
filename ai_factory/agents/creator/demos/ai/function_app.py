#!/usr/bin/env python3
# Azure Functions Python v2 Programming Model
# Auto-register HTTP routes for all agents under ./ai/<AgentName>/agent.py
# Route default: /api/<agentname-lower>; if ai/<AgentName>/recipe.yaml has azure.route/auth_level, use them.

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Tuple

import azure.functions as func

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

app = func.FunctionApp()

BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def _load_agent_class(agent_name: str):
    """Dynamically import ai.<AgentName>.agent and return the class <AgentName>."""
    module_path = AI_DIR / agent_name / "agent.py"
    if not module_path.exists():
        raise ImportError(f"agent.py not found for {agent_name}")
    spec = importlib.util.spec_from_file_location(f"ai.{agent_name}.agent", str(module_path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot create spec for {agent_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, agent_name):
        raise ImportError(f"Class {agent_name} not found in ai/{agent_name}/agent.py")
    return getattr(module, agent_name)


def _load_recipe_route(agent_name: str) -> Tuple[str, str]:
    route_default = f"/api/{agent_name.lower()}"
    auth_default = "function"
    recipe_path = AI_DIR / agent_name / "recipe.yaml"
    if not recipe_path.exists() or not yaml:
        return route_default, auth_default
    try:
        data = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        azure_cfg = data.get("azure") or {}
        route = azure_cfg.get("route") or route_default
        auth = (azure_cfg.get("auth_level") or auth_default).lower()
        if auth not in {"anonymous", "function", "admin"}:
            auth = auth_default
        return route, auth
    except Exception:
        return route_default, auth_default


def _register(agent_name: str) -> None:
    AgentClass = _load_agent_class(agent_name)
    route, auth = _load_recipe_route(agent_name)
    auth_level = {
        "anonymous": func.AuthLevel.ANONYMOUS,
        "function": func.AuthLevel.FUNCTION,
        "admin": func.AuthLevel.ADMIN,
    }[auth]

    def make_handler():
        @app.route(route=route, methods=["GET", "POST"], auth_level=auth_level)
        def handler(req: func.HttpRequest) -> func.HttpResponse:  # type: ignore[misc]
            agent = AgentClass()
            q = req.params.get("q")
            if not q:
                try:
                    body = req.get_json() or {}
                except Exception:
                    body = {}
                q = body.get("q") or body.get("input")
            if not q:
                return func.HttpResponse("Missing q or input", status_code=400)
            try:
                result = agent.invoke(q)
                content = result.get("messages", [])[-1].content if isinstance(result, dict) else str(result)
                return func.HttpResponse(content, status_code=200)
            except Exception as e:
                return func.HttpResponse(f"Error: {e}", status_code=500)

        handler.__name__ = f"{agent_name}_http"
        return handler

    make_handler()


# Discover and register
if AI_DIR.exists():
    for p in sorted(AI_DIR.glob("*/agent.py")):
        _register(p.parent.name)

from GoodbyeAgent.function import GoodbyeAgent
#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional
from jinja2 import Environment, BaseLoader
from pathlib import Path
import yaml

from .schema import Recipe, sanitize_class_name
from .registry import resolve_tools


AGENT_TEMPLATE = None
FUNCTION_INIT_TEMPLATE = None
FUNCTION_JSON_TEMPLATE = None


def _env() -> Environment:
    return Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)


def _load_templates():
    global AGENT_TEMPLATE, FUNCTION_INIT_TEMPLATE, FUNCTION_JSON_TEMPLATE
    if AGENT_TEMPLATE is None or FUNCTION_INIT_TEMPLATE is None or FUNCTION_JSON_TEMPLATE is None:
        base_dir = Path(__file__).resolve().parent / "templates"
        if AGENT_TEMPLATE is None:
            AGENT_TEMPLATE = (base_dir / "agent.py.j2").read_text(encoding="utf-8")
        if FUNCTION_INIT_TEMPLATE is None:
            FUNCTION_INIT_TEMPLATE = (base_dir / "function__init__.py.j2").read_text(encoding="utf-8")
        if FUNCTION_JSON_TEMPLATE is None:
            FUNCTION_JSON_TEMPLATE = (base_dir / "function_function.json.j2").read_text(encoding="utf-8")


def generate_files_from_recipe(recipe: Recipe, original_yaml: Optional[str] = None) -> Dict[str, str]:
    _load_templates()
    env = _env()

    class_name = sanitize_class_name(recipe.name)

    imports, tool_class_names = resolve_tools(recipe.tools)
    memory_enabled = bool(recipe.memory.get("enabled", False))

    pubsub_cfg = recipe.pubsub or {}
    pubsub_enabled = bool(pubsub_cfg.get("enable", False))
    pubsub_sub_channels = pubsub_cfg.get("sub_channels") or []
    pubsub_pub_channel = pubsub_cfg.get("pub_channel")

    # Defaults for azure
    route = recipe.azure.get("route") or f"/{class_name.lower()}"
    auth_level = recipe.azure.get("auth_level", "function")

    agent_py = env.from_string(AGENT_TEMPLATE).render(
        system_prompt=recipe.system_prompt,
        class_name=class_name,
        tool_imports=imports,
        tools=tool_class_names,
        memory_enabled=memory_enabled,
        pubsub_enabled=pubsub_enabled,
        pubsub_sub_channels=pubsub_sub_channels,
        pubsub_pub_channel=pubsub_pub_channel,
    )

    func_init_py = env.from_string(FUNCTION_INIT_TEMPLATE).render(
        class_name=class_name,
    )

    func_json = env.from_string(FUNCTION_JSON_TEMPLATE).render(
        route=route,
        auth_level=auth_level,
    )

    # Return in-memory file map relative to base folder
    files: Dict[str, str] = {
        f"{class_name}/agent.py": agent_py,
        f"{class_name}/function/__init__.py": func_init_py,
        f"{class_name}/function/function.json": func_json,
    }

    # Preserve original YAML if provided; otherwise dump normalized YAML
    if original_yaml is not None:
        files[f"{class_name}/recipe.yaml"] = original_yaml
    else:
        files[f"{class_name}/recipe.yaml"] = yaml.safe_dump(recipe.model_dump(), sort_keys=True, allow_unicode=True)

    return files



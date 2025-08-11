#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List, Dict, Optional
import re
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class Recipe(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    azure: Dict[str, Any] = Field(default_factory=lambda: {"route": None, "auth_level": "function"})
    pubsub: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> "Recipe":
        auth = self.azure.get("auth_level", "function")
        if auth not in {"anonymous", "function", "admin"}:
            raise ValueError("azure.auth_level must be one of: anonymous|function|admin")
        return self


def sanitize_class_name(raw_name: str) -> str:
    # Allow [A-Za-z0-9_], collapse spaces/hyphens to underscores, strip others
    cleaned = re.sub(r"[\s\-]+", "_", raw_name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    if not cleaned:
        cleaned = "Agent"
    # Ensure first letter uppercase, rest as-is
    if not cleaned[0].isalpha():
        cleaned = f"A_{cleaned}"
    return cleaned[0].upper() + cleaned[1:]


def parse_recipe_yaml(yaml_text: str) -> Recipe:
    try:
        data = yaml.safe_load(yaml_text) or {}
        return Recipe(**data)
    except ValidationError as ve:
        raise ValueError(f"Recipe validation error: {ve}")
    except Exception as e:
        raise ValueError(f"Invalid YAML or schema: {e}")



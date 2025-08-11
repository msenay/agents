#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Any

import requests

try:
    import redis  # type: ignore
except Exception:
    redis = None

from .schema import parse_recipe_yaml
from .generator import generate_files_from_recipe
from .storage import persist_files


IN_CHANNEL = "creator:recipes:in"
OUT_CHANNEL = "creator:recipes:out"


class CreatorService:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client = None

    def _ensure_client(self):
        if not redis:
            raise RuntimeError("redis package not installed. Add 'redis' to requirements.txt")
        if not self.redis_url:
            raise RuntimeError("REDIS_URL not set for CreatorService")
        self._client = redis.from_url(self.redis_url)

    def _publish(self, payload: Dict[str, Any]):
        try:
            self._client.publish(OUT_CHANNEL, json.dumps(payload))
        except Exception:
            pass

    def _handle_message(self, raw: bytes | str):
        try:
            data = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            obj = json.loads(data)
        except Exception:
            self._publish({"status": "error", "message": "Invalid JSON payload"})
            return

        try:
            # Load recipe
            recipe_yaml = obj.get("recipe_yaml")
            if not recipe_yaml and obj.get("recipe_url"):
                resp = requests.get(obj["recipe_url"], timeout=10)
                resp.raise_for_status()
                recipe_yaml = resp.text
            if not recipe_yaml:
                raise ValueError("Missing recipe_yaml or recipe_url")

            recipe = parse_recipe_yaml(recipe_yaml)

            # Generate
            files = generate_files_from_recipe(recipe, original_yaml=recipe_yaml)

            # Persist
            dest = obj.get("dest") or os.environ.get("CREATOR_DEST", "local")
            out_base = obj.get("path") or os.environ.get("CREATOR_OUT_BASE", "ai")
            container = obj.get("container") or os.environ.get("BLOB_CONTAINER", "agents")
            overwrite = bool(obj.get("overwrite", False))

            target, paths = persist_files(files, dest=dest, out_base=out_base, container=container, overwrite=overwrite)

            self._publish({
                "status": "ok",
                "event": "code_ready",
                "name": recipe.name,
                "dest": target,
                "paths": paths,
            })
        except Exception as e:
            self._publish({"status": "error", "message": str(e)})

    def _loop(self):
        try:
            self._ensure_client()
            psub = self._client.pubsub()
            psub.subscribe(IN_CHANNEL)
            for item in psub.listen():
                if self._stop.is_set():
                    break
                if item and item.get("type") == "message":
                    self._handle_message(item.get("data"))
        except Exception as e:
            self._publish({"status": "error", "message": f"Service error: {e}"})

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._thread:
            return
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass



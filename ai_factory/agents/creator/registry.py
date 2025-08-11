#!/usr/bin/env python3
from __future__ import annotations

TOOL_REGISTRY = {
    "python_executor": {"module": "core.tools", "class": "PythonExecutorTool"},
    "file_writer": {"module": "core.tools", "class": "FileWriterTool"},
    "file_reader": {"module": "core.tools", "class": "FileReaderTool"},
    "directory_list": {"module": "core.tools", "class": "DirectoryListTool"},
}

def resolve_tools(tool_keys: list[str]) -> tuple[list[dict], list[str]]:
    imports: list[dict] = []  # {module, class_name}
    tool_class_names: list[str] = []
    seen = set()
    for key in tool_keys:
        if key not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool key: {key}")
        entry = TOOL_REGISTRY[key]
        imp_sig = (entry["module"], entry["class"])
        if imp_sig not in seen:
            seen.add(imp_sig)
            imports.append({"module": entry["module"], "class_name": entry["class"]})
        tool_class_names.append(entry["class"])
    return imports, tool_class_names



#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List, Tuple


def _write_local(base_path: str, files: Dict[str, str], overwrite: bool = False) -> List[str]:
    written: List[str] = []
    for rel_path, content in files.items():
        target = os.path.join(base_path, rel_path)
        dir_name = os.path.dirname(target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"Target exists: {target}. Use overwrite=True to replace.")
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(target)
    return written


def _write_blob(container: str, prefix: str, files: Dict[str, str]) -> List[str]:
    from azure.storage.blob import BlobServiceClient  # type: ignore
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        account = os.environ.get("AZURE_STORAGE_ACCOUNT")
        key = os.environ.get("AZURE_STORAGE_KEY")
        if not account or not key:
            raise RuntimeError("Azure Blob credentials missing. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT/AZURE_STORAGE_KEY.")
        conn = f"DefaultEndpointsProtocol=https;AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"

    bsc = BlobServiceClient.from_connection_string(conn)
    try:
        bsc.create_container(container)
    except Exception:
        pass
    written: List[str] = []
    for rel_path, content in files.items():
        blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
        blob = bsc.get_blob_client(container=container, blob=blob_path)
        blob.upload_blob(content.encode("utf-8"), overwrite=True)
        written.append(f"blob://{container}/{blob_path}")
    return written


def persist_files(files: Dict[str, str], *, dest: str = "local", out_base: str = "ai", container: str = "agents", overwrite: bool = False) -> Tuple[str, List[str]]:
    if dest not in {"local", "blob"}:
        raise ValueError("dest must be 'local' or 'blob'")
    if dest == "local":
        os.makedirs(out_base, exist_ok=True)
        paths = _write_local(out_base, files, overwrite=overwrite)
        return ("local", paths)
    else:
        # Blob
        # Prefix by nothing (files already include class base folder)
        paths = _write_blob(container, prefix="", files=files)
        return ("blob", paths)



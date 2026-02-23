from __future__ import annotations

import os
from typing import Dict


PROXY_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def apply_network_env_guard() -> Dict[str, str]:
    # self-heal:network-guard-v1
    mode = str(os.getenv("SYSTEMATIC_ALPHA_PROXY_MODE", "auto")).strip().lower()
    if mode in {"off", "disabled", "keep"}:
        return {}

    removed: Dict[str, str] = {}
    for key in PROXY_KEYS:
        value = os.getenv(key)
        if value is None:
            continue
        clean = str(value).strip().lower()
        should_remove = False
        if mode in {"clear", "clear_all", "force"}:
            should_remove = True
        elif mode == "auto":
            # Frequent local broken-proxy setup seen in this environment.
            should_remove = "127.0.0.1:9" in clean or "localhost:9" in clean
        if should_remove:
            removed[key] = value
            os.environ.pop(key, None)
    return removed

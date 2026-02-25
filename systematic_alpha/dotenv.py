from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ENV_STACK: tuple[str, ...] = (
    "config/agent_lab.config",
    ".env",
)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _iter_env_items(env_path: Path) -> Iterable[tuple[str, str]]:
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        yield key, value


def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    loaded = False
    for key, value in _iter_env_items(env_path):
        if not override and key in os.environ:
            continue

        os.environ[key] = value
        loaded = True

    return loaded


def load_env_stack(
    project_root: str | Path = ".",
    files: Sequence[str | Path] | None = None,
) -> list[str]:
    """Load layered env files while preserving externally provided env vars.

    Precedence:
    1) Existing process environment (never overridden)
    2) Later files override earlier files within the provided stack
       (default: config/agent_lab.config -> .env)
    """

    root = Path(project_root)
    stack = tuple(files) if files is not None else DEFAULT_ENV_STACK
    protected = set(os.environ.keys())
    loaded_files: list[str] = []

    for entry in stack:
        env_path = Path(entry)
        if not env_path.is_absolute():
            env_path = root / env_path
        if not env_path.exists():
            continue

        applied = False
        for key, value in _iter_env_items(env_path):
            if key in protected:
                continue
            os.environ[key] = value
            applied = True

        if applied:
            loaded_files.append(str(env_path))

    return loaded_files

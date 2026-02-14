from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


def load_credentials(cli_key_file: Optional[str]) -> Tuple[str, str, str, Optional[str]]:
    env_key = os.getenv("KIS_APP_KEY")
    env_secret = os.getenv("KIS_APP_SECRET")
    env_acc = os.getenv("KIS_ACC_NO")
    env_user = os.getenv("KIS_USER_ID")
    if env_key and env_secret and env_acc:
        return env_key, env_secret, env_acc, env_user

    candidate_files = []
    if cli_key_file:
        candidate_files.append(Path(cli_key_file))

    env_key_file = os.getenv("KIS_KEY_FILE")
    if env_key_file:
        candidate_files.append(Path(env_key_file))

    candidate_files.append(Path("koreainvestment.key"))

    for key_file in candidate_files:
        if not key_file.exists():
            continue
        lines = [
            line.strip()
            for line in key_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(lines) >= 3:
            user_id = lines[3] if len(lines) >= 4 else env_user
            return lines[0], lines[1], lines[2], user_id

    raise RuntimeError(
        "Credentials not found. Put KIS_APP_KEY/KIS_APP_SECRET/KIS_ACC_NO in .env (recommended), "
        "or set env vars directly, or provide --key-file."
    )

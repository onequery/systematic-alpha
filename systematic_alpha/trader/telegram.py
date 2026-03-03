from __future__ import annotations

from typing import Any, Dict

import requests

from systematic_alpha.trader.config import TraderConfig


class TelegramClient:
    def __init__(self, config: TraderConfig):
        self.config = config
        self.enabled = bool(config.telegram_enabled and config.telegram_bot_token and config.telegram_chat_id)
        self.http = requests.Session()
        self.http.trust_env = False

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        payload: Dict[str, Any] = {
            "chat_id": self.config.telegram_chat_id,
            "text": str(text or "")[:3300],
            "disable_web_page_preview": True,
        }
        if self.config.telegram_disable_notification:
            payload["disable_notification"] = True
        if self.config.telegram_thread_id:
            payload["message_thread_id"] = self.config.telegram_thread_id
        try:
            res = self.http.post(
                f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage",
                data=payload,
                timeout=15,
            )
            return bool(res.ok)
        except Exception:
            return False

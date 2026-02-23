from __future__ import annotations

import os
from typing import Any, Dict

import requests

REPORT_EVENTS = {
    "trade_executed",
    "preopen_plan",
    "session_close_report",
    "weekly_council",
}

ACTION_REQUIRED_EVENTS = {
    "llm_limit_alert",
    "daemon_error",
    "sync_mismatch",
    "refresh_timeout",
    "broker_api_error",
}


def _truthy(value: str) -> bool:
    norm = str(value or "").strip().lower()
    return norm in {"1", "true", "yes", "y", "on"}


def _normalize_token(token: str) -> str:
    clean = str(token or "").strip()
    if clean.lower().startswith("bot"):
        return clean[3:]
    return clean


def _truncate(text: str, max_chars: int = 3500) -> str:
    body = str(text or "")
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "\n...(중략)..."


def _parse_csv_set(value: str) -> set[str]:
    out: set[str] = set()
    for raw in str(value or "").split(","):
        token = raw.strip().lower()
        if token:
            out.add(token)
    return out


def event_label(event: str) -> str:
    key = str(event or "").strip().lower()
    if key in ACTION_REQUIRED_EVENTS:
        return "Action required"
    if key in REPORT_EVENTS:
        return "보고"
    return "이벤트"


def event_prefixed_text(text: str, event: str) -> str:
    body = str(text or "").strip()
    if not body:
        body = "-"
    label = event_label(event)
    prefix = f"[{label}]"
    if body.startswith(prefix):
        return body
    return f"{prefix} {body}"


class TelegramNotifier:
    def __init__(self):
        token = _normalize_token(os.getenv("TELEGRAM_BOT_TOKEN", ""))
        chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        thread_id = str(os.getenv("TELEGRAM_THREAD_ID", "")).strip()
        enabled_raw = str(os.getenv("TELEGRAM_ENABLED", "")).strip()
        self.disable_notification = _truthy(os.getenv("TELEGRAM_DISABLE_NOTIFICATION", "0"))

        if enabled_raw:
            self.enabled = _truthy(enabled_raw) and bool(token and chat_id)
        else:
            self.enabled = bool(token and chat_id)

        self.token = token
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.session = requests.Session()
        if not _truthy(os.getenv("AGENT_LAB_TELEGRAM_USE_ENV_PROXY", "0")):
            self.session.trust_env = False
        default_events = "trade_executed,preopen_plan,session_close_report,weekly_council"
        raw_events = os.getenv("AGENT_LAB_NOTIFY_EVENTS", default_events)
        mandatory = {
            "trade_executed",
            "preopen_plan",
            "session_close_report",
            "weekly_council",
            "sync_mismatch",
            "refresh_timeout",
            "daemon_error",
            "llm_limit_alert",
            "broker_api_error",
        }
        parsed = _parse_csv_set(raw_events)
        self.allowed_events = parsed.union(mandatory)

    def _allow_event(self, event: str) -> bool:
        if "*" in self.allowed_events:
            return True
        key = str(event or "").strip().lower()
        if key in self.allowed_events:
            return True
        alias = {
            "session_monitor": "intraday_monitor",
            "intraday_monitor": "session_monitor",
        }.get(key)
        return bool(alias and alias in self.allowed_events)

    def send(self, text: str, *, event: str = "misc") -> bool:
        if not self.enabled:
            return False
        if not self._allow_event(event):
            return False
        rendered = event_prefixed_text(text, event)
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": _truncate(rendered),
            "disable_web_page_preview": "true",
        }
        if self.disable_notification:
            payload["disable_notification"] = "true"
        if self.thread_id:
            payload["message_thread_id"] = self.thread_id
        try:
            resp = self.session.post(
                f"{self.base_url}/sendMessage",
                data=payload,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return bool(data.get("ok"))
        except Exception:
            return False

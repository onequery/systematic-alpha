from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from systematic_alpha.agent_lab.storage import AgentLabStorage

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _truthy(value: str) -> bool:
    norm = str(value or "").strip().lower()
    return norm in {"1", "true", "yes", "y", "on"}


class LLMClient:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage
        self.api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        enabled_raw = os.getenv("AGENT_LAB_ENABLED")
        if enabled_raw is None or str(enabled_raw).strip() == "":
            # Backward-compatible default: if key exists, enable LLM by default.
            self.enabled = bool(self.api_key)
        else:
            self.enabled = _truthy(str(enabled_raw))
        self.model = str(os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
        self.max_daily_cost = float(os.getenv("OPENAI_MAX_DAILY_COST", "5.0") or 5.0)
        self._client = None
        self._disabled_reason = ""
        if not self.enabled:
            self._disabled_reason = "disabled_by_env"
        elif not self.api_key:
            self._disabled_reason = "missing_api_key"
        elif OpenAI is None:
            self._disabled_reason = "openai_sdk_not_installed"
        else:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception as exc:
                self._client = None
                self._disabled_reason = f"client_init_failed:{exc}"

        if self.max_daily_cost <= 0:
            self._disabled_reason = "daily_budget_nonpositive"

    def is_live(self) -> bool:
        return self.enabled and self._client is not None and self.max_daily_cost > 0

    def unavailable_reason(self) -> str:
        if self.is_live():
            return ""
        if self._disabled_reason:
            return self._disabled_reason
        return "llm_disabled_or_unavailable"

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_cost = prompt_tokens * 0.0000005
        out_cost = completion_tokens * 0.0000015
        return in_cost + out_cost

    def _current_daily_cost(self) -> float:
        rows = self.storage.list_events(event_type="llm_usage", limit=5000)
        today = _today()
        cost = 0.0
        for row in rows:
            payload = row.get("payload", {})
            if str(payload.get("date")) != today:
                continue
            try:
                cost += float(payload.get("cost_usd", 0.0))
            except Exception:
                continue
        return cost

    def _within_budget(self) -> bool:
        return self._current_daily_cost() < self.max_daily_cost

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self.is_live():
            return {
                "mode": "fallback",
                "reason": self.unavailable_reason(),
                "result": fallback,
            }
        if not self._within_budget():
            current_cost = self._current_daily_cost()
            self.storage.log_event(
                event_type="llm_budget_exceeded",
                payload={
                    "date": _today(),
                    "model": self.model,
                    "current_cost_usd": current_cost,
                    "max_daily_cost_usd": self.max_daily_cost,
                },
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            return {
                "mode": "fallback",
                "reason": "daily_budget_exceeded",
                "result": fallback,
            }

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content or "{}"
            parsed = json.loads(text)

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)
            self.storage.log_event(
                event_type="llm_usage",
                payload={
                    "date": _today(),
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                },
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            return {
                "mode": "live",
                "reason": "",
                "result": parsed,
            }
        except Exception as exc:
            self.storage.log_event(
                event_type="llm_error",
                payload={"error": repr(exc)},
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            return {
                "mode": "fallback",
                "reason": f"llm_error:{exc}",
                "result": fallback,
            }

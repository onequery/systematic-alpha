from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.orchestrator import AgentLabOrchestrator
from systematic_alpha.agent_lab.schemas import STATUS_DATA_QUALITY_LOW, STATUS_INVALID_SIGNAL
from systematic_alpha.agent_lab.storage import AgentLabStorage


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truthy(value: str) -> bool:
    norm = str(value or "").strip().lower()
    return norm in {"1", "true", "yes", "y", "on"}


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _truncate(text: str, max_chars: int = 3300) -> str:
    body = str(text or "")
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "\n...(truncated)..."


@dataclass
class TriggerSignal:
    code: str
    severity: str
    detail: str


class AutoStrategyDaemon:
    def __init__(
        self,
        *,
        project_root: str | Path,
        poll_seconds: int = 300,
        cooldown_minutes: int = 180,
        max_updates_per_day: int = 2,
    ):
        self.project_root = Path(project_root).resolve()
        self.poll_seconds = max(30, int(poll_seconds))
        self.cooldown_minutes = max(10, int(cooldown_minutes))
        self.max_updates_per_day = max(1, int(max_updates_per_day))

        self.storage = AgentLabStorage(self.project_root / "state" / "agent_lab" / "agent_lab.sqlite")
        self.orchestrator = AgentLabOrchestrator(self.project_root)
        self.llm = LLMClient(self.storage)
        self.kst = ZoneInfo("Asia/Seoul")
        self.et = ZoneInfo("America/New_York")

        token = str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        if token.lower().startswith("bot"):
            token = token[3:]
        self.telegram_token = token
        self.telegram_chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.telegram_thread_id = str(os.getenv("TELEGRAM_THREAD_ID", "")).strip()
        enabled_raw = str(os.getenv("TELEGRAM_ENABLED", "")).strip()
        self.telegram_disable_notification = _truthy(os.getenv("TELEGRAM_DISABLE_NOTIFICATION", "0"))
        if enabled_raw:
            self.telegram_enabled = _truthy(enabled_raw) and bool(self.telegram_token and self.telegram_chat_id)
        else:
            self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)

        self.http = requests.Session()
        self.http.trust_env = False

    def close(self) -> None:
        self.orchestrator.close()
        self.storage.close()

    @staticmethod
    def _is_llm_budget_or_quota_reason(reason: str) -> bool:
        text = str(reason or "").lower()
        keys = (
            "daily_budget_exceeded",
            "insufficient_quota",
            "quota",
            "billing",
            "context_length",
            "max_tokens",
            "token",
        )
        return any(key in text for key in keys)

    def _send_telegram(self, text: str) -> None:
        if not self.telegram_enabled:
            return
        payload: Dict[str, Any] = {
            "chat_id": self.telegram_chat_id,
            "text": _truncate(text),
            "disable_web_page_preview": True,
        }
        if self.telegram_disable_notification:
            payload["disable_notification"] = True
        if self.telegram_thread_id:
            payload["message_thread_id"] = self.telegram_thread_id
        try:
            self.http.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                data=payload,
                timeout=15,
            )
        except Exception:
            pass

    def _maybe_send_llm_limit_alert(self, reason: str) -> None:
        if not self._is_llm_budget_or_quota_reason(reason):
            return
        latest = self.storage.get_latest_event("auto_strategy_llm_limit_alert")
        if latest:
            created = _parse_iso(str(latest.get("created_at", "")))
            if created is not None:
                delta = datetime.now() - created
                if delta < timedelta(hours=2):
                    return
        payload = {"reason": str(reason or ""), "at": _now_iso()}
        self.storage.log_event("auto_strategy_llm_limit_alert", payload, payload["at"])
        self._send_telegram(
            "[AgentLab] alert\n"
            "auto-strategy daemon hit OpenAI token/quota/budget limit.\n"
            f"reason={payload['reason']}\n"
            "action=Increase OPENAI_MAX_DAILY_COST, check billing/quota, or reduce auto-strategy frequency."
        )

    def _latest_equity_rows(self) -> Dict[str, List[Dict[str, Any]]]:
        rows = self.storage.query_all(
            """
            SELECT *
            FROM equity_curve
            ORDER BY as_of_date DESC, equity_id DESC
            LIMIT 400
            """
        )
        by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            aid = str(row.get("agent_id", ""))
            by_agent.setdefault(aid, [])
            if len(by_agent[aid]) < 2:
                by_agent[aid].append(row)
        return by_agent

    def _recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.storage.query_all(
            """
            SELECT *
            FROM session_signals
            ORDER BY created_at DESC, session_signal_id DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def _recent_auto_updates(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.storage.list_events(event_type="auto_strategy_update", limit=limit)

    def _within_restricted_window(self, now_kst: datetime) -> bool:
        # Keep strategy mutation away from open/volatile windows.
        if now_kst.weekday() < 5:
            kr_start = now_kst.replace(hour=8, minute=50, second=0, microsecond=0)
            kr_end = now_kst.replace(hour=9, minute=40, second=0, microsecond=0)
            if kr_start <= now_kst <= kr_end:
                return True

        now_et = now_kst.astimezone(self.et)
        if now_et.weekday() < 5:
            us_start = now_et.replace(hour=9, minute=20, second=0, microsecond=0)
            us_end = now_et.replace(hour=10, minute=10, second=0, microsecond=0)
            if us_start <= now_et <= us_end:
                return True
        return False

    def _detect_triggers(self, now_kst: datetime) -> List[TriggerSignal]:
        triggers: List[TriggerSignal] = []
        by_agent = self._latest_equity_rows()

        for aid, rows in by_agent.items():
            if not rows:
                continue
            latest = rows[0]
            drawdown = float(latest.get("drawdown", 0.0) or 0.0)
            if drawdown <= -0.03:
                triggers.append(
                    TriggerSignal(
                        code="DRAWDOWN_SPIKE",
                        severity="high",
                        detail=f"{aid} drawdown={drawdown:.4f}",
                    )
                )
            if len(rows) >= 2:
                prev = rows[1]
                prev_eq = float(prev.get("equity_krw", 0.0) or 0.0)
                cur_eq = float(latest.get("equity_krw", 0.0) or 0.0)
                if prev_eq > 0:
                    day_ret = (cur_eq / prev_eq) - 1.0
                    if day_ret <= -0.015:
                        triggers.append(
                            TriggerSignal(
                                code="DAILY_LOSS_SPIKE",
                                severity="high",
                                detail=f"{aid} day_return={day_ret:.4f}",
                            )
                        )

        recent_signals = self._recent_signals(limit=12)
        bad = 0
        for row in recent_signals:
            status = str(row.get("status_code", "") or "")
            invalid_reason = str(row.get("invalid_reason", "") or "")
            if status in {STATUS_DATA_QUALITY_LOW, STATUS_INVALID_SIGNAL}:
                bad += 1
            elif invalid_reason.startswith("realtime_coverage_too_low"):
                bad += 1
        if bad >= 4:
            triggers.append(
                TriggerSignal(
                    code="SIGNAL_QUALITY_DRIFT",
                    severity="high",
                    detail=f"bad_session_signals={bad}/{len(recent_signals)}",
                )
            )

        weekly = self.storage.query_one(
            """
            SELECT *
            FROM weekly_councils
            ORDER BY created_at DESC, weekly_council_id DESC
            LIMIT 1
            """
        )
        stale_days = 7
        if weekly is None:
            triggers.append(
                TriggerSignal(
                    code="NO_COUNCIL_HISTORY",
                    severity="medium",
                    detail="weekly_councils is empty",
                )
            )
        else:
            created = _parse_iso(str(weekly.get("created_at", "")))
            if created is not None and (now_kst.replace(tzinfo=None) - created.replace(tzinfo=None)).days >= stale_days:
                triggers.append(
                    TriggerSignal(
                        code="STALE_STRATEGY_REVIEW",
                        severity="medium",
                        detail=f"last_council_at={weekly.get('created_at')}",
                    )
                )
        return triggers

    def _cooldown_ok(self, now_kst: datetime) -> bool:
        updates = self._recent_auto_updates(limit=80)
        today = now_kst.strftime("%Y-%m-%d")
        today_count = 0
        latest_at: Optional[datetime] = None
        for row in updates:
            created = _parse_iso(str(row.get("created_at", "")))
            if created is None:
                continue
            if created.strftime("%Y-%m-%d") == today:
                today_count += 1
            if latest_at is None:
                latest_at = created
        if today_count >= self.max_updates_per_day:
            return False
        if latest_at is None:
            return True
        elapsed = now_kst.replace(tzinfo=None) - latest_at.replace(tzinfo=None)
        return elapsed >= timedelta(minutes=self.cooldown_minutes)

    def _llm_vote(self, now_kst: datetime, triggers: List[TriggerSignal]) -> Dict[str, Any]:
        fallback = {
            "should_update_now": True,
            "reason": "rule_based_trigger",
            "wait_minutes": 0,
        }
        payload = [{"code": t.code, "severity": t.severity, "detail": t.detail} for t in triggers]
        response = self.llm.generate_json(
            system_prompt=(
                "You are a portfolio committee scheduler. "
                "Return JSON with keys: should_update_now (bool), reason (string), wait_minutes (integer)."
            ),
            user_prompt=(
                f"now_kst={now_kst.isoformat()}\n"
                f"triggers={json.dumps(payload, ensure_ascii=False)}\n"
                "If update timing is risky, propose short wait_minutes."
            ),
            fallback=fallback,
            temperature=0.1,
        )
        result = response.get("result", fallback)
        if not isinstance(result, dict):
            result = dict(fallback)
        out = {
            "should_update_now": bool(result.get("should_update_now", True)),
            "reason": str(result.get("reason", "")) or "rule_based_trigger",
            "wait_minutes": max(0, int(result.get("wait_minutes", 0) or 0)),
            "mode": str(response.get("mode", "fallback")),
            "raw_reason": str(response.get("reason", "")),
        }
        return out

    def _run_council_update(self, now_kst: datetime, triggers: List[TriggerSignal], vote: Dict[str, Any]) -> Dict[str, Any]:
        iso_year, iso_week, _ = now_kst.isocalendar()
        week_id = f"{iso_year}-W{iso_week:02d}"
        decision = self.orchestrator.weekly_council(week_id=week_id)
        out = {
            "updated": True,
            "week_id": week_id,
            "champion_agent_id": decision.get("champion_agent_id", ""),
            "promoted_versions": decision.get("promoted_versions", {}),
            "trigger_codes": [t.code for t in triggers],
            "vote": vote,
            "updated_at": _now_iso(),
        }

        yyyymmdd = now_kst.strftime("%Y%m%d")
        out_dir = self.project_root / "out" / "agent_lab" / yyyymmdd
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = now_kst.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"auto_strategy_update_{stamp}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["artifact_path"] = str(path)
        return out

    def run_once(self) -> Dict[str, Any]:
        now_kst = datetime.now(self.kst)
        triggers = self._detect_triggers(now_kst)

        if not triggers:
            result = {
                "action": "skip",
                "reason": "no_triggers",
                "now_kst": now_kst.isoformat(),
                "trigger_count": 0,
            }
            return result

        if self._within_restricted_window(now_kst):
            result = {
                "action": "skip",
                "reason": "restricted_window",
                "now_kst": now_kst.isoformat(),
                "triggers": [t.__dict__ for t in triggers],
            }
            return result

        if not self._cooldown_ok(now_kst):
            result = {
                "action": "skip",
                "reason": "cooldown_or_daily_limit",
                "now_kst": now_kst.isoformat(),
                "triggers": [t.__dict__ for t in triggers],
            }
            return result

        vote = self._llm_vote(now_kst, triggers)
        self._maybe_send_llm_limit_alert(vote.get("raw_reason") or vote.get("reason") or "")
        if (not vote["should_update_now"]) or vote["wait_minutes"] > 0:
            result = {
                "action": "skip",
                "reason": "agent_vote_delay",
                "vote": vote,
                "now_kst": now_kst.isoformat(),
                "triggers": [t.__dict__ for t in triggers],
            }
            return result

        updated = self._run_council_update(now_kst, triggers, vote)
        self.storage.log_event(
            event_type="auto_strategy_update",
            payload=updated,
            created_at=_now_iso(),
        )
        self._send_telegram(
            "[AgentLab] auto-strategy-update\n"
            f"week={updated.get('week_id')}\n"
            f"champion={updated.get('champion_agent_id')}\n"
            f"trigger_codes={','.join(updated.get('trigger_codes', []))}\n"
            f"vote_mode={vote.get('mode')}\n"
            f"vote_reason={vote.get('reason')}\n"
            f"artifact={updated.get('artifact_path')}"
        )
        return {"action": "updated", **updated}

    def run(self, *, once: bool = False) -> Dict[str, Any]:
        self.storage.log_event(
            event_type="auto_strategy_daemon_start",
            payload={"poll_seconds": self.poll_seconds},
            created_at=_now_iso(),
        )
        if once:
            return self.run_once()

        latest: Dict[str, Any] = {"action": "idle"}
        while True:
            try:
                latest = self.run_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                err = {"action": "error", "error": repr(exc), "at": _now_iso()}
                latest = err
                self.storage.log_event("auto_strategy_daemon_error", err, _now_iso())
                self._send_telegram(
                    "[AgentLab] auto-strategy-daemon error\n"
                    f"error={repr(exc)}"
                )
                time.sleep(10)
                continue
            time.sleep(self.poll_seconds)
        self.storage.log_event("auto_strategy_daemon_stop", latest, _now_iso())
        return latest


def run_auto_strategy_daemon(
    *,
    project_root: str | Path,
    poll_seconds: int = 300,
    cooldown_minutes: int = 180,
    max_updates_per_day: int = 2,
    once: bool = False,
) -> Dict[str, Any]:
    daemon = AutoStrategyDaemon(
        project_root=project_root,
        poll_seconds=poll_seconds,
        cooldown_minutes=cooldown_minutes,
        max_updates_per_day=max_updates_per_day,
    )
    try:
        return daemon.run(once=once)
    finally:
        daemon.close()

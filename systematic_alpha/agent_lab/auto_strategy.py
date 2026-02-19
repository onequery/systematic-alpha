from __future__ import annotations

import json
import os
import subprocess
import sys
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
    return body[:max_chars] + "\n...(중략)..."


def _parse_csv_set(value: str) -> set[str]:
    out: set[str] = set()
    for raw in str(value or "").split(","):
        token = raw.strip().lower()
        if token:
            out.add(token)
    return out


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
        poll_seconds: int = 30,
        cooldown_minutes: int = 180,
        max_updates_per_day: int = 2,
    ):
        self.project_root = Path(project_root).resolve()
        self.poll_seconds = max(10, int(poll_seconds))
        self.cooldown_minutes = max(10, int(cooldown_minutes))
        self.max_updates_per_day = max(1, int(max_updates_per_day))
        self.heartbeat_minutes = max(5, int(float(os.getenv("AGENT_LAB_HEARTBEAT_MINUTES", "30") or 30)))

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
        default_events = "trade_executed,preopen_plan,session_close_report,weekly_council"
        self.allowed_notify_events = _parse_csv_set(os.getenv("AGENT_LAB_NOTIFY_EVENTS", default_events))

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

    def _allow_event(self, event: str) -> bool:
        if "*" in self.allowed_notify_events:
            return True
        return str(event or "").strip().lower() in self.allowed_notify_events

    def _send_telegram(self, text: str, *, event: str = "misc") -> None:
        if not self.telegram_enabled:
            return
        if not self._allow_event(event):
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
            "[AgentLab] 알림\n"
            "오토전략 데몬에서 OpenAI 토큰/쿼터/예산 제한에 도달했습니다.\n"
            f"사유={payload['reason']}\n"
            "조치=OPENAI_MAX_DAILY_COST 상향, 결제/쿼터 점검, 오토전략 빈도 축소"
            ,
            event="llm_limit_alert",
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

    def _latest_event_by_market(self, event_type: str, market: str, limit: int = 240) -> Optional[Dict[str, Any]]:
        rows = self.storage.list_events(event_type=event_type, limit=limit)
        m = str(market or "").upper()
        for row in rows:
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("market", "")).upper() == m:
                return row
        return None

    def _latest_session_signal_for_market(self, market: str) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            """
            SELECT *
            FROM session_signals
            WHERE market = ?
            ORDER BY created_at DESC, session_signal_id DESC
            LIMIT 1
            """,
            (str(market).upper(),),
        )
        if row is None:
            return None
        try:
            row["payload"] = json.loads(str(row.get("payload_json") or "{}"))
        except Exception:
            row["payload"] = {}
        return row

    @staticmethod
    def _max_freedom_mode() -> bool:
        return _truthy(os.getenv("AGENT_LAB_MAX_FREEDOM", "1"))

    @staticmethod
    def _ko_signal_status(value: str) -> str:
        mapping = {
            "SIGNAL_OK": "신호정상",
            "MARKET_CLOSED": "장종료",
            "INVALID_SIGNAL": "신호무효",
            "DATA_QUALITY_LOW": "데이터품질저하",
            "NO_SIGNAL": "신호없음",
        }
        key = str(value or "").strip().upper()
        ko = mapping.get(key)
        if ko:
            return f"{ko}({key})"
        return str(value or "-")

    @staticmethod
    def _ko_action(value: str) -> str:
        mapping = {
            "monitor_only": "모니터링만",
            "skip": "건너뜀",
            "updated": "업데이트",
            "idle": "유휴",
            "error": "오류",
        }
        return mapping.get(str(value or "").strip(), str(value or "-"))

    @staticmethod
    def _ko_monitor_state(value: str) -> str:
        mapping = {
            "outside_window": "장외",
            "disabled": "비활성",
            "waiting": "대기",
            "executed": "실행",
            "idle": "유휴",
        }
        raw = str(value or "").strip()
        return mapping.get(raw, raw or "-")

    def _refresh_intraday_signal(self, market: str, now_kst: datetime) -> Dict[str, Any]:
        market = str(market).upper()
        market_lc = market.lower()
        run_date = now_kst.strftime("%Y%m%d")
        stamp = now_kst.strftime("%Y%m%d_%H%M%S")

        out_base = self.project_root / "out" / market_lc / run_date
        result_dir = out_base / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.project_root / "logs" / "agent_lab" / run_date
        log_dir.mkdir(parents=True, exist_ok=True)

        output_json = result_dir / f"{market_lc}_daily_{stamp}.json"
        log_path = log_dir / f"intraday_signal_refresh_{market_lc}_{stamp}.log"
        analytics_dir = out_base / "analytics"
        overnight_path = out_base / "selection_overnight_report.csv"

        collect_seconds = int(float(os.getenv("AGENT_LAB_INTRADAY_COLLECT_SECONDS", "45") or 45))
        final_picks = int(float(os.getenv("AGENT_LAB_INTRADAY_FINAL_PICKS", "20") or 20))
        pre_candidates = int(float(os.getenv("AGENT_LAB_INTRADAY_PRE_CANDIDATES", "120") or 120))
        max_symbols_scan = int(float(os.getenv("AGENT_LAB_INTRADAY_MAX_SYMBOLS_SCAN", "200") or 200))
        kr_universe_size = int(float(os.getenv("AGENT_LAB_INTRADAY_KR_UNIVERSE_SIZE", "300") or 300))
        us_universe_size = int(float(os.getenv("AGENT_LAB_INTRADAY_US_UNIVERSE_SIZE", "500") or 500))
        us_exchange = str(os.getenv("AGENT_LAB_US_EXCHANGE", "NASD")).strip().upper() or "NASD"

        cmd = [
            sys.executable,
            "-u",
            "main.py",
            "--market",
            market_lc,
            "--collect-seconds",
            str(max(5, collect_seconds)),
            "--final-picks",
            str(max(1, final_picks)),
            "--pre-candidates",
            str(max(1, pre_candidates)),
            "--max-symbols-scan",
            str(max(50, max_symbols_scan)),
            "--kr-universe-size",
            str(max(50, kr_universe_size)),
            "--us-universe-size",
            str(max(50, us_universe_size)),
            "--min-change-pct",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_CHANGE_PCT", "0") or 0.0)),
            "--min-gap-pct",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_GAP_PCT", "0") or 0.0)),
            "--min-strength",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_STRENGTH", "0") or 0.0)),
            "--min-vol-ratio",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_VOL_RATIO", "0") or 0.0)),
            "--min-bid-ask-ratio",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_BID_ASK_RATIO", "0") or 0.0)),
            "--min-pass-conditions",
            str(int(float(os.getenv("AGENT_LAB_INTRADAY_MIN_PASS_CONDITIONS", "1") or 1))),
            "--min-maintain-ratio",
            str(float(os.getenv("AGENT_LAB_INTRADAY_MIN_MAINTAIN_RATIO", "0") or 0.0)),
            "--output-json",
            str(output_json),
            "--analytics-dir",
            str(analytics_dir),
            "--overnight-report-path",
            str(overnight_path),
            "--long-only",
        ]
        allow_low_cov = _truthy(os.getenv("AGENT_LAB_INTRADAY_ALLOW_LOW_COVERAGE", "1"))
        cmd.append("--allow-low-coverage" if allow_low_cov else "--invalidate-on-low-coverage")
        if market == "US":
            cmd.extend(["--exchange", us_exchange])

        started = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(120, collect_seconds + 180),
        )
        elapsed = time.perf_counter() - started
        log_body = (
            f"[intraday-refresh] started={now_kst.isoformat()}\n"
            f"command={' '.join(cmd)}\n"
            f"returncode={completed.returncode}\n"
            f"elapsed_sec={elapsed:.1f}\n\n"
            f"[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}\n"
        )
        log_path.write_text(log_body, encoding="utf-8")

        payload = {
            "market": market,
            "run_date": run_date,
            "ok": bool(completed.returncode == 0 and output_json.exists()),
            "returncode": int(completed.returncode),
            "elapsed_sec": round(elapsed, 2),
            "output_json": str(output_json),
            "log_path": str(log_path),
        }
        if completed.returncode != 0:
            payload["error"] = f"returncode={completed.returncode}"
        self.storage.log_event("auto_intraday_signal_refresh", payload, _now_iso())
        return payload

    def _intraday_monitor_policy(self) -> Dict[str, Any]:
        agents = self.storage.list_agents()
        enabled_agents: List[str] = []
        intervals: List[int] = []
        by_agent: Dict[str, Dict[str, Any]] = {}
        forced_max_freedom = self._max_freedom_mode()
        forced_interval = int(float(os.getenv("AGENT_LAB_MAX_FREEDOM_INTERVAL_SEC", "30") or 30))
        forced_interval = max(10, min(3600, forced_interval))
        for row in agents:
            aid = str(row.get("agent_id", ""))
            try:
                active = self.orchestrator.registry.get_active_strategy(aid)
            except Exception:
                continue
            params = active.get("params", {}) or {}
            enabled = bool(float(params.get("intraday_monitor_enabled", 0) or 0) >= 0.5)
            interval = int(float(params.get("intraday_monitor_interval_sec", params.get("collect_seconds", self.poll_seconds)) or self.poll_seconds))
            interval = max(10, min(3600, interval))
            if forced_max_freedom:
                enabled = True
                interval = forced_interval
            by_agent[aid] = {
                "enabled": enabled,
                "interval_sec": interval,
                "version_tag": str(active.get("version_tag", "")),
                "forced_max_freedom": forced_max_freedom,
            }
            if enabled:
                enabled_agents.append(aid)
                intervals.append(interval)
        enabled = len(enabled_agents) > 0
        interval_sec = min(intervals) if intervals else 0
        return {
            "enabled": enabled,
            "interval_sec": int(interval_sec),
            "enabled_agents": enabled_agents,
            "by_agent": by_agent,
            "markets": ["KR", "US"],
        }

    def _is_market_open_now(self, market: str, now_kst: datetime) -> bool:
        m = str(market).upper()
        if m == "KR":
            if now_kst.weekday() >= 5:
                return False
            start = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
            end = now_kst.replace(hour=15, minute=35, second=0, microsecond=0)
            return start <= now_kst <= end

        now_et = now_kst.astimezone(self.et)
        if now_et.weekday() >= 5:
            return False
        start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
        return start <= now_et <= end

    def _proposal_status_counts(self, market: str, session_date: str) -> Dict[str, int]:
        rows = self.storage.query_all(
            """
            SELECT status, COUNT(*) AS c
            FROM order_proposals
            WHERE market = ? AND session_date = ?
            GROUP BY status
            """,
            (str(market).upper(), str(session_date)),
        )
        out: Dict[str, int] = {}
        for row in rows:
            out[str(row.get("status", "UNKNOWN"))] = int(row.get("c", 0) or 0)
        return out

    def _run_intraday_monitor_cycle(self, market: str, now_kst: datetime) -> Dict[str, Any]:
        market = str(market).upper()
        refresh_enabled = self._max_freedom_mode() or _truthy(os.getenv("AGENT_LAB_INTRADAY_SIGNAL_REFRESH", "1"))
        payload: Dict[str, Any] = {
            "market": market,
            "executed_at_kst": now_kst.isoformat(),
            "mode": "observe",
            "session_date": "",
            "signal_status": "NO_SIGNAL",
            "proposal_status_counts": {},
            "refresh_enabled": refresh_enabled,
            "refresh_triggered": False,
            "refresh_ok": False,
            "propose_triggered": False,
            "propose_ok": False,
        }
        if refresh_enabled:
            payload["refresh_triggered"] = True
            refresh = self._refresh_intraday_signal(market, now_kst)
            payload["refresh_ok"] = bool(refresh.get("ok", False))
            payload["refresh_log"] = refresh.get("log_path", "")
            payload["refresh_elapsed_sec"] = refresh.get("elapsed_sec", 0.0)
            if not bool(refresh.get("ok", False)):
                payload["refresh_error"] = refresh.get("error", "intraday_refresh_failed")
            else:
                signal_date = now_kst.strftime("%Y%m%d")
                try:
                    self.orchestrator.ingest_session(market, signal_date)
                except Exception as exc:
                    payload["refresh_ok"] = False
                    payload["refresh_error"] = f"ingest_failed:{repr(exc)}"
                    payload["refresh_ingest_ok"] = False
                else:
                    payload["refresh_ingest_ok"] = True

        signal = self._latest_session_signal_for_market(market)
        signal_date = str(signal.get("session_date", "")) if signal else ""
        signal_status = str(signal.get("status_code", "NO_SIGNAL")) if signal else "NO_SIGNAL"
        status_counts = self._proposal_status_counts(market, signal_date) if signal_date else {}
        payload["session_date"] = signal_date
        payload["signal_status"] = signal_status
        payload["proposal_status_counts"] = status_counts

        # Optional advanced mode: re-run propose cycle during open sessions.
        propose_enabled = self._max_freedom_mode() or _truthy(os.getenv("AGENT_LAB_INTRADAY_MONITOR_PROPOSE", "1"))
        if propose_enabled and signal and signal_status == "SIGNAL_OK":
            try:
                proposed = self.orchestrator.propose_orders(
                    market=market,
                    yyyymmdd=signal_date,
                    auto_execute=True,
                )
                proposals = list(proposed.get("proposals") or [])
                payload["mode"] = "observe+propose"
                payload["propose_triggered"] = True
                payload["propose_ok"] = True
                payload["propose_rows"] = len(proposals)
                payload["propose_executed"] = sum(
                    1 for row in proposals if str((row or {}).get("status", "")).upper() == "EXECUTED"
                )
            except Exception as exc:
                payload["mode"] = "observe+propose"
                payload["propose_triggered"] = True
                payload["propose_ok"] = False
                payload["propose_error"] = repr(exc)

        self.storage.log_event("auto_intraday_monitor_cycle", payload, _now_iso())
        signal_text = self._ko_signal_status(signal_status)
        self._send_telegram(
            "[AgentLab] 장중 모니터링\n"
            f"시장={market}\n"
            f"모드={payload.get('mode')}\n"
            f"신호={signal_text} ({signal_date or '-'})\n"
            f"리프레시_성공={payload.get('refresh_ok')}\n"
            f"제안_상태={status_counts if status_counts else '(없음)'}\n"
            f"제안_트리거={payload.get('propose_triggered')}\n"
            f"제안_성공={payload.get('propose_ok')}"
            ,
            event="intraday_monitor",
        )
        return payload

    def _run_intraday_monitor(self, now_kst: datetime) -> Dict[str, Any]:
        policy = self._intraday_monitor_policy()
        result: Dict[str, Any] = {
            "enabled": bool(policy.get("enabled", False)),
            "interval_sec": int(policy.get("interval_sec", 0) or 0),
            "enabled_agents": list(policy.get("enabled_agents", [])),
            "markets": {},
            "executed_markets": [],
        }
        if not result["enabled"]:
            for mk in ["KR", "US"]:
                result["markets"][mk] = {"state": "disabled", "open": self._is_market_open_now(mk, now_kst)}
            return result

        interval_sec = max(10, int(result["interval_sec"] or 10))
        now_naive = now_kst.replace(tzinfo=None)
        for mk in ["KR", "US"]:
            open_now = self._is_market_open_now(mk, now_kst)
            state: Dict[str, Any] = {"open": open_now, "state": "idle"}
            last_cycle = self._latest_event_by_market("auto_intraday_monitor_cycle", mk)
            if last_cycle:
                state["last_cycle_at"] = str(last_cycle.get("created_at", ""))
            if not open_now:
                state["state"] = "outside_window"
                result["markets"][mk] = state
                continue

            due = True
            if last_cycle:
                created = _parse_iso(str(last_cycle.get("created_at", "")))
                if created is not None:
                    elapsed = (now_naive - created).total_seconds()
                    if elapsed < interval_sec:
                        due = False
                        state["state"] = "waiting"
                        state["seconds_until_next"] = int(max(0.0, interval_sec - elapsed))
            if not due:
                result["markets"][mk] = state
                continue

            cycle = self._run_intraday_monitor_cycle(mk, now_kst)
            state["state"] = "executed"
            state["session_date"] = str(cycle.get("session_date", ""))
            state["signal_status"] = str(cycle.get("signal_status", ""))
            state["propose_triggered"] = bool(cycle.get("propose_triggered", False))
            state["propose_ok"] = bool(cycle.get("propose_ok", False))
            state["seconds_until_next"] = interval_sec
            result["executed_markets"].append(mk)
            result["markets"][mk] = state
        return result

    def _maybe_send_heartbeat_telegram(self, payload: Dict[str, Any]) -> None:
        latest = self.storage.get_latest_event("auto_strategy_heartbeat_notice")
        if latest:
            created = _parse_iso(str(latest.get("created_at", "")))
            if created is not None:
                delta = datetime.now() - created
                if delta < timedelta(minutes=self.heartbeat_minutes):
                    return
        monitor = payload.get("intraday_monitor", {}) if isinstance(payload, dict) else {}
        mk = monitor.get("markets", {}) if isinstance(monitor, dict) else {}
        kr = mk.get("KR", {}) if isinstance(mk, dict) else {}
        us = mk.get("US", {}) if isinstance(mk, dict) else {}
        action_ko = self._ko_action(str(payload.get("action", "-")))
        kr_state = self._ko_monitor_state(str(kr.get("state", "-")))
        us_state = self._ko_monitor_state(str(us.get("state", "-")))
        self._send_telegram(
            "[AgentLab] 하트비트\n"
            f"액션={action_ko}({payload.get('action', '-')})\n"
            f"사유={payload.get('reason', '-')}\n"
            f"다음_폴링={payload.get('next_poll_in_sec', self.poll_seconds)}s\n"
            f"모니터_활성={monitor.get('enabled', False)} 주기={monitor.get('interval_sec', 0)}s\n"
            f"KR={kr_state}, US={us_state}"
            ,
            event="heartbeat",
        )
        self.storage.log_event("auto_strategy_heartbeat_notice", payload, _now_iso())

    def _emit_heartbeat(self, latest: Dict[str, Any], daemon_status: str = "ok") -> None:
        now_kst = datetime.now(self.kst)
        next_poll_at = now_kst + timedelta(seconds=self.poll_seconds)
        payload = {
            "daemon_status": daemon_status,
            "action": str(latest.get("action", "idle")),
            "reason": str(latest.get("reason", "")),
            "next_poll_in_sec": int(self.poll_seconds),
            "next_poll_at_kst": next_poll_at.isoformat(),
            "intraday_monitor": latest.get("intraday_monitor", {}),
            "updated_at_kst": now_kst.isoformat(),
        }
        self.storage.log_event("auto_strategy_heartbeat", payload, _now_iso())
        self._maybe_send_heartbeat_telegram(payload)

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
        intraday_monitor = self._run_intraday_monitor(now_kst)
        triggers = self._detect_triggers(now_kst)

        if not triggers:
            result = {
                "action": "monitor_only" if intraday_monitor.get("executed_markets") else "skip",
                "reason": "no_triggers",
                "now_kst": now_kst.isoformat(),
                "trigger_count": 0,
                "intraday_monitor": intraday_monitor,
            }
            return result

        if self._within_restricted_window(now_kst):
            result = {
                "action": "skip",
                "reason": "restricted_window",
                "now_kst": now_kst.isoformat(),
                "triggers": [t.__dict__ for t in triggers],
                "intraday_monitor": intraday_monitor,
            }
            return result

        if not self._cooldown_ok(now_kst):
            result = {
                "action": "skip",
                "reason": "cooldown_or_daily_limit",
                "now_kst": now_kst.isoformat(),
                "triggers": [t.__dict__ for t in triggers],
                "intraday_monitor": intraday_monitor,
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
                "intraday_monitor": intraday_monitor,
            }
            return result

        updated = self._run_council_update(now_kst, triggers, vote)
        self.storage.log_event(
            event_type="auto_strategy_update",
            payload=updated,
            created_at=_now_iso(),
        )
        self._send_telegram(
            "[AgentLab] 오토전략 업데이트\n"
            f"주차={updated.get('week_id')}\n"
            f"우승전략={updated.get('champion_agent_id')}\n"
            f"트리거_코드={','.join(updated.get('trigger_codes', []))}\n"
            f"투표_모드={vote.get('mode')}\n"
            f"투표_사유={vote.get('reason')}\n"
            f"결과_파일={updated.get('artifact_path')}"
            ,
            event="auto_strategy_update",
        )
        return {"action": "updated", "intraday_monitor": intraday_monitor, **updated}

    def run(self, *, once: bool = False) -> Dict[str, Any]:
        self.storage.log_event(
            event_type="auto_strategy_daemon_start",
            payload={"poll_seconds": self.poll_seconds},
            created_at=_now_iso(),
        )
        self._send_telegram(
            "[AgentLab] 오토전략 데몬 시작\n"
            f"폴링_주기(초)={self.poll_seconds}\n"
            f"쿨다운(분)={self.cooldown_minutes}\n"
            f"일일_최대_업데이트={self.max_updates_per_day}\n"
            f"하트비트_주기(분)={self.heartbeat_minutes}"
            ,
            event="daemon_start",
        )
        if once:
            latest_once = self.run_once()
            self._emit_heartbeat(latest_once, daemon_status="ok")
            return latest_once

        latest: Dict[str, Any] = {"action": "idle"}
        while True:
            try:
                latest = self.run_once()
                self._emit_heartbeat(latest, daemon_status="ok")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                err = {"action": "error", "error": repr(exc), "at": _now_iso()}
                latest = err
                self.storage.log_event("auto_strategy_daemon_error", err, _now_iso())
                self._emit_heartbeat(err, daemon_status="error")
                self._send_telegram(
                    "[AgentLab] 오토전략 데몬 오류\n"
                    f"오류={repr(exc)}"
                    ,
                    event="daemon_error",
                )
                time.sleep(10)
                continue
            time.sleep(self.poll_seconds)
        self.storage.log_event("auto_strategy_daemon_stop", latest, _now_iso())
        return latest


def run_auto_strategy_daemon(
    *,
    project_root: str | Path,
    poll_seconds: int = 30,
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

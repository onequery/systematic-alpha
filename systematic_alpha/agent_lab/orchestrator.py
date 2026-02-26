from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from systematic_alpha.agent_lab.accounting import AccountingEngine
from systematic_alpha.agent_lab.agents import (
    AgentDecisionEngine,
    build_default_agent_profiles,
    normalize_agent_constraints,
    profile_from_agent_row,
)
from systematic_alpha.agent_lab.identity import AgentIdentityStore
from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.notify import TelegramNotifier
from systematic_alpha.agent_lab.paper_broker import PaperBroker
from systematic_alpha.agent_lab.risk_engine import RiskEngine
from systematic_alpha.agent_lab.schemas import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_BLOCKED,
    PROPOSAL_STATUS_EXECUTED,
    STATUS_DATA_QUALITY_LOW,
    STATUS_INVALID_SIGNAL,
    STATUS_MARKET_CLOSED,
    STATUS_SIGNAL_OK,
)
from systematic_alpha.agent_lab.storage import AgentLabStorage
from systematic_alpha.agent_lab.strategy_registry import DEFAULT_STRATEGY_PARAMS, StrategyRegistry

PROMOTION_SCORE_THRESHOLD = 0.60
PROMOTION_MAX_RISK_VIOLATIONS = 3
PROMOTION_MAX_RISK_RATE = 0.15
PROMOTION_MIN_PROPOSALS = 10
FORCED_CONSERVATIVE_CONSECUTIVE_WEEKS = 3
FORCED_CONSERVATIVE_MIN_VIOLATIONS = 4
FORCED_CONSERVATIVE_MIN_RATE = 0.25


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truthy(value: str) -> bool:
    norm = str(value or "").strip().lower()
    return norm in {"1", "true", "yes", "y", "on"}


def parse_week_id(week_id: str) -> Tuple[date, date]:
    year_part, week_part = week_id.split("-W")
    year = int(year_part)
    week = int(week_part)
    start = date.fromisocalendar(year, week, 1)
    end = date.fromisocalendar(year, week, 7)
    return start, end


def _valid_usdkrw(rate: float) -> bool:
    # Conservative sanity band for USD/KRW.
    return 500.0 <= float(rate) <= 3000.0


class AgentLabOrchestrator:
    UNIFIED_EXECUTOR_AGENT_ID = "__unified_portfolio__"

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.state_root = self.project_root / "state" / "agent_lab"
        self.out_root = self.project_root / "out" / "agent_lab"
        self.db_path = self.state_root / "agent_lab.sqlite"
        self.storage = AgentLabStorage(self.db_path)
        self.registry = StrategyRegistry(self.storage)
        self.identity = AgentIdentityStore(self.state_root, self.storage)
        self.llm = LLMClient(self.storage)
        self.agent_engine = AgentDecisionEngine(self.llm, self.registry)
        self.accounting = AccountingEngine(self.storage)
        self.risk = RiskEngine()
        self.paper_broker = PaperBroker(self.storage)
        self.notifier = TelegramNotifier()
        self.kst = ZoneInfo("Asia/Seoul")
        self.et = ZoneInfo("America/New_York")
        self._ensure_epoch_id()

    @staticmethod
    def _execution_model() -> str:
        return str(os.getenv("AGENT_LAB_EXECUTION_MODEL", "legacy_multi_agent") or "legacy_multi_agent").strip().lower()

    @classmethod
    def _unified_shadow_mode(cls) -> bool:
        return cls._execution_model() == "unified_shadow"

    @staticmethod
    def _event_batch_enabled() -> bool:
        return _truthy(os.getenv("AGENT_LAB_EVENT_BATCH_ENABLED", "1"))

    @staticmethod
    def _sync_max_staleness_sec() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_MAX_STALENESS_SEC", "30") or 30)
        except Exception:
            raw = 30.0
        return max(5, int(raw))

    @staticmethod
    def _sync_alert_consecutive_threshold() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_ALERT_CONSECUTIVE", "2") or 2)
        except Exception:
            raw = 2.0
        return max(1, int(raw))

    @staticmethod
    def _sync_alert_cooldown_minutes() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_ALERT_COOLDOWN_MINUTES", "10") or 10)
        except Exception:
            raw = 10.0
        return max(0, int(raw))

    @staticmethod
    def _post_exec_sync_retries() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_POST_EXEC_SYNC_RETRIES", "2") or 2)
        except Exception:
            raw = 2.0
        return max(0, min(8, int(raw)))

    @staticmethod
    def _post_exec_sync_retry_delay_sec() -> float:
        try:
            raw = float(os.getenv("AGENT_LAB_POST_EXEC_SYNC_RETRY_DELAY_SEC", "1.5") or 1.5)
        except Exception:
            raw = 1.5
        return max(0.1, min(10.0, float(raw)))

    @staticmethod
    def _sync_precheck_retries() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_PRECHECK_RETRIES", "2") or 2)
        except Exception:
            raw = 2.0
        return max(0, min(8, int(raw)))

    @staticmethod
    def _sync_precheck_retry_delay_sec() -> float:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_PRECHECK_RETRY_DELAY_SEC", "1.0") or 1.0)
        except Exception:
            raw = 1.0
        return max(0.0, min(10.0, float(raw)))

    @staticmethod
    def _sync_fail_open_on_transient() -> bool:
        return _truthy(os.getenv("AGENT_LAB_SYNC_FAIL_OPEN_ON_TRANSIENT", "1"))

    @staticmethod
    def _sync_fail_open_grace_sec() -> int:
        try:
            raw = float(os.getenv("AGENT_LAB_SYNC_FAIL_OPEN_GRACE_SEC", "20") or 20)
        except Exception:
            raw = 20.0
        return max(1, min(120, int(raw)))

    @staticmethod
    def _strict_report_windows_enabled() -> bool:
        return _truthy(os.getenv("AGENT_LAB_STRICT_REPORT_WINDOWS", "1"))

    @staticmethod
    def _allow_offschedule_weekly() -> bool:
        return _truthy(os.getenv("AGENT_LAB_ALLOW_OFFSCHEDULE_WEEKLY", "0"))

    @staticmethod
    def _weekly_apply_mode() -> str:
        # Default policy: apply weekly council outputs immediately as active strategy.
        # Set AGENT_LAB_WEEKLY_APPLY_MODE=gated to restore legacy score-gated promotion.
        mode = str(os.getenv("AGENT_LAB_WEEKLY_APPLY_MODE", "immediate") or "immediate").strip().lower()
        if mode in {"gated", "legacy"}:
            return "gated"
        return "immediate"

    @staticmethod
    def _enforce_market_hours() -> bool:
        return _truthy(os.getenv("AGENT_LAB_ENFORCE_MARKET_HOURS", "1"))

    @staticmethod
    def _order_enabled_markets() -> List[str]:
        raw = str(os.getenv("AGENT_LAB_ORDER_ENABLED_MARKETS", "KR,US") or "KR,US")
        normalized = raw.replace(";", ",").replace("|", ",")
        out: List[str] = []
        for token in normalized.split(","):
            mk = str(token or "").strip().upper()
            if mk in {"KR", "US"} and mk not in out:
                out.append(mk)
        return out or ["KR", "US"]

    @classmethod
    def _is_order_market_enabled(cls, market: str) -> bool:
        mk = str(market or "").strip().upper()
        return mk in cls._order_enabled_markets()

    @classmethod
    def _execution_sync_scope(cls) -> str:
        """
        Sync scope used for execution gates.
        - If only one order-enabled market is active (e.g., KR only), sync that market only.
        - If both KR/US are enabled, keep ALL sync.
        """
        enabled = cls._order_enabled_markets()
        if len(enabled) == 1:
            return enabled[0]
        return "ALL"

    def _is_market_open_now(self, market: str, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        market_upper = str(market).strip().upper()
        if market_upper == "KR":
            now = now_utc.astimezone(self.kst) if now_utc else datetime.now(self.kst)
            if now.weekday() >= 5:
                return False, f"weekend_kst={now.isoformat(timespec='seconds')}"
            start = now.replace(hour=9, minute=0, second=0, microsecond=0)
            end = now.replace(hour=15, minute=35, second=0, microsecond=0)
            return (start <= now <= end), f"now_kst={now.isoformat(timespec='seconds')}, window=09:00~15:35"

        now = now_utc.astimezone(self.et) if now_utc else datetime.now(self.et)
        if now.weekday() >= 5:
            return False, f"weekend_et={now.isoformat(timespec='seconds')}"
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=5, second=0, microsecond=0)
        return (start <= now <= end), f"now_et={now.isoformat(timespec='seconds')}, window=09:30~16:05"

    def _session_date_now(self, market: str, now_utc: Optional[datetime] = None) -> str:
        market_upper = str(market).strip().upper()
        if market_upper == "US":
            now = now_utc.astimezone(self.et) if now_utc else datetime.now(self.et)
            return now.strftime("%Y%m%d")
        now = now_utc.astimezone(self.kst) if now_utc else datetime.now(self.kst)
        return now.strftime("%Y%m%d")

    def _event_already_reported(self, event_type: str, market: str, yyyymmdd: str, *, limit: int = 500) -> bool:
        market_upper = str(market).strip().upper()
        date_str = str(yyyymmdd).strip()
        rows = self.storage.list_events(event_type=event_type, limit=limit)
        for row in rows:
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("market", "")).strip().upper() != market_upper:
                continue
            if str(payload.get("date", "")).strip() != date_str:
                continue
            return True
        return False

    def _within_preopen_window(self, market: str, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        market_upper = str(market).strip().upper()
        now = now_utc.astimezone(self.kst if market_upper == "KR" else self.et) if now_utc else datetime.now(self.kst if market_upper == "KR" else self.et)
        if market_upper == "KR":
            if now.weekday() >= 5:
                return False, f"weekend_kst={now.isoformat(timespec='seconds')}"
            start = now.replace(hour=8, minute=40, second=0, microsecond=0)
            end = now.replace(hour=9, minute=5, second=0, microsecond=0)
            return (start <= now <= end), f"now_kst={now.isoformat(timespec='seconds')}, window=08:40~09:05"
        if now.weekday() >= 5:
            return False, f"weekend_et={now.isoformat(timespec='seconds')}"
        start = now.replace(hour=9, minute=5, second=0, microsecond=0)
        end = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return (start <= now <= end), f"now_et={now.isoformat(timespec='seconds')}, window=09:05~09:30"

    def _within_close_report_window(self, market: str, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        market_upper = str(market).strip().upper()
        now = now_utc.astimezone(self.kst if market_upper == "KR" else self.et) if now_utc else datetime.now(self.kst if market_upper == "KR" else self.et)
        if market_upper == "KR":
            if now.weekday() >= 5:
                return False, f"weekend_kst={now.isoformat(timespec='seconds')}"
            start = now.replace(hour=15, minute=35, second=0, microsecond=0)
            end = now.replace(hour=16, minute=10, second=0, microsecond=0)
            return (start <= now <= end), f"now_kst={now.isoformat(timespec='seconds')}, window=15:35~16:10"
        if now.weekday() >= 5:
            return False, f"weekend_et={now.isoformat(timespec='seconds')}"
        start = now.replace(hour=16, minute=5, second=0, microsecond=0)
        end = now.replace(hour=16, minute=40, second=0, microsecond=0)
        return (start <= now <= end), f"now_et={now.isoformat(timespec='seconds')}, window=16:05~16:40"

    def _within_weekly_window(self, now_kst: Optional[datetime] = None) -> Tuple[bool, str]:
        now = now_kst or datetime.now(self.kst)
        if now.weekday() != 6:
            return False, f"weekday={now.weekday()}, now_kst={now.isoformat(timespec='seconds')}, required=Sunday"
        start = now.replace(hour=7, minute=50, second=0, microsecond=0)
        end = now.replace(hour=8, minute=30, second=0, microsecond=0)
        return (start <= now <= end), f"now_kst={now.isoformat(timespec='seconds')}, window=07:50~08:30"

    def _ensure_epoch_id(self) -> str:
        existing = self.storage.get_system_meta("epoch_id", "")
        epoch = str(existing or "").strip()
        if epoch:
            return epoch
        epoch = datetime.now(self.kst).strftime("epoch_%Y%m%d_%H%M%S")
        ts = now_iso()
        self.storage.upsert_system_meta("epoch_id", epoch, ts)
        self.storage.upsert_system_meta("execution_model", self._execution_model(), ts)
        return epoch

    def _epoch_id(self) -> str:
        return str(self.storage.get_system_meta("epoch_id", "")) or self._ensure_epoch_id()

    def _reload_components(self) -> None:
        try:
            self.storage.close()
        except Exception:
            pass
        self.storage = AgentLabStorage(self.db_path)
        self.registry = StrategyRegistry(self.storage)
        self.identity = AgentIdentityStore(self.state_root, self.storage)
        self.llm = LLMClient(self.storage)
        self.agent_engine = AgentDecisionEngine(self.llm, self.registry)
        self.accounting = AccountingEngine(self.storage)
        self.risk = RiskEngine()
        self.paper_broker = PaperBroker(self.storage)
        self.notifier = TelegramNotifier()
        self._ensure_epoch_id()

    def close(self) -> None:
        self.storage.close()

    def _artifact_dir(self, yyyymmdd: str) -> Path:
        d = self.out_root / yyyymmdd
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_json_artifact(self, yyyymmdd: str, name: str, payload: Dict[str, Any]) -> Path:
        out = self._artifact_dir(yyyymmdd) / f"{name}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _write_text_artifact(self, yyyymmdd: str, name: str, text: str) -> Path:
        out = self._artifact_dir(yyyymmdd) / f"{name}.md"
        out.write_text(text, encoding="utf-8")
        return out

    def _append_activity_artifact(self, yyyymmdd: str, event_type: str, payload: Dict[str, Any]) -> None:
        path = self._artifact_dir(yyyymmdd) / "activity_log.jsonl"
        row = {
            "ts": now_iso(),
            "event_type": event_type,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _notify(self, text: str, *, event: str = "misc") -> None:
        key = str(event or "").strip().lower()
        immediate_events = {
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
        if self._event_batch_enabled() and key not in immediate_events:
            self.storage.insert_event_batch(
                batch_key="pending",
                events={"event": key, "text": str(text or ""), "queued_at": now_iso()},
                sent=False,
                created_at=now_iso(),
                sent_at=None,
            )
            return
        self.notifier.send(text, event=event)

    def _llm_explain(
        self,
        *,
        topic: str,
        context: Dict[str, Any],
        fallback: str,
        max_chars: int = 280,
    ) -> str:
        fallback_text = str(fallback or "").strip()
        response = self.llm.generate_json(
            system_prompt=(
                "당신은 트레이딩 운영 리포트 작성자다. "
                "입력 컨텍스트를 바탕으로 한국어 설명 1문장을 작성하고 JSON으로만 응답하라. "
                "키는 summary 하나만 사용한다."
            ),
            user_prompt=(
                f"topic={topic}\n"
                f"context={json.dumps(context, ensure_ascii=False)}\n"
                "설명은 간결하고 실행 가능한 관찰 위주로 작성하라."
            ),
            fallback={"summary": fallback_text},
            temperature=0.2,
        )
        result = response.get("result", {})
        if isinstance(result, dict):
            text = str(result.get("summary", "")).strip()
            if text:
                return text[:max_chars]
        return fallback_text[:max_chars]

    @staticmethod
    def _build_symbol_name_map_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        rows = []
        if isinstance(payload, dict):
            rows = list(payload.get("all_ranked") or payload.get("final") or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code", "")).strip().upper()
            name = str(row.get("name", "")).strip()
            if code and name:
                out[code] = name
        return out

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(value)))

    @staticmethod
    def _status_base_signal_strength(status_code: str) -> float:
        code = str(status_code or "").strip().upper()
        if code == STATUS_SIGNAL_OK:
            return 55.0
        if code == STATUS_DATA_QUALITY_LOW:
            return 30.0
        if code == STATUS_MARKET_CLOSED:
            return 20.0
        if code == STATUS_INVALID_SIGNAL:
            return 20.0
        return 50.0

    @classmethod
    def _signal_strength_from_payload(cls, payload: Dict[str, Any], topn: int = 12) -> float:
        rows = []
        if isinstance(payload, dict):
            rows = list(payload.get("all_ranked") or payload.get("final") or [])
        scores: List[float] = []
        for row in rows[: max(1, int(topn))]:
            if not isinstance(row, dict):
                continue
            score = cls._safe_float(row.get("recommendation_score"), -1.0)
            if score < 0:
                continue
            scores.append(score)
        if not scores:
            return 0.0
        return float(sum(scores) / float(len(scores)))

    def _load_recent_signal_snapshot(self, market: str, yyyymmdd: str, *, lookback_days: int) -> Dict[str, Any]:
        market_upper = str(market or "").strip().upper()
        max_lookback = max(1, int(lookback_days))
        try:
            run_date = datetime.strptime(str(yyyymmdd), "%Y%m%d").date()
        except Exception:
            run_date = datetime.now(self.kst).date()
        min_date = (run_date - timedelta(days=max_lookback)).strftime("%Y%m%d")
        max_date = run_date.strftime("%Y%m%d")
        row = self.storage.query_one(
            """
            SELECT *
            FROM session_signals
            WHERE market = ? AND session_date >= ? AND session_date <= ?
            ORDER BY session_date DESC, generated_at DESC, session_signal_id DESC
            LIMIT 1
            """,
            (market_upper, min_date, max_date),
        )
        if row is None:
            return {
                "market": market_upper,
                "session_date": "",
                "status_code": "NO_SIGNAL",
                "signal_strength": 50.0,
                "signal_age_days": None,
                "lookback_days": max_lookback,
            }
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except Exception:
            payload = {}
        status_code = str(row.get("status_code", "NO_SIGNAL") or "NO_SIGNAL").strip().upper()
        strength = self._status_base_signal_strength(status_code)
        payload_strength = self._signal_strength_from_payload(payload)
        if payload_strength > 0:
            # Keep a status prior while still reflecting ranked-signal quality.
            strength = 0.35 * strength + 0.65 * payload_strength
        strength = self._clamp(strength, 0.0, 100.0)
        signal_date = str(row.get("session_date", "") or "")
        age_days: Optional[int] = None
        try:
            age_days = max(0, (run_date - datetime.strptime(signal_date, "%Y%m%d").date()).days)
        except Exception:
            age_days = None
        return {
            "market": market_upper,
            "session_date": signal_date,
            "status_code": status_code,
            "signal_strength": float(strength),
            "signal_age_days": age_days,
            "lookback_days": max_lookback,
        }

    def _build_cross_market_plan(
        self,
        *,
        profile: Any,
        active_params: Dict[str, Any],
        market: str,
        yyyymmdd: str,
        ledger: Dict[str, Any],
    ) -> Dict[str, Any]:
        market_upper = str(market or "").strip().upper()
        peer_market = "US" if market_upper == "KR" else "KR"
        constraints = normalize_agent_constraints(getattr(profile, "constraints", {}), risk_style=getattr(profile, "risk_style", ""))
        cross_market = constraints.get("cross_market", {})
        if not isinstance(cross_market, dict):
            cross_market = {}

        base_weights = cross_market.get("base_weights", {})
        if not isinstance(base_weights, dict):
            base_weights = {}
        base_kr = self._safe_float(
            active_params.get("market_split_kr"),
            self._safe_float(base_weights.get("KR"), 0.50),
        )
        base_kr = self._clamp(base_kr, 0.05, 0.95)

        min_weight = self._safe_float(active_params.get("market_min_weight"), self._safe_float(cross_market.get("min_weight"), 0.20))
        max_weight = self._safe_float(active_params.get("market_max_weight"), self._safe_float(cross_market.get("max_weight"), 0.80))
        min_weight = self._clamp(min_weight, 0.05, 0.95)
        max_weight = self._clamp(max_weight, min_weight, 0.95)

        tilt_scale = self._safe_float(
            active_params.get("market_tilt_scale"),
            self._safe_float(cross_market.get("signal_tilt_scale"), 0.16),
        )
        tilt_scale = self._clamp(tilt_scale, 0.0, 0.50)
        lookback_days = int(
            self._clamp(
                self._safe_float(active_params.get("market_signal_lookback_days"), 5.0),
                1.0,
                15.0,
            )
        )
        enabled = bool(cross_market.get("enabled", True))

        signal_kr = self._load_recent_signal_snapshot("KR", yyyymmdd, lookback_days=lookback_days)
        signal_us = self._load_recent_signal_snapshot("US", yyyymmdd, lookback_days=lookback_days)
        signal_strength_kr = float(signal_kr.get("signal_strength", 50.0) or 50.0)
        signal_strength_us = float(signal_us.get("signal_strength", 50.0) or 50.0)
        if enabled:
            signal_gap_norm = (signal_strength_kr - signal_strength_us) / 100.0
            target_kr = base_kr + (tilt_scale * signal_gap_norm)
        else:
            target_kr = base_kr
        target_kr = self._clamp(target_kr, min_weight, max_weight)
        target_us = 1.0 - target_kr

        positions = list(ledger.get("positions") or [])
        exposure_by_market = {"KR": 0.0, "US": 0.0}
        for pos in positions:
            pos_market = str(pos.get("market", "")).strip().upper()
            if pos_market not in exposure_by_market:
                continue
            exposure_by_market[pos_market] += float(pos.get("market_value_krw", 0.0) or 0.0)
        cash_krw = max(0.0, float(ledger.get("cash_krw", 0.0) or 0.0))
        total_equity = cash_krw + exposure_by_market["KR"] + exposure_by_market["US"]
        if total_equity <= 0:
            total_equity = max(1.0, float(getattr(profile, "allocated_capital_krw", 1.0) or 1.0))

        target_weights = {"KR": float(target_kr), "US": float(target_us)}
        market_target_exposure = total_equity * target_weights[market_upper]
        market_current_exposure = float(exposure_by_market.get(market_upper, 0.0) or 0.0)
        market_raw_budget_cap = max(0.0, market_target_exposure - market_current_exposure)
        market_buy_budget_cap = min(cash_krw, market_raw_budget_cap)

        if _truthy(os.getenv("AGENT_LAB_MAX_FREEDOM", "1")):
            global_picks = int(float(os.getenv("AGENT_LAB_MAX_FREEDOM_PICKS", "20") or 20))
        else:
            global_picks = int(self._safe_float(active_params.get("max_daily_picks"), 20.0))
        global_picks = max(1, min(200, global_picks))
        picks = max(1, int(round(global_picks * target_weights[market_upper])))
        rebalance_gap_ratio = (market_target_exposure - market_current_exposure) / max(total_equity, 1.0)
        if rebalance_gap_ratio >= 0.15:
            picks += 2
        elif rebalance_gap_ratio <= -0.10:
            picks -= 2
        picks = max(1, min(global_picks, picks))

        signal_strength_market = signal_strength_kr if market_upper == "KR" else signal_strength_us
        signal_strength_peer = signal_strength_us if market_upper == "KR" else signal_strength_kr
        market_overweight = market_current_exposure > (market_target_exposure * 1.05)
        weakness = max(0.0, signal_strength_peer - signal_strength_market)
        min_recommendation_score = 0.0
        if weakness >= 8.0 or market_overweight:
            min_recommendation_score = 58.0 + 0.35 * weakness + (6.0 if market_overweight else 0.0)
            min_recommendation_score = self._clamp(min_recommendation_score, 55.0, 88.0)

        return {
            "enabled": enabled,
            "market": market_upper,
            "peer_market": peer_market,
            "signal_lookback_days": lookback_days,
            "base_weights": {"KR": float(base_kr), "US": float(1.0 - base_kr)},
            "target_weights": target_weights,
            "signal_strength": {"KR": signal_strength_kr, "US": signal_strength_us},
            "signal_status": {
                "KR": str(signal_kr.get("status_code", "NO_SIGNAL")),
                "US": str(signal_us.get("status_code", "NO_SIGNAL")),
            },
            "signal_session_date": {
                "KR": str(signal_kr.get("session_date", "")),
                "US": str(signal_us.get("session_date", "")),
            },
            "signal_age_days": {
                "KR": signal_kr.get("signal_age_days"),
                "US": signal_us.get("signal_age_days"),
            },
            "cash_krw": cash_krw,
            "total_equity_krw": total_equity,
            "exposure_by_market_krw": exposure_by_market,
            "market_target_exposure_krw": market_target_exposure,
            "market_current_exposure_krw": market_current_exposure,
            "buy_budget_cap_krw": market_buy_budget_cap,
            "max_picks_override": picks,
            "min_recommendation_score": float(min_recommendation_score),
        }

    @classmethod
    def _agent_signal_score(
        cls,
        agent_id: str,
        row: Dict[str, Any],
        *,
        session_date: str,
        index: int,
    ) -> float:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        code = str(row.get("code", "")).strip().upper()
        rec = cls._safe_float(row.get("recommendation_score"), 0.0)
        strength = cls._safe_float(metrics.get("strength_avg"), 0.0)
        vol = cls._safe_float(metrics.get("volume_ratio"), 0.0)
        bid_ask = cls._safe_float(metrics.get("bid_ask_avg"), 0.0)
        change = cls._safe_float(metrics.get("current_change_pct"), 0.0)
        gap = cls._safe_float(metrics.get("gap_pct"), 0.0)

        seed_text = f"{agent_id}:{session_date}:{code}:{index}"
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_text))
        jitter = ((seed % 997) / 997.0) - 0.5  # [-0.5, +0.5)

        if agent_id == "agent_a":
            return rec + 0.12 * strength + 4.0 * max(change, 0.0) + 2.0 * max(gap, 0.0) + 0.8 * vol
        if agent_id == "agent_b":
            downside = abs(min(change, 0.0)) + abs(min(gap, 0.0))
            return rec + 0.08 * strength + 0.5 * bid_ask + 0.4 * vol - 3.5 * downside
        # agent_c: diversify exploration with deterministic jitter.
        crowd_penalty = rec / 100.0
        return rec + 0.05 * strength + 1.0 * vol + 0.8 * bid_ask - 0.8 * crowd_penalty + 6.0 * jitter

    def _build_agent_specific_payload(
        self,
        *,
        base_payload: Dict[str, Any],
        agent_id: str,
        market: str,
        session_date: str,
    ) -> Dict[str, Any]:
        payload = copy.deepcopy(base_payload if isinstance(base_payload, dict) else {})
        rows = list(payload.get("all_ranked") or payload.get("final") or [])
        rows = [row for row in rows if isinstance(row, dict)]
        if not rows:
            return payload

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows, start=1):
            score = self._agent_signal_score(agent_id, row, session_date=session_date, index=idx)
            rank = int(row.get("rank", idx) or idx)
            scored.append((score, rank, row))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        reordered = [row for _, __, row in scored]

        if agent_id == "agent_c" and len(reordered) > 8:
            # Interleave head/mid/tail to widen exploration while keeping strong names.
            head = reordered[: max(6, len(reordered) // 3)]
            tail = reordered[max(6, len(reordered) // 3) :]
            mixed: List[Dict[str, Any]] = []
            while head or tail:
                if head:
                    mixed.append(head.pop(0))
                if tail:
                    mixed.append(tail.pop(0))
                if tail:
                    mixed.append(tail.pop(-1))
            reordered = mixed

        for idx, row in enumerate(reordered, start=1):
            row["rank"] = idx

        topn = int(float(os.getenv("AGENT_LAB_AGENT_SIGNAL_TOPN", "80") or 80))
        topn = max(3, min(len(reordered), topn))
        payload["all_ranked"] = reordered
        payload["final"] = reordered[:topn]
        payload["agent_signal_view"] = {
            "agent_id": agent_id,
            "market": str(market).upper(),
            "session_date": session_date,
            "base_ranked_count": len(rows),
            "view_ranked_count": len(reordered),
            "view_topn": topn,
            "mode": "agent_specific_reorder",
        }
        return payload

    def _load_symbol_name_map(self, market: str, yyyymmdd: str) -> Dict[str, str]:
        signal = self.storage.get_latest_session_signal(str(market).upper(), str(yyyymmdd))
        if not signal:
            return {}
        payload = signal.get("payload", {})
        if not isinstance(payload, dict):
            return {}
        return self._build_symbol_name_map_from_payload(payload)

    def _latest_proposal_by_agent_market(self, agent_id: str, market: str) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            """
            SELECT *
            FROM order_proposals
            WHERE agent_id = ? AND market = ?
            ORDER BY created_at DESC, proposal_id DESC
            LIMIT 1
            """,
            (str(agent_id), str(market).upper()),
        )
        if row is None:
            return None
        try:
            row["orders"] = json.loads(str(row.pop("orders_json") or "[]"))
        except Exception:
            row["orders"] = []
        return row

    def _proposal_order_results(self, proposal_id: int) -> List[Dict[str, Any]]:
        rows = self.storage.query_all(
            """
            SELECT
                paper_order_id,
                symbol,
                side,
                order_type,
                quantity,
                limit_price,
                reference_price,
                status,
                broker_order_id,
                submitted_at
            FROM paper_orders
            WHERE proposal_id = ?
            ORDER BY paper_order_id ASC
            """,
            (int(proposal_id),),
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "paper_order_id": int(row.get("paper_order_id")),
                    "symbol": str(row.get("symbol", "")),
                    "side": str(row.get("side", "")),
                    "order_type": str(row.get("order_type", "")),
                    "quantity": float(row.get("quantity", 0.0) or 0.0),
                    "limit_price": row.get("limit_price"),
                    "reference_price": float(row.get("reference_price", 0.0) or 0.0),
                    "status": str(row.get("status", "")),
                    "broker_order_id": str(row.get("broker_order_id", "")),
                    "submitted_at": str(row.get("submitted_at", "")),
                }
            )
        return out

    @staticmethod
    def _orders_to_text(
        orders: List[Dict[str, Any]],
        limit: int = 5,
        symbol_name_map: Optional[Dict[str, str]] = None,
    ) -> str:
        if not orders:
            return "없음"
        chunks: List[str] = []
        name_map = symbol_name_map or {}
        side_map = {
            "BUY": "매수",
            "SELL": "매도",
        }
        status_map = {
            "FILLED": "체결",
            "SUBMITTED": "접수",
            "REJECTED": "거부",
            "CANCELED": "취소",
            "BLOCKED": "차단",
            "EXECUTED": "실행",
            "APPROVED": "승인",
        }
        for row in orders[:limit]:
            side_raw = str(row.get("side", "")).strip().upper()
            side = side_map.get(side_raw, str(row.get("side", "")))
            symbol = str(row.get("symbol", "")).strip().upper()
            company_name = str(name_map.get(symbol, "")).strip()
            symbol_text = f"{symbol}({company_name})" if company_name else symbol
            qty = float(row.get("quantity", 0.0) or 0.0)
            status_raw = str(row.get("status", "")).strip().upper()
            status = status_map.get(status_raw, str(row.get("status", "")).strip())
            if status:
                chunks.append(f"{side} {symbol_text} x{qty:.0f} ({status})")
            else:
                chunks.append(f"{side} {symbol_text} x{qty:.0f}")
        if len(orders) > limit:
            chunks.append(f"... +{len(orders) - limit}")
        return ", ".join(chunks)

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def _sanitized_agent_constraints(risk_style: str = "") -> Dict[str, Any]:
        return normalize_agent_constraints({}, risk_style=risk_style)

    def _find_session_json(self, market: str, yyyymmdd: str) -> Optional[Path]:
        market_tag = market.strip().lower()
        result_dir = self.project_root / "out" / market_tag / yyyymmdd / "results"
        if not result_dir.exists():
            return None
        files = sorted(result_dir.glob(f"{market_tag}_daily_*.json"))
        if not files:
            return None
        return files[-1]

    def _classify_signal(self, payload: Dict[str, Any]) -> Tuple[str, str]:
        signal_valid = bool(payload.get("signal_valid", True))
        invalid_reason = str(payload.get("invalid_reason") or "")
        if signal_valid and (payload.get("final") or payload.get("all_ranked")):
            return STATUS_SIGNAL_OK, ""
        if invalid_reason.startswith("realtime_coverage_too_low"):
            rq = payload.get("realtime_quality") or {}
            cov = float(rq.get("coverage_ratio", 0.0) or 0.0)
            eligible = int(rq.get("eligible_count", 0) or 0)
            total = int(rq.get("total_count", 0) or 0)
            if cov <= 1e-6 and eligible == 0 and total > 0:
                return STATUS_MARKET_CLOSED, invalid_reason
            return STATUS_DATA_QUALITY_LOW, invalid_reason
        if invalid_reason:
            return STATUS_INVALID_SIGNAL, invalid_reason
        return STATUS_INVALID_SIGNAL, "empty_signal_payload"

    def _days_since_initialized(self, yyyymmdd: str) -> int:
        evt = self.storage.get_latest_event("lab_initialized")
        if not evt:
            return 10_000
        raw = evt.get("payload", {}).get("date", "")
        if not raw:
            return 10_000
        try:
            d0 = datetime.strptime(str(raw), "%Y%m%d").date()
            d1 = datetime.strptime(yyyymmdd, "%Y%m%d").date()
            return (d1 - d0).days
        except Exception:
            return 10_000

    def _risk_violation_count(self, agent_id: str, week_start: date, week_end: date) -> int:
        rows = self.storage.list_events(event_type="risk_violation", limit=5000)
        start_key = week_start.strftime("%Y%m%d")
        end_key = week_end.strftime("%Y%m%d")
        count = 0
        for row in rows:
            payload = row.get("payload", {})
            if str(payload.get("agent_id", "")) != agent_id:
                continue
            d = str(payload.get("session_date", ""))
            if not d:
                continue
            if start_key <= d <= end_key:
                count += 1
        return count

    def _proposal_count(self, agent_id: str, week_start: date, week_end: date) -> int:
        start_key = week_start.strftime("%Y%m%d")
        end_key = week_end.strftime("%Y%m%d")
        row = self.storage.query_one(
            """
            SELECT COUNT(*) AS c
            FROM order_proposals
            WHERE agent_id = ? AND session_date >= ? AND session_date <= ?
            """,
            (str(agent_id), start_key, end_key),
        )
        return int((row or {}).get("c", 0) or 0)

    def _risk_violation_stats(self, agent_id: str, week_start: date, week_end: date) -> Dict[str, float]:
        violations = self._risk_violation_count(agent_id, week_start, week_end)
        proposals = self._proposal_count(agent_id, week_start, week_end)
        rate = (float(violations) / float(proposals)) if proposals > 0 else 0.0
        return {
            "risk_violations": int(violations),
            "proposal_count": int(proposals),
            "risk_violation_rate": float(rate),
        }

    @staticmethod
    def _shift_week(week_start: date, weeks_back: int) -> Tuple[date, date]:
        start = week_start - timedelta(days=7 * int(weeks_back))
        end = start + timedelta(days=6)
        return start, end

    @staticmethod
    def _is_high_risk_week(stats: Dict[str, float]) -> bool:
        violations = int(stats.get("risk_violations", 0) or 0)
        proposals = int(stats.get("proposal_count", 0) or 0)
        rate = float(stats.get("risk_violation_rate", 0.0) or 0.0)
        if violations >= FORCED_CONSERVATIVE_MIN_VIOLATIONS:
            return True
        if proposals >= PROMOTION_MIN_PROPOSALS and rate >= FORCED_CONSERVATIVE_MIN_RATE:
            return True
        return False

    @staticmethod
    def _build_forced_conservative_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(base_params)

        def _f(key: str, default: float) -> float:
            try:
                return float(params.get(key, default) or default)
            except Exception:
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(float(params.get(key, default) or default))
            except Exception:
                return default

        params["risk_budget_ratio"] = min(_f("risk_budget_ratio", 1.0), 0.60)
        params["exposure_cap_ratio"] = min(_f("exposure_cap_ratio", 1.0), 0.70)
        params["position_cap_ratio"] = min(_f("position_cap_ratio", 1.0), 0.33)
        params["max_daily_picks"] = max(1, min(_i("max_daily_picks", 20), 5))
        params["min_strength"] = max(_f("min_strength", 0.0), 120.0)
        params["min_vol_ratio"] = max(_f("min_vol_ratio", 0.0), 0.20)
        params["min_bid_ask_ratio"] = max(_f("min_bid_ask_ratio", 0.0), 1.10)
        params["min_pass_conditions"] = max(_i("min_pass_conditions", 1), 3)
        params["day_loss_limit"] = max(_f("day_loss_limit", -1.0), -0.02)
        params["week_loss_limit"] = max(_f("week_loss_limit", -1.0), -0.05)
        return params

    def _usdkrw_rate(self, yyyymmdd: str) -> Optional[float]:
        default_rate = float(os.getenv("AGENT_LAB_USDKRW_DEFAULT", "1300") or 1300.0)
        latest = self.storage.get_latest_event("fx_rate_usdkrw")
        run_date = datetime.strptime(yyyymmdd, "%Y%m%d").date()
        use_live_fx = _truthy(os.getenv("AGENT_LAB_USE_LIVE_FX", "1"))

        def _extract_cached_rate(row: Optional[Dict[str, Any]], *, max_age_days: int) -> Optional[float]:
            if not row:
                return None
            payload = row.get("payload", {})
            try:
                rate = float(payload.get("rate"))
                if not _valid_usdkrw(rate):
                    return None
                d0 = datetime.fromisoformat(str(row["created_at"])).date()
                if (run_date - d0).days <= max_age_days:
                    return rate
            except Exception:
                return None
            return None

        cached_today = _extract_cached_rate(latest, max_age_days=0)
        latest_source = ""
        if latest:
            latest_source = str((latest.get("payload", {}) or {}).get("source", "")).strip().lower()

        # If today's cache is already from live/manual source, trust it.
        if cached_today is not None and (not use_live_fx or latest_source.startswith("live_") or latest_source == "manual"):
            return cached_today

        if use_live_fx:
            live = self._fetch_live_usdkrw_rate()
            if live is not None:
                rate, source = live
                self.storage.log_event(
                    event_type="fx_rate_usdkrw",
                    payload={"rate": rate, "source": source, "date": yyyymmdd},
                    created_at=now_iso(),
                )
                return rate
            self.storage.log_event(
                event_type="fx_rate_usdkrw_fetch_failed",
                payload={"date": yyyymmdd, "fallback": "cache_or_env_default"},
                created_at=now_iso(),
            )
            if cached_today is not None:
                return cached_today

        cached_recent = _extract_cached_rate(latest, max_age_days=3)
        if cached_recent is not None:
            return cached_recent

        self.storage.log_event(
            event_type="fx_rate_usdkrw",
            payload={"rate": default_rate, "source": "env_default", "date": yyyymmdd},
            created_at=now_iso(),
        )
        return default_rate

    def _fetch_live_usdkrw_rate(self) -> Optional[Tuple[float, str]]:
        # Primary + fallback public FX endpoints.
        providers: List[Tuple[str, str]] = [
            ("open_er_api", "https://open.er-api.com/v6/latest/USD"),
            ("frankfurter", "https://api.frankfurter.app/latest?from=USD&to=KRW"),
        ]
        timeout_sec = float(os.getenv("AGENT_LAB_FX_TIMEOUT_SECONDS", "8") or 8.0)

        for name, url in providers:
            try:
                resp = requests.get(url, timeout=timeout_sec)
                resp.raise_for_status()
                data = resp.json()
                rate: Optional[float] = None
                if name == "open_er_api":
                    rates = data.get("rates", {}) if isinstance(data, dict) else {}
                    rate = float(rates.get("KRW"))
                elif name == "frankfurter":
                    rates = data.get("rates", {}) if isinstance(data, dict) else {}
                    rate = float(rates.get("KRW"))
                if rate is not None and _valid_usdkrw(rate):
                    return rate, f"live_{name}"
            except Exception:
                continue
        return None

    def _ensure_unified_executor_agent(self) -> None:
        if not self._unified_shadow_mode():
            return
        aid = self.UNIFIED_EXECUTOR_AGENT_ID
        row = self.storage.query_one("SELECT agent_id FROM agents WHERE agent_id = ? LIMIT 1", (aid,))
        if row is None:
            self.storage.upsert_agent(
                agent_id=aid,
                name="Unified Portfolio Executor",
                role="system_executor",
                philosophy="Single account execution rail for unified shadow mode.",
                allocated_capital_krw=0.0,
                risk_style="system",
                constraints={},
                created_at=now_iso(),
                is_active=0,
            )
        active = self.storage.get_active_strategy(aid)
        if active is None:
            self.storage.insert_strategy_version(
                agent_id=aid,
                version_tag="v1.0.0",
                params=copy.deepcopy(DEFAULT_STRATEGY_PARAMS),
                promoted=True,
                notes="bootstrap unified executor strategy",
                created_at=now_iso(),
            )

    @staticmethod
    def _position_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for row in rows:
            market = str(row.get("market", "")).strip().upper()
            symbol = str(row.get("symbol", "")).strip().upper()
            qty = float(row.get("quantity", 0.0) or 0.0)
            if not market or not symbol:
                continue
            out[(market, symbol)] = out.get((market, symbol), 0.0) + qty
        return out

    @staticmethod
    def _broker_response_has_api_error(raw: Any) -> bool:
        payload: Dict[str, Any] = {}
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str):
            text = str(raw or "").strip()
            if not text:
                return False
            try:
                decoded = json.loads(text)
            except Exception:
                return False
            if isinstance(decoded, dict):
                payload = decoded
        if not payload:
            return False
        if payload.get("ok") is False:
            return True
        candidates: List[Dict[str, Any]] = []
        resp = payload.get("response")
        if isinstance(resp, dict):
            candidates.append(resp)
        candidates.append(payload)
        for cand in candidates:
            rt_cd = str(cand.get("rt_cd", "")).strip()
            if rt_cd and rt_cd != "0":
                return True
        return False

    def _local_positions_from_fills(self, market_scope: str = "ALL") -> List[Dict[str, Any]]:
        where_market = ""
        params: List[Any] = []
        if str(market_scope).upper() in {"KR", "US"}:
            where_market = " AND po.market = ? "
            params.append(str(market_scope).upper())
        if self._unified_shadow_mode():
            where_agent = " AND po.agent_id = ? "
            params.append(self.UNIFIED_EXECUTOR_AGENT_ID)
        else:
            where_agent = ""
        rows = self.storage.query_all(
            f"""
            SELECT po.market, po.symbol, po.side, pf.fill_quantity, pf.fill_price, pf.fx_rate, po.broker_response_json
            FROM paper_fills pf
            JOIN paper_orders po ON po.paper_order_id = pf.paper_order_id
            WHERE 1=1
            {where_market}
            {where_agent}
            ORDER BY pf.filled_at ASC, pf.paper_fill_id ASC
            """,
            tuple(params),
        )
        positions: Dict[Tuple[str, str], Dict[str, float]] = {}
        for row in rows:
            if self._broker_response_has_api_error(row.get("broker_response_json")):
                continue
            market = str(row.get("market", "")).strip().upper()
            symbol = str(row.get("symbol", "")).strip().upper()
            side = str(row.get("side", "")).strip().upper()
            qty = float(row.get("fill_quantity", 0.0) or 0.0)
            fx = float(row.get("fx_rate", 1.0) or 1.0)
            price = float(row.get("fill_price", 0.0) or 0.0) * fx
            if not market or not symbol or qty <= 0:
                continue
            key = (market, symbol)
            st = positions.get(key, {"quantity": 0.0, "avg_price": 0.0})
            if side == "BUY":
                total = (st["quantity"] * st["avg_price"]) + (qty * price)
                new_qty = st["quantity"] + qty
                st["quantity"] = new_qty
                st["avg_price"] = (total / new_qty) if new_qty > 0 else st["avg_price"]
            elif side == "SELL":
                st["quantity"] = max(0.0, st["quantity"] - qty)
            positions[key] = st
        out: List[Dict[str, Any]] = []
        for (market, symbol), st in positions.items():
            qty = float(st.get("quantity", 0.0) or 0.0)
            if qty <= 0:
                continue
            avg = float(st.get("avg_price", 0.0) or 0.0)
            out.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "quantity": qty,
                    "avg_price": avg,
                    "market_value_krw": qty * avg,
                }
            )
        return out

    @staticmethod
    def _decode_json_obj(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = str(raw).strip()
            if not text:
                return {}
            try:
                decoded = json.loads(text)
            except Exception:
                return {}
            if isinstance(decoded, dict):
                return decoded
        return {}

    def _list_submitted_paper_orders(self, market_scope: str = "ALL") -> List[Dict[str, Any]]:
        where_market = ""
        params: List[Any] = ["SUBMITTED"]
        scope = str(market_scope or "ALL").strip().upper()
        if scope in {"KR", "US"}:
            where_market = " AND market = ? "
            params.append(scope)
        where_agent = ""
        if self._unified_shadow_mode():
            where_agent = " AND agent_id = ? "
            params.append(self.UNIFIED_EXECUTOR_AGENT_ID)
        return self.storage.query_all(
            f"""
            SELECT
                paper_order_id,
                market,
                symbol,
                side,
                order_type,
                quantity,
                limit_price,
                reference_price,
                broker_order_id,
                broker_response_json,
                submitted_at
            FROM paper_orders
            WHERE status = ?
            {where_market}
            {where_agent}
            ORDER BY submitted_at ASC, paper_order_id ASC
            """,
            tuple(params),
        )

    @staticmethod
    def _submitted_delta_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for row in rows:
            market = str(row.get("market", "")).strip().upper()
            symbol = str(row.get("symbol", "")).strip().upper()
            side = str(row.get("side", "")).strip().upper()
            qty = float(row.get("quantity", 0.0) or 0.0)
            if not market or not symbol or qty <= 0:
                continue
            signed = qty if side == "BUY" else (-qty if side == "SELL" else 0.0)
            out[(market, symbol)] = out.get((market, symbol), 0.0) + signed
        return out

    def _settle_submitted_orders_from_server(
        self,
        *,
        market_scope: str,
        server_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        submitted = self._list_submitted_paper_orders(market_scope)
        if not submitted:
            return {
                "submitted_count": 0,
                "filled_count": 0,
                "pending_count": 0,
                "filled_orders": [],
            }

        server_map = self._position_map(server_positions)
        local_rows = self._local_positions_from_fills(market_scope)
        local_map = self._position_map(local_rows)
        open_orders_snapshot = self.paper_broker.fetch_open_orders_snapshot(market_scope)
        open_lookup_ok: Dict[str, bool] = {}
        open_order_ids_by_market: Dict[str, set[str]] = {}
        open_lookup_summary: Dict[str, Any] = {}
        markets_payload = open_orders_snapshot.get("markets", {})
        if isinstance(markets_payload, dict):
            for mk, payload in markets_payload.items():
                mk_upper = str(mk or "").strip().upper()
                if mk_upper not in {"KR", "US"}:
                    continue
                payload_obj = payload if isinstance(payload, dict) else {}
                mk_ok = bool(payload_obj.get("ok", False))
                open_lookup_ok[mk_upper] = mk_ok
                ids: set[str] = set()
                for item in list(payload_obj.get("open_orders", []) or []):
                    if not isinstance(item, dict):
                        continue
                    odno = str(item.get("broker_order_id", "")).strip()
                    if odno:
                        ids.add(odno)
                open_order_ids_by_market[mk_upper] = ids
                open_lookup_summary[mk_upper] = {
                    "ok": mk_ok,
                    "open_count": len(ids),
                    "error_count": len(list(payload_obj.get("errors", []) or [])),
                    "best_effort": bool(payload_obj.get("best_effort", False)),
                }
        odno_close_grace_sec = self._submitted_odno_close_grace_seconds()
        odno_priority_enabled = self._env_bool("AGENT_LAB_ODNO_STATUS_PRIORITY", True)
        filled_orders: List[Dict[str, Any]] = []
        filled_order_ids: set[int] = set()
        closed_orders: List[Dict[str, Any]] = []
        closed_order_ids: set[int] = set()

        for row in submitted:
            market = str(row.get("market", "")).strip().upper()
            symbol = str(row.get("symbol", "")).strip().upper()
            side = str(row.get("side", "")).strip().upper()
            order_type = str(row.get("order_type", "MARKET")).strip().upper()
            qty = float(row.get("quantity", 0.0) or 0.0)
            if qty <= 0 or not market or not symbol:
                continue
            paper_order_id = int(row.get("paper_order_id"))
            broker_order_id = str(row.get("broker_order_id", "")).strip()
            age_sec = self._age_seconds_from_iso(str(row.get("submitted_at", "")))
            lookup_ok = bool(open_lookup_ok.get(market, False))
            odno_open: Optional[bool] = None
            if broker_order_id and lookup_ok:
                odno_open = broker_order_id in open_order_ids_by_market.get(market, set())
            key = (market, symbol)
            server_qty = float(server_map.get(key, 0.0) or 0.0)
            local_qty = float(local_map.get(key, 0.0) or 0.0)
            should_fill_delta = False
            if side == "BUY":
                should_fill_delta = (server_qty - local_qty) >= (qty - 1e-9)
            elif side == "SELL":
                should_fill_delta = (local_qty - server_qty) >= (qty - 1e-9)
            settlement_basis = "delta_fallback"

            # ODNO-first mode:
            # - ODNO=open   => keep SUBMITTED regardless of quantity delta.
            # - ODNO=closed => settle only when delta proves fill, else close as rejected after grace.
            if odno_priority_enabled and broker_order_id and lookup_ok:
                if odno_open is True:
                    continue
                if odno_open is False:
                    if not should_fill_delta:
                        if age_sec is not None and float(age_sec) >= float(odno_close_grace_sec):
                            broker_payload = self._decode_json_obj(row.get("broker_response_json"))
                            broker_payload["odno_settlement"] = {
                                "applied_at": now_iso(),
                                "from_status": "SUBMITTED",
                                "to_status": "REJECTED",
                                "reason": "odno_closed_without_fill_delta",
                                "odno": broker_order_id,
                                "age_sec": float(age_sec),
                                "odno_priority_enabled": True,
                            }
                            self.storage.update_paper_order_status_with_response(
                                paper_order_id=paper_order_id,
                                status="REJECTED",
                                broker_response_json=broker_payload,
                            )
                            closed_orders.append(
                                {
                                    "paper_order_id": paper_order_id,
                                    "market": market,
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": qty,
                                    "broker_order_id": broker_order_id,
                                    "age_sec": float(age_sec),
                                    "reason": "odno_closed_without_fill_delta",
                                }
                            )
                            closed_order_ids.add(paper_order_id)
                        continue
                    settlement_basis = "odno_closed+delta"
            elif not should_fill_delta:
                continue

            if not should_fill_delta:
                continue

            limit_price = row.get("limit_price")
            reference_price = float(row.get("reference_price", 0.0) or 0.0)
            fill_price = float(limit_price or reference_price) if order_type == "LIMIT" else reference_price
            if fill_price <= 0:
                fill_price = max(0.0, reference_price)
            broker_payload = self._decode_json_obj(row.get("broker_response_json"))
            fx_rate = float(
                broker_payload.get("fx_rate", broker_payload.get("requested_fx_rate", 1.0)) or 1.0
            )
            if fx_rate <= 0:
                fx_rate = 1.0
            fill_value_krw = float(fill_price) * float(qty) * float(fx_rate)
            self.storage.insert_paper_fill(
                paper_order_id=paper_order_id,
                fill_price=float(fill_price),
                fill_quantity=float(qty),
                fill_value_krw=float(fill_value_krw),
                fx_rate=float(fx_rate),
                filled_at=now_iso(),
            )
            self.storage.update_paper_order_status(paper_order_id=paper_order_id, status="FILLED")
            if side == "BUY":
                local_map[key] = local_qty + qty
            elif side == "SELL":
                local_map[key] = max(0.0, local_qty - qty)
            filled_orders.append(
                {
                    "paper_order_id": paper_order_id,
                    "market": market,
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "broker_order_id": broker_order_id,
                    "settlement_basis": settlement_basis,
                }
            )
            filled_order_ids.add(int(paper_order_id))

        pending_orders: List[Dict[str, Any]] = []
        now_iso_ts = now_iso()
        for row in submitted:
            paper_order_id = int(row.get("paper_order_id"))
            if paper_order_id in filled_order_ids or paper_order_id in closed_order_ids:
                continue
            market = str(row.get("market", "")).strip().upper()
            broker_order_id = str(row.get("broker_order_id", "")).strip()
            lookup_ok = bool(open_lookup_ok.get(market, False))
            odno_open: Optional[bool] = None
            if broker_order_id and lookup_ok:
                odno_open = broker_order_id in open_order_ids_by_market.get(market, set())
            age_sec = self._age_seconds_from_iso(str(row.get("submitted_at", "")))
            pending_orders.append(
                {
                    "paper_order_id": paper_order_id,
                    "broker_order_id": broker_order_id,
                    "market": market,
                    "symbol": str(row.get("symbol", "")).strip().upper(),
                    "side": str(row.get("side", "")).strip().upper(),
                    "quantity": float(row.get("quantity", 0.0) or 0.0),
                    "submitted_at": str(row.get("submitted_at", "")),
                    "age_sec": None if age_sec is None else float(age_sec),
                    "observed_at": now_iso_ts,
                    "odno_lookup_ok": lookup_ok,
                    "odno_open": odno_open,
                }
            )

        out = {
            "submitted_count": len(submitted),
            "filled_count": len(filled_orders),
            "closed_count": len(closed_orders),
            "pending_count": max(0, len(submitted) - len(filled_orders) - len(closed_orders)),
            "filled_orders": filled_orders,
            "closed_orders": closed_orders,
            "pending_orders": pending_orders,
            "open_order_lookup": {
                "ok": bool(open_orders_snapshot.get("ok", False)),
                "markets": open_lookup_summary,
                "errors": list(open_orders_snapshot.get("errors", []) or []),
            },
        }
        if filled_orders:
            self.storage.log_event(
                "order_settlement_confirmed",
                {
                    "market_scope": str(market_scope or "ALL").strip().upper(),
                    "filled_count": len(filled_orders),
                    "pending_count": out["pending_count"],
                    "filled_orders": filled_orders,
                },
                now_iso(),
            )
        if closed_orders:
            self.storage.log_event(
                "order_settlement_closed",
                {
                    "market_scope": str(market_scope or "ALL").strip().upper(),
                    "closed_count": len(closed_orders),
                    "closed_orders": closed_orders[:50],
                    "odno_close_grace_sec": int(odno_close_grace_sec),
                },
                now_iso(),
            )
        return out

    @staticmethod
    def _submitted_alert_stale_seconds() -> int:
        try:
            value = int(float(os.getenv("AGENT_LAB_SUBMITTED_STALE_ALERT_SEC", "180") or 180))
        except Exception:
            value = 180
        return max(30, min(3600, value))

    @staticmethod
    def _submitted_alert_cooldown_minutes() -> int:
        try:
            value = int(float(os.getenv("AGENT_LAB_SUBMITTED_ALERT_COOLDOWN_MINUTES", "5") or 5))
        except Exception:
            value = 5
        return max(1, min(240, value))

    @staticmethod
    def _submitted_odno_close_grace_seconds() -> int:
        try:
            value = int(float(os.getenv("AGENT_LAB_SUBMITTED_ODNO_CLOSE_GRACE_SEC", "300") or 300))
        except Exception:
            value = 300
        return max(30, min(7200, value))

    @staticmethod
    def _submitted_alert_ts_meta_key(market_scope: str) -> str:
        return f"submitted_alert_last_sent:{str(market_scope or 'ALL').strip().upper() or 'ALL'}"

    @staticmethod
    def _submitted_alert_sig_meta_key(market_scope: str) -> str:
        return f"submitted_alert_last_sig:{str(market_scope or 'ALL').strip().upper() or 'ALL'}"

    def _allow_submitted_alert_notification(self, market_scope: str, signature: str) -> bool:
        cooldown_min = self._submitted_alert_cooldown_minutes()
        ts_key = self._submitted_alert_ts_meta_key(market_scope)
        sig_key = self._submitted_alert_sig_meta_key(market_scope)
        now_s = now_iso()
        last_sig = str(self.storage.get_system_meta(sig_key, "") or "")
        last_sent_raw = str(self.storage.get_system_meta(ts_key, "") or "")
        if signature != last_sig:
            self.storage.upsert_system_meta(ts_key, now_s, now_s)
            self.storage.upsert_system_meta(sig_key, signature, now_s)
            return True
        if not last_sent_raw:
            self.storage.upsert_system_meta(ts_key, now_s, now_s)
            self.storage.upsert_system_meta(sig_key, signature, now_s)
            return True
        try:
            last_sent = datetime.fromisoformat(last_sent_raw)
            elapsed = datetime.now() - last_sent
            if elapsed < timedelta(minutes=cooldown_min):
                return False
        except Exception:
            pass
        self.storage.upsert_system_meta(ts_key, now_s, now_s)
        self.storage.upsert_system_meta(sig_key, signature, now_s)
        return True

    def _maybe_alert_stale_submitted_orders(self, *, market_scope: str, settlement: Dict[str, Any]) -> None:
        pending = list((settlement or {}).get("pending_orders") or [])
        if not pending:
            return
        stale_sec = self._submitted_alert_stale_seconds()
        stale_rows: List[Dict[str, Any]] = []
        for row in pending:
            if not isinstance(row, dict):
                continue
            age = row.get("age_sec")
            try:
                age_f = float(age)
            except Exception:
                age_f = -1.0
            if age_f >= float(stale_sec):
                stale_rows.append({**row, "age_sec": age_f})
        if not stale_rows:
            return
        stale_rows.sort(key=lambda r: float(r.get("age_sec", 0.0) or 0.0), reverse=True)
        sig_parts: List[str] = []
        for row in stale_rows:
            sig_parts.append(
                f"{int(row.get('paper_order_id', 0))}:"
                f"{str(row.get('broker_order_id', '')).strip()}:"
                f"{str(row.get('market', '')).strip().upper()}:"
                f"{str(row.get('symbol', '')).strip().upper()}:"
                f"{str(row.get('side', '')).strip().upper()}:"
                f"{float(row.get('quantity', 0.0) or 0.0):g}"
            )
        signature = "|".join(sig_parts)
        if not self._allow_submitted_alert_notification(market_scope, signature):
            return
        lines = [
            "[AgentLab] 미체결 주문 장기지연",
            f"시장범위={str(market_scope or 'ALL').strip().upper()}",
            f"기준_초={stale_sec}",
            f"건수={len(stale_rows)}",
        ]
        for row in stale_rows[:5]:
            lines.append(
                "주문="
                f"paper_order_id={int(row.get('paper_order_id', 0))}, "
                f"broker_order_id={str(row.get('broker_order_id', '') or '-').strip() or '-'}, "
                f"{str(row.get('side', '')).strip().upper()} {str(row.get('symbol', '')).strip().upper()} x{float(row.get('quantity', 0.0) or 0.0):g}, "
                f"경과={int(float(row.get('age_sec', 0.0) or 0.0))}s"
            )
        lines.append("조치=브로커 체결/정정취소 상태 확인 필요")
        self.storage.log_event(
            "submitted_orders_stale_alert",
            {
                "market_scope": str(market_scope or "ALL").strip().upper(),
                "threshold_sec": int(stale_sec),
                "count": int(len(stale_rows)),
                "orders": stale_rows[:10],
            },
            now_iso(),
        )
        self._notify("\n".join(lines), event="broker_api_error")

    def _is_pending_settlement_mismatch(self, detail: Dict[str, Any], market_scope: str) -> bool:
        mismatches = list((detail or {}).get("mismatches") or [])
        if not mismatches:
            return False
        submitted = self._list_submitted_paper_orders(market_scope)
        if not submitted:
            return False
        pending = self._submitted_delta_map(submitted)
        if not pending:
            return False
        for item in mismatches:
            if not isinstance(item, dict):
                return False
            market = str(item.get("market", "")).strip().upper()
            symbol = str(item.get("symbol", "")).strip().upper()
            key = (market, symbol)
            if key not in pending:
                return False
            try:
                delta = float(item.get("delta_qty", 0.0) or 0.0)
            except Exception:
                return False
            pending_signed = float(pending.get(key, 0.0) or 0.0)
            # delta is server_qty - local_qty.
            # pending buy => expected positive delta; pending sell => expected negative delta.
            if pending_signed > 0:
                if delta < -1e-9 or abs(delta) > abs(pending_signed) + 1e-9:
                    return False
            elif pending_signed < 0:
                if delta > 1e-9 or abs(delta) > abs(pending_signed) + 1e-9:
                    return False
            else:
                return False
        return True

    @staticmethod
    def _age_seconds_from_iso(value: str) -> Optional[float]:
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
        now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
        return max(0.0, float((now - parsed).total_seconds()))

    @staticmethod
    def _sync_streak_meta_key(market_scope: str) -> str:
        return f"sync_mismatch_streak:{str(market_scope or 'ALL').strip().upper() or 'ALL'}"

    @staticmethod
    def _sync_alert_ts_meta_key(market_scope: str) -> str:
        return f"sync_alert_last_sent:{str(market_scope or 'ALL').strip().upper() or 'ALL'}"

    @staticmethod
    def _sync_alert_sig_meta_key(market_scope: str) -> str:
        return f"sync_alert_last_sig:{str(market_scope or 'ALL').strip().upper() or 'ALL'}"

    @staticmethod
    def _sync_alert_signature(payload: Dict[str, Any]) -> str:
        detail = payload.get("detail")
        mismatches = (detail or {}).get("mismatches") if isinstance(detail, dict) else None
        if isinstance(mismatches, list) and mismatches:
            parts: List[str] = []
            for item in mismatches:
                if not isinstance(item, dict):
                    continue
                market = str(item.get("market", "")).strip().upper()
                symbol = str(item.get("symbol", "")).strip().upper()
                try:
                    delta = float(item.get("delta_qty", 0.0) or 0.0)
                except Exception:
                    delta = 0.0
                parts.append(f"{market}:{symbol}:{delta:.6f}")
            if parts:
                return "|".join(sorted(parts))
        reason = str(payload.get("reason", "") or "").strip().lower()
        errors = payload.get("errors")
        if isinstance(errors, list):
            err = "|".join(str(x) for x in errors[:2])
        else:
            err = str(errors or "")
        return f"reason={reason}|errors={err}"

    def _allow_sync_alert_notification(self, market_scope: str, payload: Dict[str, Any]) -> bool:
        cooldown_min = self._sync_alert_cooldown_minutes()
        if cooldown_min <= 0:
            return True
        ts_key = self._sync_alert_ts_meta_key(market_scope)
        sig_key = self._sync_alert_sig_meta_key(market_scope)
        now = datetime.now()
        now_s = now_iso()
        signature = self._sync_alert_signature(payload)
        last_sig = str(self.storage.get_system_meta(sig_key, "") or "")
        last_sent_raw = str(self.storage.get_system_meta(ts_key, "") or "")
        send = False
        if signature != last_sig:
            send = True
        else:
            if not last_sent_raw:
                send = True
            else:
                try:
                    last_sent = datetime.fromisoformat(last_sent_raw)
                    elapsed = now - last_sent
                    send = elapsed >= timedelta(minutes=cooldown_min)
                except Exception:
                    send = True
        if send:
            self.storage.upsert_system_meta(ts_key, now_s, now_s)
            self.storage.upsert_system_meta(sig_key, signature, now_s)
        return send

    def _get_sync_mismatch_streak(self, market_scope: str) -> int:
        key = self._sync_streak_meta_key(market_scope)
        raw = str(self.storage.get_system_meta(key, "0") or "0")
        try:
            return max(0, int(float(raw)))
        except Exception:
            return 0

    def _set_sync_mismatch_streak(self, market_scope: str, value: int) -> None:
        key = self._sync_streak_meta_key(market_scope)
        self.storage.upsert_system_meta(key, str(max(0, int(value))), now_iso())

    def _inc_sync_mismatch_streak(self, market_scope: str) -> int:
        cur = self._get_sync_mismatch_streak(market_scope)
        nxt = cur + 1
        self._set_sync_mismatch_streak(market_scope, nxt)
        return nxt

    def _sync_after_execution_with_retry(self, market_scope: str = "ALL") -> Dict[str, Any]:
        retries = self._post_exec_sync_retries()
        delay_sec = self._post_exec_sync_retry_delay_sec()
        attempts = retries + 1
        last: Dict[str, Any] = {}
        for idx in range(attempts):
            last = self.sync_account(
                market=market_scope,
                strict=False,
                notify=False,
                track_streak=False,
            )
            if not bool(last.get("blocked", False)):
                final_ok = self.sync_account(
                    market=market_scope,
                    strict=False,
                    notify=False,
                    track_streak=True,
                )
                final_ok["post_exec_retry_attempts"] = int(idx + 1)
                final_ok["post_exec_retry_used"] = bool(idx > 0)
                return final_ok
            if idx < (attempts - 1):
                time.sleep(delay_sec)
        final_fail = self.sync_account(
            market=market_scope,
            strict=False,
            notify=True,
            track_streak=True,
        )
        final_fail["post_exec_retry_attempts"] = int(attempts)
        final_fail["post_exec_retry_used"] = bool(retries > 0)
        final_fail["post_exec_retry_failed"] = True
        return final_fail

    @staticmethod
    def _is_transient_sync_error_text(text: str) -> bool:
        raw = str(text or "")
        norm = raw.lower()
        keys = (
            "keyerror('tr_cont')",
            'keyerror("tr_cont")',
            "remote end closed connection without response",
            "remotedisconnected",
            "connection aborted",
            "max retries exceeded",
            "failed to resolve",
            "name or service not known",
            "temporary failure in name resolution",
            "read timed out",
            "connect timeout",
            "connection reset by peer",
            "sslerror",
            "protocolerror",
        )
        return any(k in norm for k in keys)

    def _is_transient_sync_failure(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        if str(payload.get("reason", "")).strip().lower() != "broker_fetch_failed":
            return False
        errors = payload.get("errors")
        if not isinstance(errors, list):
            return False
        return any(self._is_transient_sync_error_text(str(x)) for x in errors)

    def _recent_strict_sync_success(self, market_scope: str = "ALL", *, max_age_sec: int = 20) -> bool:
        scope = str(market_scope or "ALL").strip().upper() or "ALL"
        rows = self.storage.list_events(event_type="account_synced", limit=50)
        for row in rows:
            payload = row.get("payload", {}) if isinstance(row, dict) else {}
            if not isinstance(payload, dict):
                continue
            row_scope = str(payload.get("market_scope", "")).strip().upper()
            if row_scope and row_scope != scope:
                continue
            if not bool(payload.get("strict", False)):
                continue
            if bool(payload.get("blocked", False)):
                continue
            if not bool(payload.get("ok", False)):
                continue
            if not bool(payload.get("matched", False)):
                continue
            age = self._age_seconds_from_iso(str(row.get("created_at", "")))
            if age is None:
                age = self._age_seconds_from_iso(str(payload.get("fetched_at", "")))
            if age is not None and float(age) <= float(max_age_sec):
                return True
        return False

    def _sync_account_strict_with_retry(
        self,
        *,
        market_scope: str = "ALL",
        notify_on_fail: bool = True,
        track_streak_on_fail: bool = True,
    ) -> Dict[str, Any]:
        retries = self._sync_precheck_retries()
        delay_sec = self._sync_precheck_retry_delay_sec()
        attempts = retries + 1
        last: Dict[str, Any] = {}
        for idx in range(attempts):
            last = self.sync_account(
                market=market_scope,
                strict=True,
                notify=False,
                track_streak=False,
            )
            blocked = bool(last.get("blocked", False))
            if not blocked:
                self._set_sync_mismatch_streak(market_scope, 0)
                out = dict(last)
                out["sync_retry_attempts"] = int(idx + 1)
                out["sync_retry_used"] = bool(idx > 0)
                return out
            if idx < (attempts - 1) and self._is_transient_sync_failure(last):
                if delay_sec > 0:
                    time.sleep(delay_sec)
                continue
            break
        final_fail = self.sync_account(
            market=market_scope,
            strict=True,
            notify=bool(notify_on_fail),
            track_streak=bool(track_streak_on_fail),
        )
        final_fail["sync_retry_attempts"] = int(attempts)
        final_fail["sync_retry_used"] = bool(retries > 0)
        final_fail["sync_retry_failed"] = True
        return final_fail

    @staticmethod
    def _has_rate_limit_error(errors: List[Any]) -> bool:
        text = " ".join(str(x) for x in (errors or []))
        norm = text.lower()
        return (
            ("egw00201" in norm)
            or ("초당 거래건수를 초과" in text)
            or ("too many requests" in norm)
            or ("rate limit" in norm)
        )

    def _reconcile_positions(
        self,
        *,
        server_positions: List[Dict[str, Any]],
        local_positions: List[Dict[str, Any]],
        epsilon: float = 1e-6,
    ) -> Tuple[bool, Dict[str, Any]]:
        smap = self._position_map(server_positions)
        lmap = self._position_map(local_positions)
        keys = sorted(set(smap.keys()).union(set(lmap.keys())))
        mismatches: List[Dict[str, Any]] = []
        for key in keys:
            sq = float(smap.get(key, 0.0))
            lq = float(lmap.get(key, 0.0))
            if abs(sq - lq) <= epsilon:
                continue
            mismatches.append(
                {
                    "market": key[0],
                    "symbol": key[1],
                    "server_qty": sq,
                    "local_qty": lq,
                    "delta_qty": sq - lq,
                }
            )
        return len(mismatches) == 0, {"mismatches": mismatches, "mismatch_count": len(mismatches)}

    def sync_account(
        self,
        market: str = "ALL",
        strict: bool = False,
        *,
        notify: bool = True,
        track_streak: bool = True,
    ) -> Dict[str, Any]:
        market_scope = str(market or "ALL").strip().upper()
        strict_flag = bool(strict)
        if not strict and _truthy(os.getenv("AGENT_LAB_SYNC_STRICT", "1")):
            strict_flag = True
        alert_threshold = self._sync_alert_consecutive_threshold()
        max_staleness_sec = self._sync_max_staleness_sec()
        self._ensure_unified_executor_agent()
        epoch_id = self._epoch_id()
        fetched = self.paper_broker.fetch_account_snapshot(market_scope)
        payload: Dict[str, Any] = {
            "market_scope": market_scope,
            "strict": strict_flag,
            "epoch_id": epoch_id,
            "fetched_at": fetched.get("fetched_at", now_iso()),
            "source": fetched.get("source", ""),
            "max_staleness_sec": max_staleness_sec,
            "ok": bool(fetched.get("ok", False)),
            "matched": False,
            "blocked": False,
            "errors": list(fetched.get("errors", []) or []),
        }
        fetched_age = self._age_seconds_from_iso(str(payload.get("fetched_at", "")))
        if fetched_age is not None:
            payload["fetched_age_sec"] = round(float(fetched_age), 3)
        if not payload["ok"]:
            payload["reason"] = "broker_rate_limited" if self._has_rate_limit_error(payload["errors"]) else "broker_fetch_failed"
            payload["blocked"] = strict_flag
            if bool(payload["blocked"]) and bool(track_streak):
                payload["sync_mismatch_streak"] = int(self._inc_sync_mismatch_streak(market_scope))
            elif bool(track_streak):
                payload["sync_mismatch_streak"] = int(self._get_sync_mismatch_streak(market_scope))
            self.storage.log_event("sync_mismatch", payload, now_iso())
            should_notify = bool(
                payload["blocked"]
                and notify
                and (
                    (not track_streak)
                    or int(payload.get("sync_mismatch_streak", 0) or 0) >= alert_threshold
                )
            )
            if should_notify:
                if self._allow_sync_alert_notification(market_scope, payload):
                    self._notify(
                        "[AgentLab] 계좌 동기화 실패\n"
                        f"시장범위={market_scope}\n"
                        f"사유={payload['reason']}\n"
                        f"오류={payload['errors'][:2]}",
                        event="sync_mismatch",
                    )
            return payload

        snapshot_id = self.storage.insert_account_snapshot(
            epoch_id=epoch_id,
            market_scope=str(fetched.get("market_scope", market_scope)).upper(),
            source=str(fetched.get("source", "")),
            cash_krw=float(fetched.get("cash_krw", 0.0) or 0.0),
            equity_krw=float(fetched.get("equity_krw", 0.0) or 0.0),
            payload=fetched if isinstance(fetched, dict) else {},
            created_at=now_iso(),
        )
        server_positions = list(fetched.get("positions", []) or [])
        self.storage.replace_account_positions(snapshot_id, server_positions)
        settlement = self._settle_submitted_orders_from_server(
            market_scope=market_scope,
            server_positions=server_positions,
        )
        payload["settlement"] = settlement
        if notify:
            self._maybe_alert_stale_submitted_orders(market_scope=market_scope, settlement=settlement)
        local_positions = self._local_positions_from_fills(market_scope)
        matched, detail = self._reconcile_positions(server_positions=server_positions, local_positions=local_positions)
        payload["matched"] = bool(matched)
        payload["snapshot_id"] = int(snapshot_id)
        payload["cash_krw"] = float(fetched.get("cash_krw", 0.0) or 0.0)
        payload["equity_krw"] = float(fetched.get("equity_krw", 0.0) or 0.0)
        payload["mismatch_count"] = int(detail.get("mismatch_count", 0) or 0)
        payload["detail"] = detail
        if (not matched) and self._is_pending_settlement_mismatch(detail=detail, market_scope=market_scope):
            payload["reason"] = "pending_settlement"
        payload["blocked"] = bool(strict_flag and not matched and _truthy(os.getenv("AGENT_LAB_SYNC_MISMATCH_BLOCK", "1")))
        if bool(track_streak):
            if matched:
                self._set_sync_mismatch_streak(market_scope, 0)
                payload["sync_mismatch_streak"] = 0
            elif bool(payload["blocked"]):
                payload["sync_mismatch_streak"] = int(self._inc_sync_mismatch_streak(market_scope))
            else:
                payload["sync_mismatch_streak"] = int(self._get_sync_mismatch_streak(market_scope))

        self.storage.insert_reconcile_event(
            check_type="portfolio_positions",
            market_scope=market_scope,
            matched=matched,
            detail=f"strict={strict_flag}",
            expected={"server_positions": server_positions},
            actual={"local_positions": local_positions},
            created_at=now_iso(),
        )
        event_type = "account_synced" if matched else "sync_mismatch"
        self.storage.log_event(event_type, payload, now_iso())
        should_notify = bool(
            payload["blocked"]
            and notify
            and (
                (not track_streak)
                or int(payload.get("sync_mismatch_streak", 0) or 0) >= alert_threshold
            )
        )
        if should_notify:
            if self._allow_sync_alert_notification(market_scope, payload):
                if str(payload.get("reason", "")).strip().lower() == "pending_settlement":
                    self._notify(
                        "[AgentLab] 계좌 동기화 대기(체결 반영 중)\n"
                        f"시장범위={market_scope}\n"
                        f"불일치_건수={payload['mismatch_count']}\n"
                        "조치=반영 확인 전 주문 차단",
                        event="sync_mismatch",
                    )
                else:
                    self._notify(
                        "[AgentLab] 계좌 동기화 불일치\n"
                        f"시장범위={market_scope}\n"
                        f"불일치_건수={payload['mismatch_count']}\n"
                        "조치=불일치 해소 전 주문 차단",
                        event="sync_mismatch",
                    )
        return payload

    def reconcile_submitted_orders(
        self,
        *,
        market: str = "ALL",
        max_age_sec: int = 1800,
        apply: bool = False,
        close_status: str = "REJECTED",
        reason: str = "manual_reconcile_submitted",
    ) -> Dict[str, Any]:
        scope = str(market or "ALL").strip().upper()
        if scope not in {"KR", "US", "ALL"}:
            scope = "ALL"
        max_age = max(30, int(max_age_sec or 1800))
        status_to = str(close_status or "REJECTED").strip().upper() or "REJECTED"
        reason_txt = str(reason or "manual_reconcile_submitted").strip() or "manual_reconcile_submitted"
        requested_apply = bool(apply)

        sync_before = self.sync_account(
            market=scope,
            strict=False,
            notify=False,
            track_streak=False,
        )

        rows = self._list_submitted_paper_orders(scope)
        row_by_id: Dict[int, Dict[str, Any]] = {}
        candidates: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []
        unknown_age: List[Dict[str, Any]] = []

        for row in rows:
            paper_order_id = int(row.get("paper_order_id"))
            row_by_id[paper_order_id] = row
            age_sec = self._age_seconds_from_iso(str(row.get("submitted_at", "")))
            item = {
                "paper_order_id": paper_order_id,
                "broker_order_id": str(row.get("broker_order_id", "")).strip(),
                "market": str(row.get("market", "")).strip().upper(),
                "symbol": str(row.get("symbol", "")).strip().upper(),
                "side": str(row.get("side", "")).strip().upper(),
                "quantity": float(row.get("quantity", 0.0) or 0.0),
                "submitted_at": str(row.get("submitted_at", "")),
                "age_sec": None if age_sec is None else float(age_sec),
            }
            if age_sec is None:
                unknown_age.append(item)
                continue
            if float(age_sec) >= float(max_age):
                candidates.append(item)
            else:
                recent.append(item)

        blocked_apply = bool(requested_apply and not bool(sync_before.get("ok", False)))
        apply_reason = "sync_before_failed" if blocked_apply else ""
        updated: List[Dict[str, Any]] = []
        applied = False
        if requested_apply and not blocked_apply:
            for item in candidates:
                paper_order_id = int(item["paper_order_id"])
                raw_row = row_by_id.get(paper_order_id, {})
                broker_payload = self._decode_json_obj(raw_row.get("broker_response_json"))
                broker_payload["reconciled_cleanup"] = {
                    "applied_at": now_iso(),
                    "from_status": "SUBMITTED",
                    "to_status": status_to,
                    "reason": reason_txt,
                    "age_sec": item.get("age_sec"),
                }
                self.storage.update_paper_order_status_with_response(
                    paper_order_id=paper_order_id,
                    status=status_to,
                    broker_response_json=broker_payload,
                )
                updated.append(
                    {
                        "paper_order_id": paper_order_id,
                        "broker_order_id": item.get("broker_order_id", ""),
                        "market": item.get("market", ""),
                        "symbol": item.get("symbol", ""),
                        "side": item.get("side", ""),
                        "quantity": item.get("quantity", 0.0),
                        "age_sec": item.get("age_sec"),
                        "status_to": status_to,
                    }
                )
            applied = True

        payload: Dict[str, Any] = {
            "market_scope": scope,
            "max_age_sec": int(max_age),
            "requested_apply": requested_apply,
            "applied": bool(applied),
            "blocked_apply": bool(blocked_apply),
            "blocked_reason": apply_reason,
            "status_to": status_to,
            "reason": reason_txt,
            "sync_before": sync_before,
            "submitted_total": int(len(rows)),
            "eligible_count": int(len(candidates)),
            "recent_count": int(len(recent)),
            "unknown_age_count": int(len(unknown_age)),
            "eligible_orders": candidates[:200],
            "updated_orders": updated[:200],
            "generated_at": now_iso(),
        }
        event_type = "submitted_orders_reconciled" if applied else "submitted_orders_reconcile_dryrun"
        self.storage.log_event(event_type, payload, now_iso())
        if applied and updated:
            lines = [
                "[AgentLab] 미해소 주문 정리 완료",
                f"시장범위={scope}",
                f"정리건수={len(updated)}",
                f"상태변경=SUBMITTED->{status_to}",
                f"기준_초={max_age}",
                f"사유={reason_txt}",
            ]
            for row in updated[:5]:
                lines.append(
                    "주문="
                    f"paper_order_id={int(row.get('paper_order_id', 0))}, "
                    f"broker_order_id={str(row.get('broker_order_id', '') or '-').strip() or '-'}, "
                    f"{str(row.get('side', '')).strip().upper()} {str(row.get('symbol', '')).strip().upper()}"
                )
            self._notify("\n".join(lines), event="broker_api_error")
        return payload

    def _latest_account_context(self, market_scope: str = "ALL") -> Dict[str, Any]:
        latest = self.storage.get_latest_account_snapshot(str(market_scope).upper())
        if latest is None:
            synced = self.sync_account(market=str(market_scope).upper(), strict=False)
            if not synced.get("ok", False):
                return {"cash_krw": 0.0, "equity_krw": 0.0, "positions": [], "snapshot_id": None}
            latest = self.storage.get_latest_account_snapshot(str(market_scope).upper())
        max_staleness_sec = self._sync_max_staleness_sec()
        latest_age = self._age_seconds_from_iso(str((latest or {}).get("created_at", ""))) if latest else None
        if latest is not None and latest_age is not None and latest_age > float(max_staleness_sec):
            synced = self.sync_account(
                market=str(market_scope).upper(),
                strict=_truthy(os.getenv("AGENT_LAB_SYNC_STRICT", "1")),
            )
            if synced.get("ok", False):
                latest = self.storage.get_latest_account_snapshot(str(market_scope).upper())
        if latest is None:
            return {"cash_krw": 0.0, "equity_krw": 0.0, "positions": [], "snapshot_id": None}
        snapshot_id = int(latest.get("snapshot_id"))
        positions = self.storage.list_account_positions_by_snapshot(snapshot_id)
        out_positions: List[Dict[str, Any]] = []
        for row in positions:
            out_positions.append(
                {
                    "market": str(row.get("market", "")),
                    "symbol": str(row.get("symbol", "")),
                    "quantity": float(row.get("quantity", 0.0) or 0.0),
                    "avg_price": float(row.get("avg_price", 0.0) or 0.0),
                    "market_value_krw": float(row.get("market_value_krw", 0.0) or 0.0),
                    "unrealized_pnl_krw": 0.0,
                }
            )
        return {
            "cash_krw": float(latest.get("cash_krw", 0.0) or 0.0),
            "equity_krw": float(latest.get("equity_krw", 0.0) or 0.0),
            "positions": out_positions,
            "snapshot_id": snapshot_id,
        }

    def _load_shadow_weights(self, agent_ids: List[str]) -> Dict[str, float]:
        if not agent_ids:
            return {}
        latest = self.storage.get_latest_shadow_scores()
        raw: Dict[str, float] = {}
        for aid in agent_ids:
            row = latest.get(aid, {})
            if row:
                raw[aid] = max(0.0, float(row.get("score", 0.0) or 0.0))
            else:
                raw[aid] = 0.0
        total = sum(raw.values())
        if total <= 1e-12:
            w = 1.0 / float(len(agent_ids))
            return {aid: w for aid in agent_ids}
        return {aid: (raw[aid] / total) for aid in agent_ids}

    def _aggregate_shadow_orders(
        self,
        *,
        market: str,
        intents: List[Dict[str, Any]],
        shared_cash_krw: float,
        shared_positions: List[Dict[str, Any]],
        usdkrw: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        by_symbol_side: Dict[Tuple[str, str], Dict[str, Any]] = {}
        total_weight = sum(float(x.get("weight", 0.0) or 0.0) for x in intents)
        if total_weight <= 1e-12:
            total_weight = float(len(intents) or 1)
        for intent in intents:
            aid = str(intent.get("agent_id", ""))
            weight = float(intent.get("weight", 0.0) or 0.0)
            orders = list(intent.get("orders", []) or [])
            for row in orders:
                symbol = str(row.get("symbol", "")).strip().upper()
                side = str(row.get("side", "")).strip().upper()
                qty = float(row.get("quantity", 0.0) or 0.0)
                px = float(row.get("reference_price", 0.0) or 0.0)
                if not symbol or side not in {"BUY", "SELL"} or qty <= 0 or px <= 0:
                    continue
                score = float(row.get("recommendation_score", 50.0) or 50.0)
                vote = max(0.0, weight * max(0.05, score / 100.0))
                key = (symbol, side)
                agg = by_symbol_side.setdefault(
                    key,
                    {
                        "symbol": symbol,
                        "side": side,
                        "order_type": "MARKET",
                        "vote_weight": 0.0,
                        "weighted_qty": 0.0,
                        "weighted_price": 0.0,
                        "contributors": [],
                    },
                )
                agg["vote_weight"] += vote
                agg["weighted_qty"] += (weight * qty)
                agg["weighted_price"] += (weight * qty * px)
                agg["contributors"].append({"agent_id": aid, "weight": weight, "qty": qty, "score": score})

        # Resolve buy/sell conflict on same symbol: keep the higher vote side.
        chosen: Dict[str, Dict[str, Any]] = {}
        for (_symbol, _side), agg in by_symbol_side.items():
            prev = chosen.get(agg["symbol"])
            if prev is None or float(agg["vote_weight"]) > float(prev.get("vote_weight", 0.0)):
                chosen[agg["symbol"]] = agg

        min_ratio_buy = float(os.getenv("AGENT_LAB_AGG_MIN_VOTE_RATIO", "0.34") or 0.34)
        min_ratio_buy = max(0.05, min(0.95, min_ratio_buy))
        min_ratio_sell = float(os.getenv("AGENT_LAB_AGG_MIN_VOTE_RATIO_SELL", "0.33") or 0.33)
        min_ratio_sell = max(0.05, min(0.95, min_ratio_sell))
        held_map = self._position_map(shared_positions)
        out_orders: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        est_sell_proceeds = 0.0
        for agg in chosen.values():
            vote_ratio = float(agg.get("vote_weight", 0.0) or 0.0) / total_weight
            side_upper = str(agg.get("side", "")).strip().upper()
            min_ratio = min_ratio_sell if side_upper == "SELL" else min_ratio_buy
            if vote_ratio < min_ratio:
                dropped.append(
                    {
                        "symbol": agg["symbol"],
                        "side": agg["side"],
                        "reason": "vote_ratio_low",
                        "vote_ratio": vote_ratio,
                        "min_vote_ratio": min_ratio,
                    }
                )
                continue
            qty = int(max(1, round(float(agg.get("weighted_qty", 0.0) or 0.0))))
            price = 0.0
            denom = float(agg.get("weighted_qty", 0.0) or 0.0)
            if denom > 0:
                price = float(agg.get("weighted_price", 0.0) or 0.0) / denom
            if price <= 0:
                dropped.append({"symbol": agg["symbol"], "side": agg["side"], "reason": "invalid_price"})
                continue
            if agg["side"] == "SELL":
                hold_qty = float(held_map.get((market, agg["symbol"]), 0.0) or 0.0)
                if hold_qty <= 0:
                    dropped.append({"symbol": agg["symbol"], "side": "SELL", "reason": "no_position"})
                    continue
                qty = int(max(0, min(float(qty), hold_qty)))
                if qty <= 0:
                    dropped.append({"symbol": agg["symbol"], "side": "SELL", "reason": "sell_qty_zero"})
                    continue
                est_sell_proceeds += float(qty) * float(price) * (float(usdkrw) if market == "US" else 1.0)
            out_orders.append(
                {
                    "market": market,
                    "symbol": agg["symbol"],
                    "side": agg["side"],
                    "order_type": "MARKET",
                    "quantity": float(qty),
                    "limit_price": None,
                    "reference_price": float(price),
                    "signal_rank": 0,
                    "recommendation_score": float(min(100.0, max(0.0, vote_ratio * 100.0))),
                    "rationale": f"unified_consensus vote_ratio={vote_ratio:.3f}",
                }
            )

        # Cash cap for BUY legs
        budget_krw = max(0.0, float(shared_cash_krw) + est_sell_proceeds)
        buys = [row for row in out_orders if str(row.get("side", "")).upper() == "BUY"]
        sells = [row for row in out_orders if str(row.get("side", "")).upper() == "SELL"]
        buys.sort(key=lambda x: float(x.get("recommendation_score", 0.0) or 0.0), reverse=True)
        buy_final: List[Dict[str, Any]] = []
        for row in buys:
            fx = float(usdkrw) if market == "US" else 1.0
            px_krw = float(row.get("reference_price", 0.0) or 0.0) * fx
            if px_krw <= 0:
                continue
            qty = int(float(row.get("quantity", 0.0) or 0.0))
            max_qty = int(math.floor(budget_krw / px_krw))
            qty = min(qty, max_qty)
            if qty <= 0:
                continue
            row = dict(row)
            row["quantity"] = float(qty)
            buy_final.append(row)
            budget_krw -= float(qty) * px_krw
            if budget_krw <= 0:
                break

        merged = sells + buy_final
        detail = {
            "intent_count": len(intents),
            "total_weight": total_weight,
            "min_vote_ratio_buy": min_ratio_buy,
            "min_vote_ratio_sell": min_ratio_sell,
            "kept_orders": len(merged),
            "dropped": dropped,
            "est_sell_proceeds_krw": est_sell_proceeds,
            "remaining_buy_budget_krw": budget_krw,
        }
        return merged, detail

    def init_lab(self, capital_krw: float, agents: int) -> Dict[str, Any]:
        profiles = build_default_agent_profiles(total_capital_krw=capital_krw, count=agents)
        for p in profiles:
            self.storage.upsert_agent(
                agent_id=p.agent_id,
                name=p.name,
                role=p.role,
                philosophy=p.philosophy,
                allocated_capital_krw=p.allocated_capital_krw,
                risk_style=p.risk_style,
                constraints=p.constraints,
                created_at=now_iso(),
            )
            self.identity.ensure_identity(p)
            self.identity.append_memory(
                p.agent_id,
                memory_type="init",
                content={"message": "Agent initialized.", "capital": p.allocated_capital_krw},
                ts=now_iso(),
            )

        self.registry.initialize_default_versions([p.agent_id for p in profiles])
        self._ensure_unified_executor_agent()
        self.storage.upsert_system_meta("execution_model", self._execution_model(), now_iso())
        self.storage.log_event(
            event_type="lab_initialized",
            payload={"capital_krw": capital_krw, "agents": len(profiles), "date": self._today_str()},
            created_at=now_iso(),
        )
        payload = {"agents": [p.agent_id for p in profiles], "capital_krw": capital_krw}
        self._append_activity_artifact(self._today_str(), "lab_initialized", payload)
        self._notify(
            "[AgentLab] 초기화\n"
            f"초기자본(원화)={capital_krw:.0f}\n"
            f"에이전트={', '.join(payload['agents'])}"
            ,
            event="init",
        )
        return payload

    def sanitize_legacy_constraints(
        self,
        *,
        clean_pending_proposals: bool = True,
        cleanup_runtime_state: bool = True,
        retain_days: int = 30,
        keep_agent_memories: int = 300,
    ) -> Dict[str, Any]:
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")

        per_agent: List[Dict[str, Any]] = []
        for row in agents:
            aid = str(row.get("agent_id", ""))
            old_profile = profile_from_agent_row(row)
            refreshed_profile = type(old_profile)(
                agent_id=old_profile.agent_id,
                name=old_profile.name,
                role=old_profile.role,
                philosophy=old_profile.philosophy,
                allocated_capital_krw=old_profile.allocated_capital_krw,
                risk_style=old_profile.risk_style,
                constraints=self._sanitized_agent_constraints(old_profile.risk_style),
            )
            self.storage.upsert_agent(
                agent_id=refreshed_profile.agent_id,
                name=refreshed_profile.name,
                role=refreshed_profile.role,
                philosophy=refreshed_profile.philosophy,
                allocated_capital_krw=refreshed_profile.allocated_capital_krw,
                risk_style=refreshed_profile.risk_style,
                constraints=refreshed_profile.constraints,
                created_at=now_iso(),
            )
            self.identity.ensure_identity(refreshed_profile, force_refresh=True)
            mem_stat = self.identity.sanitize_memory_file(aid)
            db_removed = self.storage.delete_legacy_agent_memories(aid)

            try:
                active = self.registry.get_active_strategy(aid)
                active_params = dict(active.get("params", {}) or {})
            except Exception:
                active_params = {}
            merged = dict(DEFAULT_STRATEGY_PARAMS)
            merged.update(active_params)
            merged.update(
                {
                    "intraday_monitor_enabled": 1,
                    "intraday_monitor_interval_sec": int(
                        float(os.getenv("AGENT_LAB_MAX_FREEDOM_INTERVAL_SEC", "30") or 30)
                    ),
                    "max_daily_picks": int(float(os.getenv("AGENT_LAB_MAX_FREEDOM_PICKS", "20") or 20)),
                    "risk_budget_ratio": 1.0,
                    "day_loss_limit": -1.0,
                    "week_loss_limit": -1.0,
                }
            )
            reg = self.registry.register_strategy_version(
                agent_id=aid,
                params=merged,
                notes="sanitize_legacy_constraints: refresh identity/memory and max-freedom params",
                promote=True,
            )
            per_agent.append(
                {
                    "agent_id": aid,
                    "memory_removed": int(mem_stat.get("removed", 0)),
                    "memory_removed_db": int(db_removed),
                    "strategy_version": str(reg.get("version_tag", "")),
                }
            )

        pending_cleaned = 0
        if clean_pending_proposals:
            pending_cleaned = self.storage.bulk_update_pending_proposals(
                new_status=PROPOSAL_STATUS_BLOCKED,
                blocked_reason="legacy_cleanup_no_manual_approval",
                updated_at=now_iso(),
            )

        runtime_cleanup = {
            "order_approvals_removed": 0,
            "order_proposals_removed": 0,
            "session_signals_removed": 0,
            "state_events_removed": 0,
            "daily_reviews_removed": 0,
            "agent_memories_trimmed": 0,
        }
        if cleanup_runtime_state:
            runtime_cleanup = self.storage.cleanup_legacy_runtime_state(
                retain_days=int(retain_days),
                keep_agent_memories=int(keep_agent_memories),
            )

        payload = {
            "agents": per_agent,
            "pending_proposals_cleaned": pending_cleaned,
            "clean_pending_proposals": bool(clean_pending_proposals),
            "cleanup_runtime_state": bool(cleanup_runtime_state),
            "runtime_cleanup": runtime_cleanup,
            "retain_days": int(retain_days),
            "keep_agent_memories": int(keep_agent_memories),
            "updated_at": now_iso(),
        }
        self.storage.log_event("sanitize_legacy_constraints", payload, now_iso())
        self._append_activity_artifact(self._today_str(), "sanitize_legacy_constraints", payload)
        self._notify(
            "[AgentLab] 레거시 제약 정리\n"
            f"대상_에이전트={len(per_agent)}\n"
            f"정리된_대기건수={pending_cleaned}"
            ,
            event="sanitize",
        )
        return payload

    def ingest_session(self, market: str, yyyymmdd: str) -> Dict[str, Any]:
        market = market.strip().upper()
        path = self._find_session_json(market, yyyymmdd)
        if path is None:
            raise FileNotFoundError(f"session result json not found for market={market}, date={yyyymmdd}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        status_code, invalid_reason = self._classify_signal(payload)
        generated_at = str(payload.get("generated_at") or now_iso())
        signal_valid = bool(payload.get("signal_valid", False))
        session_signal_id = self.storage.insert_session_signal(
            market=market,
            session_date=yyyymmdd,
            generated_at=generated_at,
            signal_valid=signal_valid,
            status_code=status_code,
            invalid_reason=invalid_reason,
            source_json_path=str(path),
            payload=payload,
            created_at=now_iso(),
        )
        out = {
            "session_signal_id": session_signal_id,
            "market": market,
            "session_date": yyyymmdd,
            "status_code": status_code,
            "invalid_reason": invalid_reason,
            "source_json_path": str(path),
        }
        self._write_json_artifact(yyyymmdd, f"ingest_{market.lower()}_{yyyymmdd}", out)
        self.storage.log_event("session_ingested", out, now_iso())
        self._append_activity_artifact(yyyymmdd, "session_ingested", out)
        self._notify(
            "[AgentLab] 세션 인제스트\n"
            f"시장={market}\n"
            f"일자={yyyymmdd}\n"
            f"상태={status_code}\n"
            f"무효_사유={invalid_reason or '-'}"
            ,
            event="ingest_session",
        )
        return out

    def _propose_orders_unified(
        self,
        *,
        market: str,
        effective_session_date: str,
        signal: Dict[str, Any],
        auto_execute: bool,
    ) -> Dict[str, Any]:
        self._ensure_unified_executor_agent()
        sync_scope = self._execution_sync_scope()
        sync_pre = self._sync_account_strict_with_retry(
            market_scope=sync_scope,
            notify_on_fail=True,
            track_streak_on_fail=True,
        )
        if bool(sync_pre.get("blocked", False)):
            payload = {
                "market": market,
                "date": effective_session_date,
                "auto_execute": bool(auto_execute),
                "execution_model": "unified_shadow",
                "skipped": True,
                "reason": "sync_mismatch",
                "sync": sync_pre,
                "proposals": [],
                "execution_results": [],
            }
            self.storage.log_event("orders_proposed_skipped", payload, now_iso())
            self._append_activity_artifact(effective_session_date, "orders_proposed_skipped", payload)
            return payload

        account_ctx = self._latest_account_context(sync_scope)
        shared_cash = float(account_ctx.get("cash_krw", 0.0) or 0.0)
        shared_equity = float(account_ctx.get("equity_krw", 0.0) or 0.0)
        shared_positions = list(account_ctx.get("positions", []) or [])
        shared_ledger = {
            "cash_krw": shared_cash,
            "equity_krw": shared_equity,
            "positions": shared_positions,
        }
        market_exposure = 0.0
        pos_qty_by_symbol: Dict[str, float] = {}
        for pos in shared_positions:
            if str(pos.get("market", "")).strip().upper() != market:
                continue
            market_exposure += float(pos.get("market_value_krw", 0.0) or 0.0)
            symbol = str(pos.get("symbol", "")).strip().upper()
            qty = float(pos.get("quantity", 0.0) or 0.0)
            if symbol and qty > 0:
                pos_qty_by_symbol[symbol] = qty

        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        agent_ids = [str(x.get("agent_id", "")) for x in agents]
        weights = self._load_shadow_weights(agent_ids)
        usdkrw = self._usdkrw_rate(effective_session_date) if market == "US" else 1.0
        base_payload = signal.get("payload", {})
        if not isinstance(base_payload, dict):
            base_payload = {}
        symbol_name_map = self._build_symbol_name_map_from_payload(base_payload)

        shadow_rows: List[Dict[str, Any]] = []
        intents_for_agg: List[Dict[str, Any]] = []
        for agent in agents:
            agent_id = str(agent["agent_id"])
            profile = profile_from_agent_row(agent)
            active = self.registry.get_active_strategy(agent_id)
            active_params = dict(active.get("params", {}) or {})
            cross_market_plan = self._build_cross_market_plan(
                profile=profile,
                active_params=active_params,
                market=market,
                yyyymmdd=effective_session_date,
                ledger=shared_ledger,
            )
            agent_signal_payload = self._build_agent_specific_payload(
                base_payload=base_payload,
                agent_id=agent_id,
                market=market,
                session_date=effective_session_date,
            )
            proposed_orders, rationale = self.agent_engine.propose_orders(
                agent=profile,
                market=market,
                session_payload=agent_signal_payload,
                params=active_params,
                available_cash_krw=shared_cash,
                current_positions=shared_positions,
                usdkrw_rate=usdkrw,
                market_budget_cap_krw=float(cross_market_plan.get("buy_budget_cap_krw", 0.0) or 0.0),
                max_picks_override=int(cross_market_plan.get("max_picks_override", 1) or 1),
                min_recommendation_score=(
                    None
                    if float(cross_market_plan.get("min_recommendation_score", 0.0) or 0.0) <= 0
                    else float(cross_market_plan.get("min_recommendation_score", 0.0) or 0.0)
                ),
            )
            decision = self.risk.evaluate(
                status_code=str(signal["status_code"]),
                allocated_capital_krw=max(1.0, shared_equity),
                available_cash_krw=shared_cash,
                day_return_pct=0.0,
                week_return_pct=0.0,
                current_exposure_krw=float(market_exposure),
                orders=[o.__dict__ for o in proposed_orders],
                usdkrw_rate=usdkrw,
                position_cap_ratio=float(
                    active_params.get("position_cap_ratio", self.risk.position_cap_ratio) or self.risk.position_cap_ratio
                ),
                exposure_cap_ratio=float(
                    active_params.get("exposure_cap_ratio", self.risk.exposure_cap_ratio) or self.risk.exposure_cap_ratio
                ),
                day_loss_limit=float(active_params.get("day_loss_limit", self.risk.day_loss_limit) or self.risk.day_loss_limit),
                week_loss_limit=float(active_params.get("week_loss_limit", self.risk.week_loss_limit) or self.risk.week_loss_limit),
                position_qty_by_symbol=pos_qty_by_symbol,
            )
            status = "SHADOW_READY" if decision.allowed else "SHADOW_BLOCKED"
            proposal_uuid = str(uuid.uuid4())
            proposal_id = self.storage.insert_order_proposal(
                proposal_uuid=proposal_uuid,
                agent_id=agent_id,
                market=market,
                session_date=effective_session_date,
                strategy_version_id=int(active["strategy_version_id"]),
                status=status,
                blocked_reason=decision.blocked_reason,
                orders=decision.accepted_orders,
                rationale=rationale,
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            weight = float(weights.get(agent_id, 0.0) or 0.0)
            intent_score = float(sum(float(x.get("recommendation_score", 0.0) or 0.0) for x in decision.accepted_orders))
            self.storage.insert_shadow_intent(
                proposal_id=proposal_id,
                agent_id=agent_id,
                market=market,
                session_date=effective_session_date,
                weight=weight,
                score=intent_score,
                orders=list(decision.accepted_orders),
                rationale=rationale,
                created_at=now_iso(),
            )
            if decision.allowed and decision.accepted_orders:
                intents_for_agg.append(
                    {
                        "agent_id": agent_id,
                        "weight": weight,
                        "orders": list(decision.accepted_orders),
                    }
                )
            shadow_rows.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_uuid": proposal_uuid,
                    "agent_id": agent_id,
                    "status": status,
                    "blocked_reason": decision.blocked_reason,
                    "orders": list(decision.accepted_orders),
                    "weight": weight,
                    "intent_score": intent_score,
                    "agent_signal_view": agent_signal_payload.get("agent_signal_view", {}),
                    "cross_market_plan": cross_market_plan,
                }
            )

        sync_pre_aggregate = self._sync_account_strict_with_retry(
            market_scope=sync_scope,
            notify_on_fail=True,
            track_streak_on_fail=True,
        )
        fail_open_used = False
        fail_open_reason = ""
        if bool(sync_pre_aggregate.get("blocked", False)) and self._sync_fail_open_on_transient():
            if self._is_transient_sync_failure(sync_pre_aggregate):
                grace_sec = self._sync_fail_open_grace_sec()
                if self._recent_strict_sync_success(sync_scope, max_age_sec=grace_sec):
                    fail_open_used = True
                    fail_open_reason = "transient_sync_failure_with_recent_strict_sync_success"
                    self.storage.log_event(
                        "sync_fail_open_used",
                        {
                            "market": market,
                            "date": effective_session_date,
                            "phase": "pre_aggregate",
                            "grace_sec": int(grace_sec),
                            "reason": fail_open_reason,
                            "sync_pre_aggregate": sync_pre_aggregate,
                        },
                        now_iso(),
                    )
        if bool(sync_pre_aggregate.get("blocked", False)):
            if fail_open_used:
                sync_pre_aggregate = dict(sync_pre_aggregate)
                sync_pre_aggregate["fail_open_used"] = True
                sync_pre_aggregate["fail_open_reason"] = fail_open_reason
            else:
                payload = {
                    "market": market,
                    "date": effective_session_date,
                    "auto_execute": bool(auto_execute),
                    "execution_model": "unified_shadow",
                    "skipped": True,
                    "reason": "sync_mismatch_before_aggregate",
                    "sync_pre": sync_pre,
                    "sync_pre_aggregate": sync_pre_aggregate,
                    "shadow_proposals": shadow_rows,
                    "proposals": shadow_rows,
                    "execution_results": [],
                }
                self.storage.log_event("orders_proposed_skipped", payload, now_iso())
                self._append_activity_artifact(effective_session_date, "orders_proposed_skipped", payload)
                return payload
        if bool(sync_pre_aggregate.get("blocked", False)) and fail_open_used:
            self.storage.log_event(
                "sync_pre_aggregate_soft_bypass",
                {
                    "market": market,
                    "date": effective_session_date,
                    "reason": fail_open_reason,
                    "sync_pre_aggregate": sync_pre_aggregate,
                },
                now_iso(),
            )
        merged_orders, agg_detail = self._aggregate_shadow_orders(
            market=market,
            intents=intents_for_agg,
            shared_cash_krw=shared_cash,
            shared_positions=shared_positions,
            usdkrw=usdkrw,
        )
        unified_active = self.registry.get_active_strategy(self.UNIFIED_EXECUTOR_AGENT_ID)
        unified_status = PROPOSAL_STATUS_APPROVED if merged_orders else PROPOSAL_STATUS_BLOCKED
        unified_block = "" if merged_orders else "no_consensus_order"
        unified_proposal_uuid = str(uuid.uuid4())
        unified_proposal_id = self.storage.insert_order_proposal(
            proposal_uuid=unified_proposal_uuid,
            agent_id=self.UNIFIED_EXECUTOR_AGENT_ID,
            market=market,
            session_date=effective_session_date,
            strategy_version_id=int(unified_active["strategy_version_id"]),
            status=unified_status,
            blocked_reason=unified_block,
            orders=merged_orders,
            rationale=f"unified_shadow_consensus detail={json.dumps(agg_detail, ensure_ascii=False)}",
            created_at=now_iso(),
            updated_at=now_iso(),
        )

        execution_results: List[Dict[str, Any]] = []
        unified_row = {
            "proposal_id": unified_proposal_id,
            "proposal_uuid": unified_proposal_uuid,
            "agent_id": self.UNIFIED_EXECUTOR_AGENT_ID,
            "status": unified_status,
            "blocked_reason": unified_block,
            "orders": merged_orders,
            "aggregation_detail": agg_detail,
            "execution_model": "unified_shadow",
        }
        if bool(auto_execute) and unified_status == PROPOSAL_STATUS_APPROVED:
            try:
                executed = self._execute_proposal(
                    proposal_identifier=str(unified_proposal_id),
                    approved_by="unified_auto_executor",
                    note=f"unified auto-approved market={market} date={effective_session_date}",
                    allowed_statuses=[PROPOSAL_STATUS_APPROVED],
                )
                unified_row["status"] = str(executed.get("status", unified_status))
                execution_results.append(
                    {
                        "proposal_id": unified_proposal_id,
                        "agent_id": self.UNIFIED_EXECUTOR_AGENT_ID,
                        "ok": True,
                        "status": unified_row["status"],
                        "fills": executed.get("fills", []),
                    }
                )
            except Exception as exc:
                self.storage.update_order_proposal_status(
                    proposal_id=unified_proposal_id,
                    status=PROPOSAL_STATUS_BLOCKED,
                    blocked_reason=f"auto_execute_error:{repr(exc)}",
                    updated_at=now_iso(),
                )
                unified_row["status"] = PROPOSAL_STATUS_BLOCKED
                unified_row["blocked_reason"] = f"auto_execute_error:{repr(exc)}"
                execution_results.append(
                    {
                        "proposal_id": unified_proposal_id,
                        "agent_id": self.UNIFIED_EXECUTOR_AGENT_ID,
                        "ok": False,
                        "error": repr(exc),
                    }
                )

        sync_post = self.sync_account(market=sync_scope, strict=False)
        payload = {
            "market": market,
            "date": effective_session_date,
            "auto_execute": bool(auto_execute),
            "execution_model": "unified_shadow",
            "sync_pre": sync_pre,
            "sync_pre_aggregate": sync_pre_aggregate,
            "sync_post": sync_post,
            "shadow_proposals": shadow_rows,
            "unified_execution": unified_row,
            "proposals": shadow_rows + [unified_row],
            "execution_results": execution_results,
        }
        self._write_json_artifact(
            effective_session_date,
            f"proposals_{market.lower()}_{effective_session_date}",
            payload,
        )
        self.storage.log_event("orders_proposed", payload, now_iso())
        self._append_activity_artifact(effective_session_date, "orders_proposed", payload)
        lines = [
            "[AgentLab] 주문 제안",
            f"시장={market}",
            f"일자={effective_session_date}",
            "실행모델=통합포트폴리오+섀도우",
            f"자동실행={bool(auto_execute)}",
            f"통합집행={self._orders_to_text(list(unified_row.get('orders') or []), limit=5, symbol_name_map=symbol_name_map)}",
        ]
        for row in shadow_rows:
            lines.append(
                f"{row.get('agent_id')}: {row.get('status')} | "
                f"{self._orders_to_text(list(row.get('orders') or []), limit=2, symbol_name_map=symbol_name_map)}"
            )
        self._notify("\n".join(lines), event="propose")
        return payload

    def propose_orders(
        self,
        market: str,
        yyyymmdd: str,
        *,
        auto_execute: Optional[bool] = None,
    ) -> Dict[str, Any]:
        market = market.strip().upper()
        requested_session_date = str(yyyymmdd).strip()
        if not self._is_order_market_enabled(market):
            enabled_markets = self._order_enabled_markets()
            payload = {
                "market": market,
                "date": requested_session_date,
                "auto_execute": False,
                "proposals": [],
                "execution_results": [],
                "skipped": True,
                "reason": "market_disabled_by_env",
                "enabled_order_markets": enabled_markets,
            }
            self.storage.log_event("orders_proposed_skipped", payload, now_iso())
            self._append_activity_artifact(requested_session_date, "orders_proposed_skipped", payload)
            return payload
        market_open, market_window_detail = self._is_market_open_now(market)
        if self._enforce_market_hours() and not market_open:
            payload = {
                "market": market,
                "date": requested_session_date,
                "auto_execute": False,
                "proposals": [],
                "execution_results": [],
                "skipped": True,
                "reason": "outside_market_hours",
                "detail": market_window_detail,
            }
            self.storage.log_event("orders_proposed_skipped", payload, now_iso())
            self._append_activity_artifact(requested_session_date, "orders_proposed_skipped", payload)
            return payload

        effective_session_date = requested_session_date
        if market_open:
            live_session_date = self._session_date_now(market)
            if live_session_date and live_session_date != requested_session_date:
                self.storage.log_event(
                    "proposal_session_date_corrected",
                    {
                        "market": market,
                        "requested_session_date": requested_session_date,
                        "effective_session_date": live_session_date,
                        "reason": "market_open_force_current_session_date",
                    },
                    now_iso(),
                )
                effective_session_date = live_session_date

        signal = self.storage.get_latest_session_signal(market, effective_session_date)
        if signal is None:
            self.ingest_session(market, effective_session_date)
            signal = self.storage.get_latest_session_signal(market, effective_session_date)
        if signal is None:
            raise RuntimeError("failed to load session signal")

        # Agent-first repository policy: order proposals are always auto-executed.
        requested_auto_execute = auto_execute
        auto_execute = True
        if requested_auto_execute is False:
            self.storage.log_event(
                "auto_execute_forced",
                {
                    "market": market,
                    "date": effective_session_date,
                    "requested_auto_execute": bool(requested_auto_execute),
                    "forced_auto_execute": True,
                },
                now_iso(),
            )

        if self._unified_shadow_mode():
            return self._propose_orders_unified(
                market=market,
                effective_session_date=effective_session_date,
                signal=signal,
                auto_execute=bool(auto_execute),
            )

        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        output_rows: List[Dict[str, Any]] = []
        usdkrw = self._usdkrw_rate(effective_session_date) if market == "US" else 1.0
        base_payload = signal.get("payload", {})
        if not isinstance(base_payload, dict):
            base_payload = {}
        symbol_name_map = self._build_symbol_name_map_from_payload(base_payload)
        for agent in agents:
            agent_id = str(agent["agent_id"])
            profile = profile_from_agent_row(agent)
            active = self.registry.get_active_strategy(agent_id)
            active_params = dict(active.get("params", {}) or {})
            ledger = self.accounting.rebuild_agent_ledger(agent_id)
            cross_market_plan = self._build_cross_market_plan(
                profile=profile,
                active_params=active_params,
                market=market,
                yyyymmdd=effective_session_date,
                ledger=ledger,
            )
            day_ret, week_ret = self.accounting.daily_and_weekly_return(agent_id, effective_session_date)
            position_qty_by_symbol: Dict[str, float] = {}
            for pos in list(ledger.get("positions") or []):
                pos_market = str(pos.get("market", "")).strip().upper()
                if pos_market != market:
                    continue
                symbol = str(pos.get("symbol", "")).strip().upper()
                qty = float(pos.get("quantity", 0.0) or 0.0)
                if symbol and qty > 0:
                    position_qty_by_symbol[symbol] = qty

            agent_signal_payload = self._build_agent_specific_payload(
                base_payload=base_payload,
                agent_id=agent_id,
                market=market,
                session_date=effective_session_date,
            )
            proposed_orders, rationale = self.agent_engine.propose_orders(
                agent=profile,
                market=market,
                session_payload=agent_signal_payload,
                params=active_params,
                available_cash_krw=float(ledger["cash_krw"]),
                current_positions=list(ledger.get("positions", [])),
                usdkrw_rate=usdkrw,
                market_budget_cap_krw=float(cross_market_plan.get("buy_budget_cap_krw", 0.0) or 0.0),
                max_picks_override=int(cross_market_plan.get("max_picks_override", 1) or 1),
                min_recommendation_score=(
                    None
                    if float(cross_market_plan.get("min_recommendation_score", 0.0) or 0.0) <= 0
                    else float(cross_market_plan.get("min_recommendation_score", 0.0) or 0.0)
                ),
            )
            rationale = (
                f"{rationale} | cross_market: target_weight={float(cross_market_plan.get('target_weights', {}).get(market, 0.0)):.3f}, "
                f"budget_cap_krw={float(cross_market_plan.get('buy_budget_cap_krw', 0.0) or 0.0):.0f}, "
                f"max_picks={int(cross_market_plan.get('max_picks_override', 1) or 1)}"
            )
            raw_orders = [o.__dict__ for o in proposed_orders]
            decision = self.risk.evaluate(
                status_code=str(signal["status_code"]),
                allocated_capital_krw=float(agent["allocated_capital_krw"]),
                available_cash_krw=float(ledger["cash_krw"]),
                day_return_pct=float(day_ret),
                week_return_pct=float(week_ret),
                current_exposure_krw=float(self.accounting.current_exposure_krw(agent_id)),
                orders=raw_orders,
                usdkrw_rate=usdkrw,
                position_cap_ratio=float(
                    active_params.get("position_cap_ratio", self.risk.position_cap_ratio) or self.risk.position_cap_ratio
                ),
                exposure_cap_ratio=float(
                    active_params.get("exposure_cap_ratio", self.risk.exposure_cap_ratio) or self.risk.exposure_cap_ratio
                ),
                day_loss_limit=float(
                    active_params.get("day_loss_limit", self.risk.day_loss_limit) or self.risk.day_loss_limit
                ),
                week_loss_limit=float(
                    active_params.get("week_loss_limit", self.risk.week_loss_limit) or self.risk.week_loss_limit
                ),
                position_qty_by_symbol=position_qty_by_symbol,
            )
            status = PROPOSAL_STATUS_APPROVED if decision.allowed else PROPOSAL_STATUS_BLOCKED

            proposal_uuid = str(uuid.uuid4())
            proposal_id = self.storage.insert_order_proposal(
                proposal_uuid=proposal_uuid,
                agent_id=agent_id,
                market=market,
                session_date=effective_session_date,
                strategy_version_id=int(active["strategy_version_id"]),
                status=status,
                blocked_reason=decision.blocked_reason,
                orders=decision.accepted_orders,
                rationale=rationale,
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            if (not decision.allowed) and decision.blocked_reason:
                self.storage.log_event(
                    "risk_violation",
                    {
                        "agent_id": agent_id,
                        "market": market,
                        "session_date": effective_session_date,
                        "status_code": signal["status_code"],
                        "blocked_reason": decision.blocked_reason,
                    },
                    now_iso(),
                )
            self.identity.append_memory(
                agent_id=agent_id,
                memory_type="proposal",
                content={
                    "proposal_id": proposal_id,
                    "proposal_uuid": proposal_uuid,
                    "market": market,
                    "session_date": effective_session_date,
                    "status": status,
                    "blocked_reason": decision.blocked_reason,
                    "accepted_orders": decision.accepted_orders,
                    "dropped_orders": decision.dropped_orders,
                    "cross_market_plan": cross_market_plan,
                },
                ts=now_iso(),
            )
            output_rows.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_uuid": proposal_uuid,
                    "agent_id": agent_id,
                    "status": status,
                    "blocked_reason": decision.blocked_reason,
                    "orders": decision.accepted_orders,
                    "risk_metrics": decision.metrics,
                    "fx_rate": usdkrw,
                    "agent_signal_view": agent_signal_payload.get("agent_signal_view", {}),
                    "cross_market_plan": cross_market_plan,
                }
            )

        execution_results: List[Dict[str, Any]] = []
        if bool(auto_execute):
            for row in output_rows:
                if str(row.get("status", "")) != PROPOSAL_STATUS_APPROVED:
                    continue
                proposal_id = int(row["proposal_id"])
                try:
                    executed = self._execute_proposal(
                        proposal_identifier=str(proposal_id),
                        approved_by="auto_executor",
                        note=f"auto-approved market={market} date={effective_session_date}",
                        allowed_statuses=[PROPOSAL_STATUS_APPROVED],
                    )
                    row["status"] = str(executed.get("status", row["status"]))
                    row["auto_execution"] = {"ok": True, "status": row["status"]}
                    execution_results.append(
                        {
                            "proposal_id": proposal_id,
                            "agent_id": row.get("agent_id"),
                            "ok": True,
                            "status": row["status"],
                            "fills": executed.get("fills", []),
                        }
                    )
                except Exception as exc:
                    self.storage.update_order_proposal_status(
                        proposal_id=proposal_id,
                        status=PROPOSAL_STATUS_BLOCKED,
                        blocked_reason=f"auto_execute_error:{repr(exc)}",
                        updated_at=now_iso(),
                    )
                    row["status"] = PROPOSAL_STATUS_BLOCKED
                    row["auto_execution"] = {"ok": False, "error": repr(exc)}
                    execution_results.append(
                        {
                            "proposal_id": proposal_id,
                            "agent_id": row.get("agent_id"),
                            "ok": False,
                            "error": repr(exc),
                        }
                    )

        payload = {
            "market": market,
            "date": effective_session_date,
            "auto_execute": bool(auto_execute),
            "proposals": output_rows,
            "execution_results": execution_results,
        }
        self._write_json_artifact(
            effective_session_date,
            f"proposals_{market.lower()}_{effective_session_date}",
            payload,
        )
        self.storage.log_event("orders_proposed", payload, now_iso())
        self._append_activity_artifact(effective_session_date, "orders_proposed", payload)

        lines = [
            "[AgentLab] 주문 제안",
            f"시장={market}",
            f"일자={effective_session_date}",
            f"자동실행={bool(auto_execute)}",
        ]
        status_ko = {
            PROPOSAL_STATUS_APPROVED: "승인됨",
            PROPOSAL_STATUS_EXECUTED: "실행됨",
            PROPOSAL_STATUS_BLOCKED: "차단됨",
        }
        for row in output_rows:
            row_status = status_ko.get(str(row.get("status", "")), str(row.get("status", "")))
            lines.append(
                f"{row.get('agent_id')}: {row_status} | "
                f"{self._orders_to_text(list(row.get('orders') or []), limit=3, symbol_name_map=symbol_name_map)}"
            )
        if execution_results:
            ok_count = sum(1 for x in execution_results if bool(x.get("ok")))
            lines.append(f"자동실행_결과={ok_count}/{len(execution_results)} 성공")
        self._notify("\n".join(lines), event="propose")
        return payload

    def _execute_proposal(
        self,
        proposal_identifier: str,
        approved_by: str = "manual",
        note: str = "",
        *,
        allowed_statuses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        proposal = None
        if proposal_identifier.isdigit():
            proposal = self.storage.get_order_proposal_by_id(int(proposal_identifier))
        if proposal is None:
            proposal = self.storage.get_order_proposal_by_uuid(proposal_identifier)
        if proposal is None:
            raise ValueError(f"proposal not found: {proposal_identifier}")
        expected = allowed_statuses or [PROPOSAL_STATUS_APPROVED]
        current_status = str(proposal["status"])
        if current_status not in expected:
            raise RuntimeError(
                f"proposal status is not executable: {current_status} (allowed={','.join(expected)})"
            )

        proposal_id = int(proposal["proposal_id"])
        market = str(proposal["market"]).upper()
        yyyymmdd = str(proposal["session_date"])
        agent_id = str(proposal["agent_id"])
        if not self._is_order_market_enabled(market):
            enabled_markets = self._order_enabled_markets()
            block_reason = f"market_disabled_by_env:{','.join(enabled_markets)}"
            self.storage.update_order_proposal_status(
                proposal_id=proposal_id,
                status=PROPOSAL_STATUS_BLOCKED,
                blocked_reason=block_reason,
                updated_at=now_iso(),
            )
            blocked_payload = {
                "proposal_id": proposal_id,
                "proposal_uuid": proposal["proposal_uuid"],
                "agent_id": agent_id,
                "market": market,
                "session_date": yyyymmdd,
                "status": PROPOSAL_STATUS_BLOCKED,
                "blocked_reason": block_reason,
                "enabled_order_markets": enabled_markets,
            }
            self.storage.log_event("orders_approval_blocked", blocked_payload, now_iso())
            self._append_activity_artifact(yyyymmdd, "orders_approval_blocked", blocked_payload)
            return blocked_payload
        if self._enforce_market_hours():
            market_open, market_window_detail = self._is_market_open_now(market)
            if not market_open:
                block_reason = f"outside_market_hours:{market_window_detail}"
                self.storage.update_order_proposal_status(
                    proposal_id=proposal_id,
                    status=PROPOSAL_STATUS_BLOCKED,
                    blocked_reason=block_reason,
                    updated_at=now_iso(),
                )
                blocked_payload = {
                    "proposal_id": proposal_id,
                    "proposal_uuid": proposal["proposal_uuid"],
                    "agent_id": agent_id,
                    "market": market,
                    "session_date": yyyymmdd,
                    "status": PROPOSAL_STATUS_BLOCKED,
                    "blocked_reason": block_reason,
                }
                self.storage.log_event("orders_approval_blocked", blocked_payload, now_iso())
                self._append_activity_artifact(yyyymmdd, "orders_approval_blocked", blocked_payload)
                return blocked_payload

        fx_rate = 1.0
        if market == "US":
            rate = self._usdkrw_rate(yyyymmdd)
            if rate is None:
                self.storage.update_order_proposal_status(
                    proposal_id=proposal_id,
                    status=PROPOSAL_STATUS_BLOCKED,
                    blocked_reason="fx_rate_unavailable_over_3d",
                    updated_at=now_iso(),
                )
                blocked = {"proposal_id": proposal_id, "status": PROPOSAL_STATUS_BLOCKED}
                self._append_activity_artifact(yyyymmdd, "orders_approval_blocked", blocked)
                self._notify(
                    "[AgentLab] 거래 차단\n"
                    f"제안ID={proposal_id}\n"
                    f"에이전트={agent_id}\n"
                    "사유=fx_rate_unavailable_over_3d"
                    ,
                    event="trade_blocked",
                )
                return blocked
            fx_rate = float(rate)

        self.storage.insert_order_approval(
            proposal_id=proposal_id,
            approved_by=approved_by,
            approved_at=now_iso(),
            decision="APPROVE",
            note=note,
        )
        submit_results = self.paper_broker.execute_orders(
            proposal_id=proposal_id,
            agent_id=agent_id,
            market=market,
            orders=list(proposal["orders"]),
            fx_rate=fx_rate,
        )
        rejected_count = sum(
            1 for row in submit_results if str((row or {}).get("status", "")).upper() == "REJECTED"
        )
        if rejected_count > 0:
            self.storage.log_event(
                "broker_api_error",
                {
                    "proposal_id": proposal_id,
                    "agent_id": agent_id,
                    "market": market,
                    "session_date": yyyymmdd,
                    "rejected_count": rejected_count,
                },
                now_iso(),
            )
            self._notify(
                "[AgentLab] 주문 실행 오류\n"
                f"시장={market}\n"
                f"제안ID={proposal_id}\n"
                f"거부_주문수={rejected_count}",
                    event="broker_api_error",
                )
        symbol_name_map = self._load_symbol_name_map(market, yyyymmdd)
        sync_post: Optional[Dict[str, Any]] = None
        if self._unified_shadow_mode():
            sync_post = self._sync_after_execution_with_retry(self._execution_sync_scope())
        order_results = self._proposal_order_results(proposal_id)
        if not order_results:
            order_results = submit_results
        filled_count = sum(
            1 for row in order_results if str((row or {}).get("status", "")).upper() == "FILLED"
        )
        submitted_count = sum(
            1 for row in order_results if str((row or {}).get("status", "")).upper() == "SUBMITTED"
        )
        self.storage.update_order_proposal_status(
            proposal_id=proposal_id,
            status=PROPOSAL_STATUS_EXECUTED,
            blocked_reason="",
            updated_at=now_iso(),
        )
        ledger = self.accounting.upsert_daily_snapshot(agent_id=agent_id, as_of_date=yyyymmdd)
        out = {
            "proposal_id": proposal_id,
            "proposal_uuid": proposal["proposal_uuid"],
            "agent_id": agent_id,
            "status": PROPOSAL_STATUS_EXECUTED,
            "fills": order_results,
            "filled_count": int(filled_count),
            "submitted_count": int(submitted_count),
            "equity": ledger,
        }
        if sync_post is not None:
            out["sync_post_execution"] = sync_post
        self._write_json_artifact(yyyymmdd, f"approval_{proposal_id}", out)
        self.storage.log_event("orders_approved", out, now_iso())
        self._append_activity_artifact(yyyymmdd, "orders_approved", out)
        result_label = "결과"
        if filled_count > 0 and submitted_count == 0:
            result_label = "체결"
        elif submitted_count > 0:
            result_label = "주문"
        self._notify(
            "[AgentLab] 거래 실행\n"
            f"시장={market}\n"
            f"일자={yyyymmdd}\n"
            f"에이전트={agent_id}\n"
            f"제안ID={proposal_id}\n"
            f"{result_label}={self._orders_to_text(order_results, limit=5, symbol_name_map=symbol_name_map)}"
            ,
            event="trade_executed",
        )
        return out

    def approve_orders(self, proposal_identifier: str, approved_by: str = "manual", note: str = "") -> Dict[str, Any]:
        return self._execute_proposal(
            proposal_identifier=proposal_identifier,
            approved_by=approved_by,
            note=note,
            allowed_statuses=[PROPOSAL_STATUS_APPROVED],
        )

    def preopen_plan_report(self, market: str, yyyymmdd: str) -> Dict[str, Any]:
        market = str(market).strip().upper()
        if self._strict_report_windows_enabled():
            ok, detail = self._within_preopen_window(market)
            if not ok:
                return {
                    "market": market,
                    "date": yyyymmdd,
                    "skipped": True,
                    "reason": "outside_preopen_window",
                    "detail": detail,
                    "generated_at": now_iso(),
                }
        claimed = self.storage.try_claim_report_dispatch(
            event_type="preopen_plan",
            market=market,
            report_date=yyyymmdd,
            created_at=now_iso(),
        )
        if not claimed:
            return {
                "market": market,
                "date": yyyymmdd,
                "skipped": True,
                "reason": "preopen_already_claimed",
                "generated_at": now_iso(),
            }
        if self._event_already_reported("preopen_plan", market, yyyymmdd):
            return {
                "market": market,
                "date": yyyymmdd,
                "skipped": True,
                "reason": "preopen_already_reported",
                "generated_at": now_iso(),
            }
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")

        plan_rows: List[Dict[str, Any]] = []
        for agent in agents:
            aid = str(agent["agent_id"])
            profile = profile_from_agent_row(agent)
            active = self.registry.get_active_strategy(aid)
            latest = self._latest_proposal_by_agent_market(aid, market) or {}
            latest_eq = self.storage.query_one(
                """
                SELECT *
                FROM equity_curve
                WHERE agent_id = ?
                ORDER BY as_of_date DESC, equity_id DESC
                LIMIT 1
                """,
                (aid,),
            ) or {}
            orders = list(latest.get("orders") or [])

            fallback = {
                "plan": f"{profile.name} 기본 플랜: 장중 수급/리스크 기준으로 {market} 대응.",
                "focus_points": [
                    "초반 유동성 및 체결 강도 확인",
                    "리스크 한도 우선 점검",
                ],
                "risk_watch": [
                    "급격한 변동성 확대 구간 주의",
                ],
            }
            resp = self.llm.generate_json(
                system_prompt=(
                    "너는 트레이딩 에이전트의 장 시작 전 브리핑 작성기다. "
                    "반드시 JSON으로만 답하고 키는 plan, focus_points, risk_watch만 사용한다."
                ),
                user_prompt=(
                    f"market={market}\n"
                    f"date={yyyymmdd}\n"
                    f"agent_id={aid}\n"
                    f"role={profile.role}\n"
                    f"risk_style={profile.risk_style}\n"
                    f"active_params={json.dumps(active.get('params', {}), ensure_ascii=False)}\n"
                    f"latest_proposal_status={latest.get('status', '')}\n"
                    f"latest_orders={orders}\n"
                    f"latest_equity={json.dumps(latest_eq, ensure_ascii=False)}\n"
                    "오늘 장에서 실행할 핵심 계획을 간결하게 작성해라."
                ),
                fallback=fallback,
                temperature=0.25,
            )
            parsed = resp.get("result", fallback)
            if not isinstance(parsed, dict):
                parsed = dict(fallback)
            plan = str(parsed.get("plan", fallback["plan"]))
            focus_points = parsed.get("focus_points", fallback["focus_points"])
            risk_watch = parsed.get("risk_watch", fallback["risk_watch"])
            if not isinstance(focus_points, list):
                focus_points = list(fallback["focus_points"])
            if not isinstance(risk_watch, list):
                risk_watch = list(fallback["risk_watch"])

            row = {
                "agent_id": aid,
                "role": profile.role,
                "strategy_version": str(active.get("version_tag", "")),
                "latest_proposal_status": str(latest.get("status", "")),
                "latest_orders": orders,
                "plan": plan,
                "focus_points": [str(x) for x in focus_points[:5]],
                "risk_watch": [str(x) for x in risk_watch[:5]],
                "llm_mode": str(resp.get("mode", "fallback")),
                "llm_reason": str(resp.get("reason", "")),
            }
            plan_rows.append(row)
            self.identity.append_memory(
                agent_id=aid,
                memory_type="preopen_plan",
                content={
                    "market": market,
                    "date": yyyymmdd,
                    "plan": row["plan"],
                    "focus_points": row["focus_points"],
                    "risk_watch": row["risk_watch"],
                },
                ts=now_iso(),
            )

        fallback_summary = f"{market} 개장 전에는 유동성/체결강도와 리스크 한도를 우선 점검하고, 에이전트별 계획에 따라 대응한다."
        summary_text = self._llm_explain(
            topic="preopen_plan_report",
            context={"market": market, "date": yyyymmdd, "plans": plan_rows},
            fallback=fallback_summary,
        )

        payload = {
            "market": market,
            "date": yyyymmdd,
            "generated_at": now_iso(),
            "plans": plan_rows,
            "summary_text": summary_text,
        }
        self._write_json_artifact(yyyymmdd, f"preopen_plan_{market.lower()}_{yyyymmdd}", payload)
        md_lines = [
            f"# 개장 전 플랜 보고 ({market}) {yyyymmdd}",
            "",
        ]
        for row in plan_rows:
            md_lines.append(f"## {row['agent_id']} ({row['role']})")
            md_lines.append(f"- 전략 버전: `{row['strategy_version']}`")
            md_lines.append(f"- 최근 제안 상태: `{row['latest_proposal_status'] or '-'}'")
            md_lines.append(f"- 오늘 계획: {row['plan']}")
            md_lines.append("- 핵심 포인트:")
            if row["focus_points"]:
                for item in row["focus_points"]:
                    md_lines.append(f"  - {item}")
            else:
                md_lines.append("  - 없음")
            md_lines.append("- 리스크 주시:")
            if row["risk_watch"]:
                for item in row["risk_watch"]:
                    md_lines.append(f"  - {item}")
            else:
                md_lines.append("  - 없음")
            md_lines.append("")
        self._write_text_artifact(yyyymmdd, f"preopen_plan_{market.lower()}_{yyyymmdd}", "\n".join(md_lines))
        self.storage.log_event("preopen_plan", payload, now_iso())
        self._append_activity_artifact(yyyymmdd, "preopen_plan", payload)

        msg_lines = [
            "[AgentLab] 개장 10분 전 플랜 보고",
            f"시장={market}",
            f"일자={yyyymmdd}",
            f"요약={summary_text}",
        ]
        for row in plan_rows:
            focus = ", ".join(row["focus_points"][:2]) if row["focus_points"] else "-"
            risk = row["risk_watch"][0] if row["risk_watch"] else "-"
            msg_lines.append(f"{row['agent_id']}: {row['plan']}")
            msg_lines.append(f" - 핵심={focus}")
            msg_lines.append(f" - 리스크={risk}")
        self._notify("\n".join(msg_lines), event="preopen_plan")
        return payload

    def session_close_report(self, market: str, yyyymmdd: str) -> Dict[str, Any]:
        market = str(market).strip().upper()
        if self._strict_report_windows_enabled():
            ok, detail = self._within_close_report_window(market)
            if not ok:
                return {
                    "market": market,
                    "date": yyyymmdd,
                    "skipped": True,
                    "reason": "outside_close_report_window",
                    "detail": detail,
                    "generated_at": now_iso(),
                }
        claimed = self.storage.try_claim_report_dispatch(
            event_type="session_close_report",
            market=market,
            report_date=yyyymmdd,
            created_at=now_iso(),
        )
        if not claimed:
            return {
                "market": market,
                "date": yyyymmdd,
                "skipped": True,
                "reason": "close_report_already_claimed",
                "generated_at": now_iso(),
            }
        if self._event_already_reported("session_close_report", market, yyyymmdd):
            return {
                "market": market,
                "date": yyyymmdd,
                "skipped": True,
                "reason": "close_report_already_reported",
                "generated_at": now_iso(),
            }
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")

        day_rows: List[Dict[str, Any]] = []
        active_strategy_map: Dict[str, Dict[str, Any]] = {}
        agent_profiles: List[Any] = []
        if self._unified_shadow_mode():
            signal = self.storage.get_latest_session_signal(market, yyyymmdd) or {}
            payload_signal = signal.get("payload", {}) if isinstance(signal, dict) else {}
            ranked = list(payload_signal.get("all_ranked", []) or payload_signal.get("final", []) or [])
            symbol_change: Dict[str, float] = {}
            for row in ranked:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code", "")).strip().upper()
                metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
                change_pct = float(metrics.get("current_change_pct", row.get("change_pct", 0.0)) or 0.0)
                if code:
                    symbol_change[code] = change_pct / 100.0
            shadow_rows = self.storage.list_shadow_intents(market=market, session_date=yyyymmdd, limit=20000)
            by_agent_shadow: Dict[str, List[Dict[str, Any]]] = {}
            for row in shadow_rows:
                aid = str(row.get("agent_id", ""))
                if aid:
                    by_agent_shadow.setdefault(aid, []).append(row)
            account_ctx = self._latest_account_context("ALL")
            shared_equity = float(account_ctx.get("equity_krw", 0.0) or 0.0)
            shared_cash = float(account_ctx.get("cash_krw", 0.0) or 0.0)
            for agent in agents:
                aid = str(agent["agent_id"])
                intents = list(by_agent_shadow.get(aid, []) or [])
                pnl_proxy = 0.0
                sample = 0
                for intent in intents:
                    for order in list(intent.get("orders", []) or []):
                        symbol = str(order.get("symbol", "")).strip().upper()
                        side = str(order.get("side", "")).strip().upper()
                        qty = float(order.get("quantity", 0.0) or 0.0)
                        px = float(order.get("reference_price", 0.0) or 0.0)
                        if not symbol or qty <= 0 or px <= 0:
                            continue
                        direction = 1.0 if side == "BUY" else -1.0
                        pnl_proxy += direction * qty * px * float(symbol_change.get(symbol, 0.0))
                        sample += 1
                base_notional = max(1.0, sum(
                    float(order.get("quantity", 0.0) or 0.0) * float(order.get("reference_price", 0.0) or 0.0)
                    for intent in intents
                    for order in list(intent.get("orders", []) or [])
                ))
                ret_proxy = float(pnl_proxy / base_notional)
                risk_penalty = max(0.0, 0.05 if sample == 0 else 0.0)
                score = max(0.05, min(1.0, 0.5 + (ret_proxy * 2.0) - risk_penalty))
                self.storage.upsert_shadow_score(
                    agent_id=aid,
                    as_of_date=yyyymmdd,
                    score=score,
                    return_pct=ret_proxy,
                    risk_penalty=risk_penalty,
                    sample_count=sample,
                    created_at=now_iso(),
                )
                day_rows.append(
                    {
                        "agent_id": aid,
                        "equity_krw": float(shared_equity),
                        "cash_krw": float(shared_cash),
                        "return_pct": ret_proxy,
                        "drawdown": -risk_penalty,
                        "volatility": abs(ret_proxy) * 0.35,
                        "win_rate": max(0.0, min(1.0, 0.5 + ret_proxy * 4.0)),
                        "profit_factor": max(0.1, 1.0 + ret_proxy * 8.0),
                        "turnover": float(max(0, sample)),
                        "max_consecutive_loss": int(1 if ret_proxy < 0 else 0),
                    }
                )
                active_strategy_map[aid] = self.registry.get_active_strategy(aid)
                agent_profiles.append(profile_from_agent_row(agent))
        else:
            for agent in agents:
                aid = str(agent["agent_id"])
                ledger = self.accounting.upsert_daily_snapshot(agent_id=aid, as_of_date=yyyymmdd)
                day_ret, _week_ret = self.accounting.daily_and_weekly_return(aid, yyyymmdd)
                day_rows.append(
                    {
                        "agent_id": aid,
                        "equity_krw": float(ledger["equity_krw"]),
                        "cash_krw": float(ledger["cash_krw"]),
                        "return_pct": float(day_ret),
                        "drawdown": float(ledger["drawdown"]),
                        "volatility": float(ledger["volatility"]),
                        "win_rate": float(ledger["win_rate"]),
                        "profit_factor": float(ledger["profit_factor"]),
                        "turnover": float(ledger["turnover"]),
                        "max_consecutive_loss": int(ledger["max_consecutive_loss"]),
                    }
                )
                active_strategy_map[aid] = self.registry.get_active_strategy(aid)
                agent_profiles.append(profile_from_agent_row(agent))

        score_board = self._weekly_score_board(day_rows)
        try:
            dt = datetime.strptime(yyyymmdd, "%Y%m%d").date()
        except Exception:
            dt = datetime.now().date()
        week_start = dt - timedelta(days=dt.weekday())
        week_end = week_start + timedelta(days=6)

        scored_rows: List[Dict[str, Any]] = []
        for row in day_rows:
            aid = str(row["agent_id"])
            scored_rows.append(
                {
                    **row,
                    "score": float(score_board.get(aid, 0.0)),
                    "risk_violations": self._risk_violation_count(aid, week_start, week_end),
                }
            )
        scored_rows.sort(
            key=lambda r: (
                float(r["score"]),
                -float(r["turnover"]),
                -int(r["max_consecutive_loss"]),
                -float(r["risk_violations"]),
            ),
            reverse=True,
        )

        directives_map: Dict[str, List[str]] = {}
        for agent in agents:
            aid = str(agent["agent_id"])
            memories = self.storage.list_agent_memories(aid, limit=40)
            applied: List[str] = []
            for mem in reversed(memories):
                if str(mem.get("memory_type", "")) != "operator_directive_applied":
                    continue
                content = mem.get("content", {})
                if not isinstance(content, dict):
                    continue
                dtype = str(content.get("directive_type", "")).strip()
                if dtype == "param_update":
                    key = str(content.get("param_key", "")).strip()
                    val = content.get("applied_value", content.get("requested_value"))
                    text = f"param_update: {key}={val}"
                else:
                    text = str(content.get("request_text", "")).strip()
                if text:
                    applied.append(text)
            if applied:
                directives_map[aid] = applied[-5:]

        discussion = self.agent_engine.run_weekly_council_debate(
            agent_profiles=agent_profiles,
            active_params_map={aid: dict(v.get("params", {})) for aid, v in active_strategy_map.items()},
            scored_rows=scored_rows,
            score_board=score_board,
            week_id=f"{yyyymmdd}-{market}",
            operator_directives_map=directives_map,
        )

        proposals = self.storage.list_order_proposals(market=market, session_date=yyyymmdd)
        trade_summary: Dict[str, Dict[str, int]] = {}
        for row in proposals:
            aid = str(row.get("agent_id", ""))
            st = str(row.get("status", ""))
            trade_summary.setdefault(aid, {"EXECUTED": 0, "BLOCKED": 0, "APPROVED": 0})
            if st in trade_summary[aid]:
                trade_summary[aid][st] += 1

        moderator = discussion.get("moderator", {}) if isinstance(discussion, dict) else {}
        fallback_summary = str(moderator.get("summary", "")).strip() or (
            f"{market} 장 마감 기준으로 에이전트별 손익/낙폭과 실행 결과를 점검하고, 다음 세션 조정 포인트를 확정했다."
        )
        summary_text = self._llm_explain(
            topic="session_close_report",
            context={
                "market": market,
                "date": yyyymmdd,
                "rows_scored": scored_rows,
                "trade_summary": trade_summary,
                "moderator_summary": str(moderator.get("summary", "")),
            },
            fallback=fallback_summary,
        )

        payload = {
            "market": market,
            "date": yyyymmdd,
            "rows_scored": scored_rows,
            "score_board": score_board,
            "trade_summary": trade_summary,
            "discussion": discussion,
            "summary_text": summary_text,
            "generated_at": now_iso(),
        }
        self._write_json_artifact(yyyymmdd, f"session_close_report_{market.lower()}_{yyyymmdd}", payload)
        md_lines = [
            f"# 장 종료 10분 후 결과/토의 보고 ({market}) {yyyymmdd}",
            "",
            "|Agent|Equity(KRW)|DailyRet|Drawdown|Vol|WinRate|PF|Turnover|Score|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in scored_rows:
            md_lines.append(
                f"|{r['agent_id']}|{float(r['equity_krw']):.0f}|{float(r['return_pct']):.4f}|"
                f"{float(r['drawdown']):.4f}|{float(r['volatility']):.4f}|{float(r['win_rate']):.4f}|"
                f"{float(r['profit_factor']):.4f}|{float(r['turnover']):.4f}|{float(r['score']):.4f}|"
            )
        md_lines.append("")
        md_lines.append("## 토의 요약")
        md_lines.append(summary_text)
        md_lines.append("")
        rounds = list(discussion.get("rounds", [])) if isinstance(discussion, dict) else []
        for round_row in rounds[:2]:
            md_lines.append(f"### 라운드 {round_row.get('round')} ({round_row.get('phase')})")
            for speech in list(round_row.get("speeches") or [])[:3]:
                aid = str(speech.get("agent_id", ""))
                text = str(speech.get("thesis") or speech.get("rebuttal") or "")
                md_lines.append(f"- {aid}: {text}")
            md_lines.append("")
        self._write_text_artifact(yyyymmdd, f"session_close_report_{market.lower()}_{yyyymmdd}", "\n".join(md_lines))
        self.storage.log_event("session_close_report", payload, now_iso())
        self._append_activity_artifact(yyyymmdd, "session_close_report", payload)

        msg_lines = [
            "[AgentLab] 장 종료 10분 후 결과/토의 보고",
            f"시장={market}",
            f"일자={yyyymmdd}",
            f"요약={summary_text}",
        ]
        for r in scored_rows:
            aid = str(r["agent_id"])
            t = trade_summary.get(aid, {})
            msg_lines.append(
                f"{aid}: 수익률={float(r['return_pct']):.4f}, 낙폭={float(r['drawdown']):.4f}, "
                f"평가자산={float(r['equity_krw']):.0f}, 실행제안={int(t.get('EXECUTED', 0))}"
            )
        summary = summary_text.strip()
        discussion_summary = str(moderator.get("summary", "")).strip()
        if discussion_summary and discussion_summary != summary:
            if len(discussion_summary) > 220:
                discussion_summary = discussion_summary[:220] + "..."
            msg_lines.append(f"토의요약={discussion_summary}")
        self._notify("\n".join(msg_lines), event="session_close_report")
        return payload

    def daily_review(self, yyyymmdd: str) -> Dict[str, Any]:
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        rows: List[Dict[str, Any]] = []
        for agent in agents:
            agent_id = str(agent["agent_id"])
            ledger = self.accounting.upsert_daily_snapshot(agent_id=agent_id, as_of_date=yyyymmdd)
            rows.append(
                {
                    "agent_id": agent_id,
                    "equity_krw": ledger["equity_krw"],
                    "cash_krw": ledger["cash_krw"],
                    "drawdown": ledger["drawdown"],
                    "volatility": ledger["volatility"],
                    "win_rate": ledger["win_rate"],
                    "profit_factor": ledger["profit_factor"],
                    "turnover": ledger["turnover"],
                    "max_consecutive_loss": ledger["max_consecutive_loss"],
                }
            )
            self.identity.append_memory(
                agent_id=agent_id,
                memory_type="daily_review",
                content={"date": yyyymmdd, "metrics": rows[-1]},
                ts=now_iso(),
            )

        markdown_lines = [f"# Daily Review {yyyymmdd}", "", "|Agent|Equity(KRW)|Drawdown|Volatility|WinRate|PF|Turnover|", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            markdown_lines.append(
                f"|{r['agent_id']}|{r['equity_krw']:.0f}|{r['drawdown']:.4f}|{r['volatility']:.4f}|"
                f"{r['win_rate']:.4f}|{r['profit_factor']:.4f}|{r['turnover']:.4f}|"
            )
        markdown = "\n".join(markdown_lines)
        payload = {"date": yyyymmdd, "rows": rows}
        self.storage.upsert_daily_review(yyyymmdd, payload, markdown, now_iso())
        self._write_json_artifact(yyyymmdd, f"daily_review_{yyyymmdd}", payload)
        self._write_text_artifact(yyyymmdd, f"daily_review_{yyyymmdd}", markdown)
        self.storage.log_event("daily_review", payload, now_iso())
        self._append_activity_artifact(yyyymmdd, "daily_review", payload)
        lines = [f"[AgentLab] 일간 리뷰", f"일자={yyyymmdd}"]
        for r in rows:
            lines.append(
                f"{r['agent_id']}: 평가자산={float(r['equity_krw']):.0f}, "
                f"낙폭={float(r['drawdown']):.4f}, 회전율={float(r['turnover']):.4f}"
            )
        self._notify("\n".join(lines), event="daily_review")
        return payload

    def _weekly_score_board(self, week_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not week_rows:
            return {}
        normalized_rows: List[Dict[str, Any]] = []
        for row in week_rows:
            copied = dict(row)
            copied["_drawdown_loss"] = max(0.0, -float(row.get("drawdown", 0.0) or 0.0))
            normalized_rows.append(copied)

        metrics = ["return_pct", "_drawdown_loss", "volatility", "win_rate", "profit_factor"]
        bounds: Dict[str, Tuple[float, float]] = {}
        for m in metrics:
            vals = [float(r[m]) for r in normalized_rows]
            bounds[m] = (min(vals), max(vals))

        def norm(val: float, m: str, reverse: bool = False) -> float:
            lo, hi = bounds[m]
            if abs(hi - lo) < 1e-12:
                return 0.5
            score = (val - lo) / (hi - lo)
            return 1.0 - score if reverse else score

        score_board: Dict[str, float] = {}
        for r in normalized_rows:
            s = (
                0.40 * norm(float(r["return_pct"]), "return_pct")
                + 0.25 * norm(float(r["_drawdown_loss"]), "_drawdown_loss", reverse=True)
                + 0.15 * norm(float(r["volatility"]), "volatility", reverse=True)
                + 0.10 * norm(float(r["win_rate"]), "win_rate")
                + 0.10 * norm(float(r["profit_factor"]), "profit_factor")
            )
            score_board[str(r["agent_id"])] = s
        return score_board

    @staticmethod
    def _render_weekly_markdown(decision: Dict[str, Any]) -> str:
        if not isinstance(decision, dict):
            return "# 주간 회의\n\n- 유효한 의사결정 데이터가 없습니다.\n"

        week_id = str(decision.get("week_id", "") or "")
        champion = str(decision.get("champion_agent_id", "") or "")
        score_board = decision.get("score_board", {})
        promoted_versions = decision.get("promoted_versions", {})
        apply_mode = str(decision.get("promotion_apply_mode", "") or "")
        llm_alerts = list(decision.get("llm_alerts", []) or [])
        discussion = decision.get("discussion", {}) if isinstance(decision.get("discussion"), dict) else {}
        moderator = discussion.get("moderator", {}) if isinstance(discussion.get("moderator"), dict) else {}
        summary = str(moderator.get("summary", "") or "")

        lines = [
            f"# 주간 회의 {week_id}",
            "",
            f"- 우승 전략: `{champion or '(없음)'}`",
            f"- 점수표: `{score_board}`",
            f"- 승격 버전: `{promoted_versions if promoted_versions else '(없음)'}`",
        ]
        if apply_mode:
            lines.append(f"- 반영 모드: `{apply_mode}`")
        lines.append(f"- LLM 경고 수: `{len(llm_alerts)}`")
        lines.extend(
            [
                "",
                "## 사회자 요약",
                summary,
            ]
        )

        backfill = decision.get("backfill", {})
        if isinstance(backfill, dict):
            applied_at = str(backfill.get("applied_at", "") or "").strip()
            if applied_at:
                lines.extend(
                    [
                        "",
                        f"- 백필 적용: `{applied_at}` (주간회의 즉시반영 패치 이전 결과 1회 적용)",
                    ]
                )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _discussion_has_live_content(discussion: Dict[str, Any]) -> bool:
        if not isinstance(discussion, dict):
            return False
        rounds = discussion.get("rounds", [])
        if isinstance(rounds, list):
            for round_row in rounds:
                if not isinstance(round_row, dict):
                    continue
                speeches = round_row.get("speeches", [])
                if not isinstance(speeches, list):
                    continue
                for speech in speeches:
                    if not isinstance(speech, dict):
                        continue
                    if str(speech.get("mode", "")).strip().lower() == "live":
                        return True
        moderator = discussion.get("moderator", {})
        if isinstance(moderator, dict):
            if str(moderator.get("mode", "")).strip().lower() == "live":
                return True
        return False

    def weekly_council(self, week_id: str) -> Dict[str, Any]:
        if not self._allow_offschedule_weekly():
            ok, detail = self._within_weekly_window()
            if not ok:
                return {
                    "week_id": week_id,
                    "skipped": True,
                    "reason": "outside_weekly_window",
                    "detail": detail,
                    "generated_at": now_iso(),
                }
        existing = self.storage.query_one(
            """
            SELECT weekly_council_id
            FROM weekly_councils
            WHERE week_id = ?
            LIMIT 1
            """,
            (str(week_id),),
        )
        if existing:
            return {
                "week_id": week_id,
                "skipped": True,
                "reason": "weekly_already_reported",
                "generated_at": now_iso(),
            }
        week_start, week_end = parse_week_id(week_id)
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        week_rows: List[Dict[str, Any]] = []
        if self._unified_shadow_mode():
            score_rows = self.storage.list_shadow_scores(
                date_from=week_start.strftime("%Y%m%d"),
                date_to=week_end.strftime("%Y%m%d"),
                limit=50000,
            )
            by_agent_scores: Dict[str, List[Dict[str, Any]]] = {}
            for row in score_rows:
                aid = str(row.get("agent_id", ""))
                if aid:
                    by_agent_scores.setdefault(aid, []).append(dict(row))
            for agent in agents:
                aid = str(agent["agent_id"])
                entries = sorted(by_agent_scores.get(aid, []), key=lambda x: str(x.get("as_of_date", "")))
                if not entries:
                    continue
                avg_ret = sum(float(x.get("return_pct", 0.0) or 0.0) for x in entries) / float(len(entries))
                avg_penalty = sum(float(x.get("risk_penalty", 0.0) or 0.0) for x in entries) / float(len(entries))
                avg_score = sum(float(x.get("score", 0.0) or 0.0) for x in entries) / float(len(entries))
                row = {
                    "agent_id": aid,
                    "return_pct": float(avg_ret),
                    "drawdown": float(-max(0.0, avg_penalty)),
                    "volatility": float(abs(avg_ret) * 0.35),
                    "win_rate": float(min(1.0, max(0.0, 0.5 + avg_ret * 4.0))),
                    "profit_factor": float(max(0.1, 1.0 + avg_ret * 8.0)),
                    "turnover": float(max(0.01, len(entries))),
                    "max_consecutive_loss": int(max(0.0, avg_penalty * 10.0)),
                    "shadow_avg_score": float(avg_score),
                }
                week_rows.append(row)
        else:
            for agent in agents:
                aid = str(agent["agent_id"])
                curve = self.storage.list_equity_curve(
                    agent_id=aid,
                    date_from=week_start.strftime("%Y%m%d"),
                    date_to=week_end.strftime("%Y%m%d"),
                )
                if not curve:
                    continue
                curve = sorted(curve, key=lambda x: str(x["as_of_date"]))
                first_eq = float(curve[0]["equity_krw"])
                last_eq = float(curve[-1]["equity_krw"])
                ret = ((last_eq / first_eq) - 1.0) if first_eq > 0 else 0.0
                row = {
                    "agent_id": aid,
                    "return_pct": ret,
                    "drawdown": float(min(float(x["drawdown"]) for x in curve)),
                    "volatility": float(curve[-1]["volatility"]),
                    "win_rate": float(curve[-1]["win_rate"]),
                    "profit_factor": float(curve[-1]["profit_factor"]),
                    "turnover": float(curve[-1]["turnover"]),
                    "max_consecutive_loss": int(curve[-1]["max_consecutive_loss"]),
                }
                week_rows.append(row)

        scores = self._weekly_score_board(week_rows)
        scored_rows: List[Dict[str, Any]] = []
        risk_stats_by_agent: Dict[str, Dict[str, float]] = {}
        for row in week_rows:
            aid = str(row["agent_id"])
            stats = self._risk_violation_stats(aid, week_start, week_end)
            risk_stats_by_agent[aid] = dict(stats)
            scored_rows.append(
                {
                    **row,
                    "score": float(scores.get(aid, 0.0)),
                    "risk_violations": int(stats["risk_violations"]),
                    "proposal_count": int(stats["proposal_count"]),
                    "risk_violation_rate": float(stats["risk_violation_rate"]),
                }
            )
        scored_rows.sort(
            key=lambda r: (
                float(r["score"]),
                -float(r["turnover"]),
                -int(r["max_consecutive_loss"]),
                -float(r["risk_violations"]),
            ),
            reverse=True,
        )
        champion = str(scored_rows[0]["agent_id"]) if scored_rows else ""
        promoted_versions: Dict[str, str] = {}
        active_strategy_map: Dict[str, Dict[str, Any]] = {}
        agent_profiles: List[Any] = []
        for agent in agents:
            aid = str(agent["agent_id"])
            active_strategy_map[aid] = self.registry.get_active_strategy(aid)
            agent_profiles.append(profile_from_agent_row(agent))

        operator_directives_map: Dict[str, List[str]] = {}
        for agent in agents:
            aid = str(agent["agent_id"])
            memories = self.storage.list_agent_memories(aid, limit=40)
            applied: List[str] = []
            for mem in reversed(memories):
                if str(mem.get("memory_type", "")) != "operator_directive_applied":
                    continue
                content = mem.get("content", {})
                if not isinstance(content, dict):
                    continue
                dtype = str(content.get("directive_type", "")).strip()
                if dtype == "param_update":
                    key = str(content.get("param_key", "")).strip()
                    val = content.get("applied_value", content.get("requested_value"))
                    text = f"param_update: {key}={val}"
                else:
                    text = str(content.get("request_text", "")).strip()
                if text:
                    applied.append(text)
            if applied:
                operator_directives_map[aid] = applied[-5:]

        discussion = self.agent_engine.run_weekly_council_debate(
            agent_profiles=agent_profiles,
            active_params_map={aid: dict(v["params"]) for aid, v in active_strategy_map.items()},
            scored_rows=scored_rows,
            score_board=scores,
            week_id=week_id,
            operator_directives_map=operator_directives_map,
        )
        llm_alerts = list(discussion.get("llm_warnings", []) or [])
        for alert in llm_alerts:
            reason = str(alert.get("reason", "") or "")
            if "daily_budget_exceeded" in reason:
                self.storage.log_event(
                    "llm_budget_alert",
                    {
                        "week_id": week_id,
                        "agent_id": str(alert.get("agent_id", "")),
                        "phase": str(alert.get("phase", "")),
                        "reason": reason,
                    },
                    now_iso(),
                )
        if not self._discussion_has_live_content(discussion):
            skip_payload = {
                "week_id": week_id,
                "skipped": True,
                "reason": "weekly_council_llm_unavailable",
                "llm_alerts": llm_alerts,
                "generated_at": now_iso(),
            }
            self.storage.log_event("weekly_council_skipped", skip_payload, now_iso())
            self._append_activity_artifact(f"{week_end.strftime('%Y%m%d')}_weekly", "weekly_council_skipped", skip_payload)
            self._notify(
                "[AgentLab] 주간 회의 스킵\n"
                f"주차={week_id}\n"
                "사유=LLM 연결 실패(실토론 미생성)\n"
                "조치=네트워크 복구 후 재실행 필요",
                event="llm_limit_alert",
            )
            return skip_payload

        prev_week = (week_start - timedelta(days=1)).isocalendar()
        prev_week_id = f"{prev_week.year}-W{prev_week.week:02d}"
        prev_decision = self.storage.query_one(
            "SELECT decision_json FROM weekly_councils WHERE week_id = ?",
            (prev_week_id,),
        )
        prev_scores: Dict[str, float] = {}
        if prev_decision:
            try:
                decoded = json.loads(str(prev_decision.get("decision_json") or "{}"))
                if isinstance(decoded, dict):
                    prev_scores = dict(decoded.get("score_board", {}))
            except Exception:
                prev_scores = {}

        weekly_stats_by_agent: Dict[str, Dict[str, Dict[str, float]]] = {}
        for agent in agents:
            aid = str(agent["agent_id"])
            prev1_start, prev1_end = self._shift_week(week_start, 1)
            prev2_start, prev2_end = self._shift_week(week_start, 2)
            weekly_stats_by_agent[aid] = {
                "w0": dict(risk_stats_by_agent.get(aid, self._risk_violation_stats(aid, week_start, week_end))),
                "w1": self._risk_violation_stats(aid, prev1_start, prev1_end),
                "w2": self._risk_violation_stats(aid, prev2_start, prev2_end),
            }

        forced_conservative_agents: List[str] = []
        promotion_evaluation: Dict[str, Dict[str, Any]] = {}
        apply_mode = self._weekly_apply_mode()
        for agent in agents:
            aid = str(agent["agent_id"])
            active = active_strategy_map[aid]
            row = next((x for x in scored_rows if x["agent_id"] == aid), {})
            suggested_map = discussion.get("agent_param_suggestions", {}) or {}
            params = dict(suggested_map.get(aid) or active["params"])

            stats = weekly_stats_by_agent.get(aid, {})
            w0 = dict(stats.get("w0", {}))
            w1 = dict(stats.get("w1", {}))
            w2 = dict(stats.get("w2", {}))

            current_score = float(scores.get(aid, 0.0))
            prev_score = float(prev_scores.get(aid, 0.0))
            current_risk = int(w0.get("risk_violations", 0) or 0)
            prev_risk_count = int(w1.get("risk_violations", 0) or 0)
            current_rate = float(w0.get("risk_violation_rate", 0.0) or 0.0)
            prev_rate = float(w1.get("risk_violation_rate", 0.0) or 0.0)
            current_props = int(w0.get("proposal_count", 0) or 0)
            prev_props = int(w1.get("proposal_count", 0) or 0)

            normal_promote = (
                current_score >= PROMOTION_SCORE_THRESHOLD
                and prev_score >= PROMOTION_SCORE_THRESHOLD
                and current_risk <= PROMOTION_MAX_RISK_VIOLATIONS
                and prev_risk_count <= PROMOTION_MAX_RISK_VIOLATIONS
                and current_rate <= PROMOTION_MAX_RISK_RATE
                and prev_rate <= PROMOTION_MAX_RISK_RATE
                and current_props >= PROMOTION_MIN_PROPOSALS
                and prev_props >= PROMOTION_MIN_PROPOSALS
            )
            force_conservative = (
                self._is_high_risk_week(w0)
                and self._is_high_risk_week(w1)
                and self._is_high_risk_week(w2)
            )

            if force_conservative:
                params = self._build_forced_conservative_params(params)
                forced_conservative_agents.append(aid)

            legacy_rule_promote = bool(force_conservative or normal_promote)
            if apply_mode == "immediate":
                promote = True
                mode = "forced_conservative_promote" if force_conservative else "immediate_promote"
            else:
                promote = legacy_rule_promote
                mode = "normal_promote" if normal_promote else "hold"
                if force_conservative:
                    mode = "forced_conservative_promote"

            promotion_evaluation[aid] = {
                "mode": mode,
                "normal_promote": bool(normal_promote),
                "force_conservative": bool(force_conservative),
                "legacy_rule_promote": bool(legacy_rule_promote),
                "applied_immediately": bool(apply_mode == "immediate"),
                "score_current": current_score,
                "score_prev": prev_score,
                "risk_violations_current": current_risk,
                "risk_violations_prev": prev_risk_count,
                "risk_violation_rate_current": current_rate,
                "risk_violation_rate_prev": prev_rate,
                "proposal_count_current": current_props,
                "proposal_count_prev": prev_props,
            }

            reg = self.registry.register_strategy_version(
                agent_id=aid,
                params=params,
                notes=(
                    f"weekly council update ({week_id}); "
                    f"score={current_score:.4f}, prev_score={prev_score:.4f}, "
                    f"risk={current_risk}, prev_risk={prev_risk_count}, "
                    f"rate={current_rate:.4f}, prev_rate={prev_rate:.4f}, "
                    f"proposals={current_props}, prev_proposals={prev_props}, "
                    f"mode={mode}, promote={promote}"
                ),
                promote=promote,
            )
            if promote:
                promoted_versions[aid] = reg["version_tag"]

        decision = {
            "week_id": week_id,
            "champion_agent_id": champion,
            "score_board": scores,
            "rows": week_rows,
            "rows_scored": scored_rows,
            "promoted_versions": promoted_versions,
            "promotion_apply_mode": apply_mode,
            "promotion_score_threshold": PROMOTION_SCORE_THRESHOLD,
            "promotion_rule": (
                "normal: score_current>=0.60 && score_prev>=0.60 && "
                "risk_current<=3 && risk_prev<=3 && rate_current<=0.15 && rate_prev<=0.15 && "
                "proposal_current>=10 && proposal_prev>=10; "
                "forced_conservative: 3-week consecutive high-risk "
                "(violations>=4 OR (proposals>=10 && rate>=0.25)); "
                "apply_mode=immediate -> weekly consensus params are activated immediately for all agents"
            ),
            "promotion_policy": {
                "apply_mode": apply_mode,
                "score_threshold": PROMOTION_SCORE_THRESHOLD,
                "max_risk_violations": PROMOTION_MAX_RISK_VIOLATIONS,
                "max_risk_rate": PROMOTION_MAX_RISK_RATE,
                "min_proposals": PROMOTION_MIN_PROPOSALS,
                "forced_consecutive_weeks": FORCED_CONSERVATIVE_CONSECUTIVE_WEEKS,
                "forced_min_violations": FORCED_CONSERVATIVE_MIN_VIOLATIONS,
                "forced_min_rate": FORCED_CONSERVATIVE_MIN_RATE,
            },
            "forced_conservative_agents": forced_conservative_agents,
            "weekly_risk_stats_by_agent": weekly_stats_by_agent,
            "promotion_evaluation": promotion_evaluation,
            "discussion": discussion,
            "llm_alerts": llm_alerts,
        }
        markdown = self._render_weekly_markdown(decision)
        stamp = week_end.strftime("%Y%m%d")
        weekly_stamp = f"{stamp}_weekly"
        self.storage.upsert_weekly_council(week_id, champion, decision, markdown, now_iso())
        self.identity.save_checkpoint(week_id, decision)
        self.identity.update_running_lessons(week_id, decision)
        self._write_json_artifact(weekly_stamp, f"weekly_council_{week_id.replace('-', '_')}", decision)
        self._write_text_artifact(weekly_stamp, f"weekly_council_{week_id.replace('-', '_')}", markdown)
        self.storage.log_event("weekly_council", decision, now_iso())
        self._append_activity_artifact(weekly_stamp, "weekly_council", decision)
        lines = [
            "[AgentLab] 주간 회의",
            f"주차={week_id}",
            f"우승전략={champion}",
            f"승격={promoted_versions if promoted_versions else '(없음)'}",
        ]
        rounds = list((discussion.get("rounds") or []))
        for round_row in rounds[:2]:
            phase = str(round_row.get("phase", ""))
            lines.append(f"라운드={round_row.get('round')} 단계={phase}")
            speeches = list(round_row.get("speeches") or [])
            for speech in speeches[:3]:
                aid = str(speech.get("agent_id", ""))
                text = str(speech.get("thesis") or speech.get("rebuttal") or "")
                if len(text) > 120:
                    text = text[:120] + "..."
                if text:
                    lines.append(f"{aid}: {text}")
        self._notify("\n".join(lines), event="weekly_council")
        return decision

    def report(self, date_from: str, date_to: str) -> Dict[str, Any]:
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        rows: List[Dict[str, Any]] = []
        for agent in agents:
            aid = str(agent["agent_id"])
            curve = self.storage.list_equity_curve(agent_id=aid, date_from=date_from, date_to=date_to)
            if not curve:
                rows.append({"agent_id": aid, "has_data": False})
                continue
            first_eq = float(curve[0]["equity_krw"])
            last_eq = float(curve[-1]["equity_krw"])
            ret = ((last_eq / first_eq) - 1.0) if first_eq > 0 else 0.0
            rows.append(
                {
                    "agent_id": aid,
                    "has_data": True,
                    "equity_start": first_eq,
                    "equity_end": last_eq,
                    "return_pct": ret,
                    "mdd": min(float(x["drawdown"]) for x in curve),
                    "volatility": float(curve[-1]["volatility"]),
                    "win_rate": float(curve[-1]["win_rate"]),
                    "profit_factor": float(curve[-1]["profit_factor"]),
                    "turnover": float(curve[-1]["turnover"]),
                    "max_consecutive_loss": int(curve[-1]["max_consecutive_loss"]),
                }
            )
        payload = {"from": date_from, "to": date_to, "rows": rows}
        out_day = self._today_str()
        self._write_json_artifact(out_day, f"report_{date_from}_{date_to}", payload)
        lines = [f"# Agent Lab Report {date_from}~{date_to}", "", "|Agent|Return|MDD|Vol|WinRate|PF|Turnover|", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if not r.get("has_data"):
                lines.append(f"|{r['agent_id']}|N/A|N/A|N/A|N/A|N/A|N/A|")
            else:
                lines.append(
                    f"|{r['agent_id']}|{r['return_pct']:.4f}|{r['mdd']:.4f}|{r['volatility']:.4f}|"
                    f"{r['win_rate']:.4f}|{r['profit_factor']:.4f}|{r['turnover']:.4f}|"
                )
        self._write_text_artifact(out_day, f"report_{date_from}_{date_to}", "\n".join(lines))
        self.storage.log_event("report_generated", payload, now_iso())
        self._append_activity_artifact(out_day, "report_generated", payload)
        self._notify(
            "[AgentLab] 리포트\n"
            f"시작일={date_from}\n"
            f"종료일={date_to}\n"
            f"행수={len(rows)}"
            ,
            event="report",
        )
        return payload

    def shadow_report(self, date_from: str, date_to: str) -> Dict[str, Any]:
        rows = self.storage.list_shadow_scores(date_from=date_from, date_to=date_to, limit=20000)
        by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            aid = str(row.get("agent_id", ""))
            if not aid:
                continue
            by_agent.setdefault(aid, []).append(dict(row))
        summary_rows: List[Dict[str, Any]] = []
        for aid, items in by_agent.items():
            items = sorted(items, key=lambda x: str(x.get("as_of_date", "")))
            n = float(len(items))
            avg_score = sum(float(x.get("score", 0.0) or 0.0) for x in items) / n
            avg_ret = sum(float(x.get("return_pct", 0.0) or 0.0) for x in items) / n
            avg_penalty = sum(float(x.get("risk_penalty", 0.0) or 0.0) for x in items) / n
            summary_rows.append(
                {
                    "agent_id": aid,
                    "samples": int(len(items)),
                    "avg_score": float(avg_score),
                    "avg_return_pct": float(avg_ret),
                    "avg_risk_penalty": float(avg_penalty),
                    "latest_date": str(items[-1].get("as_of_date", "")) if items else "",
                }
            )
        summary_rows.sort(key=lambda x: float(x.get("avg_score", 0.0) or 0.0), reverse=True)
        payload = {
            "from": date_from,
            "to": date_to,
            "execution_model": self._execution_model(),
            "rows": summary_rows,
        }
        out_day = self._today_str()
        self._write_json_artifact(out_day, f"shadow_report_{date_from}_{date_to}", payload)
        lines = [
            f"# Shadow Report {date_from}~{date_to}",
            "",
            "|Agent|Samples|AvgScore|AvgRet|AvgRiskPenalty|Latest|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary_rows:
            lines.append(
                f"|{row['agent_id']}|{int(row['samples'])}|{float(row['avg_score']):.4f}|"
                f"{float(row['avg_return_pct']):.4f}|{float(row['avg_risk_penalty']):.4f}|{row['latest_date']}|"
            )
        self._write_text_artifact(out_day, f"shadow_report_{date_from}_{date_to}", "\n".join(lines))
        self.storage.log_event("shadow_report_generated", payload, now_iso())
        self._append_activity_artifact(out_day, "shadow_report_generated", payload)
        return payload

    def cutover_reset(
        self,
        *,
        require_flat: bool = True,
        archive: bool = True,
        reinit: bool = True,
        restart_tasks: bool = False,
    ) -> Dict[str, Any]:
        sync = self.sync_account(market="ALL", strict=False)
        ctx = self._latest_account_context("ALL")
        server_equity_krw = float(ctx.get("equity_krw", 0.0) or 0.0)
        server_cash_krw = float(ctx.get("cash_krw", 0.0) or 0.0)
        open_positions = [
            row for row in list(ctx.get("positions", []) or [])
            if float(row.get("quantity", 0.0) or 0.0) > 0
        ]
        if require_flat and open_positions:
            payload = {
                "skipped": True,
                "reason": "positions_not_flat",
                "require_flat": True,
                "open_positions": open_positions[:20],
                "open_positions_count": len(open_positions),
                "sync": sync,
                "generated_at": now_iso(),
            }
            self.storage.log_event("cutover_reset_skipped", payload, now_iso())
            self._notify(
                "[AgentLab] 컷오버 중단\n"
                "사유=서버 계좌에 잔여 포지션 존재\n"
                f"잔여_종목수={len(open_positions)}",
                event="sync_mismatch",
            )
            return payload

        active_agents = self.storage.list_agents()
        legacy_allocated_sum = float(sum(float(x.get("allocated_capital_krw", 0.0) or 0.0) for x in active_agents))
        if server_equity_krw > 0.0:
            init_capital = server_equity_krw
            init_capital_source = "server_equity_krw"
        elif server_cash_krw > 0.0:
            init_capital = server_cash_krw
            init_capital_source = "server_cash_krw"
        else:
            init_capital = legacy_allocated_sum
            init_capital_source = "legacy_allocated_sum"
        init_agents = int(len(active_agents))
        ts = datetime.now(self.kst).strftime("%Y%m%d_%H%M%S")
        archive_root = self.project_root / "archive" / "agent_lab" / ts
        moved: List[Dict[str, str]] = []

        def _safe_move(src: Path, dst: Path) -> None:
            if not src.exists():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append({"from": str(src), "to": str(dst)})

        if archive:
            self.close()
            _safe_move(self.db_path, archive_root / "state" / "agent_lab.sqlite")
            _safe_move(self.out_root, archive_root / "out" / "agent_lab")
            _safe_move(self.project_root / "logs" / "agent_lab", archive_root / "logs" / "agent_lab")
            _safe_move(self.state_root / "agents", archive_root / "state" / "agents")
            _safe_move(self.state_root / "checkpoints", archive_root / "state" / "checkpoints")

            self.state_root.mkdir(parents=True, exist_ok=True)
            self.out_root.mkdir(parents=True, exist_ok=True)
            (self.project_root / "logs" / "agent_lab").mkdir(parents=True, exist_ok=True)
            self._reload_components()
        else:
            # Even in non-archive mode, create a new epoch and clear latest memories.
            for row in active_agents:
                aid = str(row.get("agent_id", ""))
                if not aid:
                    continue
                mem_path = self.identity.memory_path(aid)
                if mem_path.exists():
                    mem_path.unlink(missing_ok=True)

        new_epoch = f"epoch_{ts}"
        self.storage.upsert_system_meta("epoch_id", new_epoch, now_iso())
        self.storage.upsert_system_meta("execution_model", self._execution_model(), now_iso())
        reinit_payload: Dict[str, Any] = {}
        if reinit and init_agents > 0:
            reinit_payload = self.init_lab(capital_krw=max(0.0, init_capital), agents=init_agents)

        restart_result: Dict[str, Any] = {}
        if restart_tasks:
            script_path = self.project_root / "scripts" / "reset_tasks_preserve_state_wsl.sh"
            try:
                proc = subprocess.run(
                    ["/usr/bin/env", "bash", str(script_path)],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                restart_result = {
                    "ok": int(proc.returncode) == 0,
                    "returncode": int(proc.returncode),
                    "stdout_tail": str(proc.stdout or "")[-500:],
                    "stderr_tail": str(proc.stderr or "")[-500:],
                }
            except Exception as exc:
                restart_result = {"ok": False, "error": repr(exc)}

        payload = {
            "cutover_at": now_iso(),
            "archive": bool(archive),
            "reinit": bool(reinit),
            "restart_tasks": bool(restart_tasks),
            "archive_root": str(archive_root) if archive else "",
            "moved": moved,
            "new_epoch_id": new_epoch,
            "server_equity_krw": server_equity_krw,
            "server_cash_krw": server_cash_krw,
            "init_capital_source": init_capital_source,
            "reinit_payload": reinit_payload,
            "restart_result": restart_result,
        }
        self.storage.log_event("cutover_reset", payload, now_iso())
        self._append_activity_artifact(self._today_str(), "cutover_reset", payload)
        self._notify(
            "[AgentLab] 컷오버 리셋 완료\n"
            f"신규_epoch={new_epoch}\n"
            f"아카이브={bool(archive)}\n"
            f"재초기화={bool(reinit)}\n"
            f"태스크재시작={bool(restart_tasks)}",
            event="report",
        )
        return payload

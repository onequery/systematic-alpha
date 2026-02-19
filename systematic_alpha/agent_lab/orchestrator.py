from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from systematic_alpha.agent_lab.accounting import AccountingEngine
from systematic_alpha.agent_lab.agents import AgentDecisionEngine, build_default_agent_profiles, profile_from_agent_row
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
    def _sanitized_agent_constraints() -> Dict[str, Any]:
        return {
            "markets": ["KR", "US"],
            "budget_isolated": True,
            "cross_agent_budget_access": False,
            "autonomy_mode": "max_freedom_realtime",
        }

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
                constraints=self._sanitized_agent_constraints(),
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

    def propose_orders(
        self,
        market: str,
        yyyymmdd: str,
        *,
        auto_execute: Optional[bool] = None,
    ) -> Dict[str, Any]:
        market = market.strip().upper()
        signal = self.storage.get_latest_session_signal(market, yyyymmdd)
        if signal is None:
            self.ingest_session(market, yyyymmdd)
            signal = self.storage.get_latest_session_signal(market, yyyymmdd)
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
                    "date": yyyymmdd,
                    "requested_auto_execute": bool(requested_auto_execute),
                    "forced_auto_execute": True,
                },
                now_iso(),
            )

        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        output_rows: List[Dict[str, Any]] = []
        usdkrw = self._usdkrw_rate(yyyymmdd) if market == "US" else 1.0
        symbol_name_map = self._build_symbol_name_map_from_payload(signal.get("payload", {}))
        for agent in agents:
            agent_id = str(agent["agent_id"])
            profile = profile_from_agent_row(agent)
            active = self.registry.get_active_strategy(agent_id)
            active_params = dict(active.get("params", {}) or {})
            ledger = self.accounting.rebuild_agent_ledger(agent_id)
            day_ret, week_ret = self.accounting.daily_and_weekly_return(agent_id, yyyymmdd)
            position_qty_by_symbol: Dict[str, float] = {}
            for pos in list(ledger.get("positions") or []):
                pos_market = str(pos.get("market", "")).strip().upper()
                if pos_market != market:
                    continue
                symbol = str(pos.get("symbol", "")).strip().upper()
                qty = float(pos.get("quantity", 0.0) or 0.0)
                if symbol and qty > 0:
                    position_qty_by_symbol[symbol] = qty

            proposed_orders, rationale = self.agent_engine.propose_orders(
                agent=profile,
                market=market,
                session_payload=signal["payload"],
                params=active_params,
                available_cash_krw=float(ledger["cash_krw"]),
                current_positions=list(ledger.get("positions", [])),
                usdkrw_rate=usdkrw,
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
                session_date=yyyymmdd,
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
                        "session_date": yyyymmdd,
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
                    "session_date": yyyymmdd,
                    "status": status,
                    "blocked_reason": decision.blocked_reason,
                    "accepted_orders": decision.accepted_orders,
                    "dropped_orders": decision.dropped_orders,
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
                        note=f"auto-approved market={market} date={yyyymmdd}",
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
            "date": yyyymmdd,
            "auto_execute": bool(auto_execute),
            "proposals": output_rows,
            "execution_results": execution_results,
        }
        self._write_json_artifact(yyyymmdd, f"proposals_{market.lower()}_{yyyymmdd}", payload)
        self.storage.log_event("orders_proposed", payload, now_iso())
        self._append_activity_artifact(yyyymmdd, "orders_proposed", payload)

        lines = [
            "[AgentLab] 주문 제안",
            f"시장={market}",
            f"일자={yyyymmdd}",
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
        fills = self.paper_broker.execute_orders(
            proposal_id=proposal_id,
            agent_id=agent_id,
            market=market,
            orders=list(proposal["orders"]),
            fx_rate=fx_rate,
        )
        symbol_name_map = self._load_symbol_name_map(market, yyyymmdd)
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
            "fills": fills,
            "equity": ledger,
        }
        self._write_json_artifact(yyyymmdd, f"approval_{proposal_id}", out)
        self.storage.log_event("orders_approved", out, now_iso())
        self._append_activity_artifact(yyyymmdd, "orders_approved", out)
        self._notify(
            "[AgentLab] 거래 실행\n"
            f"시장={market}\n"
            f"일자={yyyymmdd}\n"
            f"에이전트={agent_id}\n"
            f"제안ID={proposal_id}\n"
            f"체결={self._orders_to_text(fills, limit=5, symbol_name_map=symbol_name_map)}"
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
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")

        day_rows: List[Dict[str, Any]] = []
        active_strategy_map: Dict[str, Dict[str, Any]] = {}
        agent_profiles: List[Any] = []
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
            md_lines.append(f"### Round {round_row.get('round')} ({round_row.get('phase')})")
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
        if summary:
            if len(summary) > 220:
                summary = summary[:220] + "..."
            msg_lines.append(f"토의요약={summary}")
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

    def weekly_council(self, week_id: str) -> Dict[str, Any]:
        week_start, week_end = parse_week_id(week_id)
        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        week_rows: List[Dict[str, Any]] = []
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
        for row in week_rows:
            aid = str(row["agent_id"])
            scored_rows.append(
                {
                    **row,
                    "score": float(scores.get(aid, 0.0)),
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

        prev_week = (week_start - timedelta(days=1)).isocalendar()
        prev_week_id = f"{prev_week.year}-W{prev_week.week:02d}"
        prev_decision = self.storage.query_one(
            "SELECT decision_json FROM weekly_councils WHERE week_id = ?",
            (prev_week_id,),
        )
        prev_scores: Dict[str, float] = {}
        prev_risk: Dict[str, int] = {}
        if prev_decision:
            try:
                decoded = json.loads(str(prev_decision.get("decision_json") or "{}"))
                if isinstance(decoded, dict):
                    prev_scores = dict(decoded.get("score_board", {}))
                    prev_rows = list(decoded.get("rows_scored", []))
                    for r in prev_rows:
                        aid = str(r.get("agent_id", ""))
                        prev_risk[aid] = int(r.get("risk_violations", 0) or 0)
            except Exception:
                prev_scores = {}
                prev_risk = {}

        for agent in agents:
            aid = str(agent["agent_id"])
            active = active_strategy_map[aid]
            row = next((x for x in scored_rows if x["agent_id"] == aid), {})
            suggested_map = discussion.get("agent_param_suggestions", {}) or {}
            params = dict(suggested_map.get(aid) or active["params"])
            current_score = float(scores.get(aid, 0.0))
            current_risk = int(row.get("risk_violations", 0) or 0) if row else 0
            prev_score = float(prev_scores.get(aid, 0.0))
            prev_risk_count = int(prev_risk.get(aid, 0) or 0)
            promote = (
                current_score >= PROMOTION_SCORE_THRESHOLD
                and prev_score >= PROMOTION_SCORE_THRESHOLD
                and current_risk == 0
                and prev_risk_count == 0
            )
            reg = self.registry.register_strategy_version(
                agent_id=aid,
                params=params,
                notes=(
                    f"weekly council update ({week_id}); "
                    f"score={current_score:.4f}, prev_score={prev_score:.4f}, "
                    f"risk={current_risk}, prev_risk={prev_risk_count}, promote={promote}"
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
            "promotion_score_threshold": PROMOTION_SCORE_THRESHOLD,
            "promotion_rule": "current_score>=threshold && prev_score>=threshold && current_week_risk_violations==0 && prev_week_risk_violations==0",
            "discussion": discussion,
            "llm_alerts": llm_alerts,
        }
        markdown = (
            f"# Weekly Council {week_id}\n\n"
            f"- Champion: `{champion}`\n"
            f"- Scores: `{scores}`\n"
            f"- Promoted Versions: `{promoted_versions}`\n"
            f"- LLM Alerts: `{len(llm_alerts)}`\n\n"
            "## Moderator Summary\n"
            f"{str((discussion.get('moderator') or {}).get('summary') or '')}\n"
        )
        stamp = week_end.strftime("%Y%m%d")
        weekly_stamp = f"{stamp}_weekly"
        self.storage.upsert_weekly_council(week_id, champion, decision, markdown, now_iso())
        self.identity.save_checkpoint(week_id, decision)
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

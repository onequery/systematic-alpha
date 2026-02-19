from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    PROPOSAL_STATUS_PENDING_APPROVAL,
    STATUS_DATA_QUALITY_LOW,
    STATUS_INVALID_SIGNAL,
    STATUS_MARKET_CLOSED,
    STATUS_SIGNAL_OK,
)
from systematic_alpha.agent_lab.storage import AgentLabStorage
from systematic_alpha.agent_lab.strategy_registry import StrategyRegistry

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

    def _notify(self, text: str) -> None:
        self.notifier.send(text)

    @staticmethod
    def _orders_to_text(orders: List[Dict[str, Any]], limit: int = 5) -> str:
        if not orders:
            return "none"
        chunks: List[str] = []
        for row in orders[:limit]:
            side = str(row.get("side", ""))
            symbol = str(row.get("symbol", ""))
            qty = float(row.get("quantity", 0.0) or 0.0)
            status = str(row.get("status", "")).strip()
            if status:
                chunks.append(f"{side} {symbol} x{qty:.0f} ({status})")
            else:
                chunks.append(f"{side} {symbol} x{qty:.0f}")
        if len(orders) > limit:
            chunks.append(f"... +{len(orders) - limit}")
        return ", ".join(chunks)

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y%m%d")

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
        if latest:
            payload = latest.get("payload", {})
            try:
                rate = float(payload.get("rate"))
                d0 = datetime.fromisoformat(str(latest["created_at"])).date()
                d1 = datetime.strptime(yyyymmdd, "%Y%m%d").date()
                if (d1 - d0).days <= 3:
                    return rate
            except Exception:
                pass
        self.storage.log_event(
            event_type="fx_rate_usdkrw",
            payload={"rate": default_rate, "source": "env_default", "date": yyyymmdd},
            created_at=now_iso(),
        )
        return default_rate

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
            "[AgentLab] init\n"
            f"capital_krw={capital_krw:.0f}\n"
            f"agents={', '.join(payload['agents'])}"
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
            "[AgentLab] ingest-session\n"
            f"market={market}\n"
            f"date={yyyymmdd}\n"
            f"status={status_code}\n"
            f"invalid_reason={invalid_reason or '-'}"
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

        requested_auto_execute = auto_execute
        if auto_execute is None:
            auto_execute = _truthy(os.getenv("AGENT_LAB_AUTO_APPROVE", "1"))
        # Agent-first repository policy: order proposals are always auto-executed.
        if not bool(auto_execute):
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
            auto_execute = True

        agents = self.storage.list_agents()
        if not agents:
            raise RuntimeError("no agents registered. run `init` first.")
        output_rows: List[Dict[str, Any]] = []
        usdkrw = self._usdkrw_rate(yyyymmdd) if market == "US" else 1.0
        for agent in agents:
            agent_id = str(agent["agent_id"])
            profile = profile_from_agent_row(agent)
            active = self.registry.get_active_strategy(agent_id)
            ledger = self.accounting.rebuild_agent_ledger(agent_id)
            day_ret, week_ret = self.accounting.daily_and_weekly_return(agent_id, yyyymmdd)

            proposed_orders, rationale = self.agent_engine.propose_orders(
                agent=profile,
                market=market,
                session_payload=signal["payload"],
                params=active["params"],
                available_cash_krw=float(ledger["cash_krw"]),
            )
            raw_orders = [o.__dict__ for o in proposed_orders]
            decision = self.risk.evaluate(
                status_code=str(signal["status_code"]),
                allocated_capital_krw=float(agent["allocated_capital_krw"]),
                day_return_pct=float(day_ret),
                week_return_pct=float(week_ret),
                current_exposure_krw=float(self.accounting.current_exposure_krw(agent_id)),
                orders=raw_orders,
            )
            status = (
                PROPOSAL_STATUS_APPROVED
                if decision.allowed and bool(auto_execute)
                else (PROPOSAL_STATUS_PENDING_APPROVAL if decision.allowed else PROPOSAL_STATUS_BLOCKED)
            )

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
            "hold_mode_pending_approval": False,
            "auto_execute": bool(auto_execute),
            "proposals": output_rows,
            "execution_results": execution_results,
        }
        self._write_json_artifact(yyyymmdd, f"proposals_{market.lower()}_{yyyymmdd}", payload)
        self.storage.log_event("orders_proposed", payload, now_iso())
        self._append_activity_artifact(yyyymmdd, "orders_proposed", payload)

        lines = [
            "[AgentLab] propose",
            f"market={market}",
            f"date={yyyymmdd}",
            f"auto_execute={bool(auto_execute)}",
        ]
        for row in output_rows:
            lines.append(
                f"{row.get('agent_id')}: {row.get('status')} | "
                f"{self._orders_to_text(list(row.get('orders') or []), limit=3)}"
            )
        if execution_results:
            ok_count = sum(1 for x in execution_results if bool(x.get("ok")))
            lines.append(f"auto_execution_result={ok_count}/{len(execution_results)} success")
        self._notify("\n".join(lines))
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
        expected = allowed_statuses or [PROPOSAL_STATUS_PENDING_APPROVAL]
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
                    "[AgentLab] trade-blocked\n"
                    f"proposal_id={proposal_id}\n"
                    f"agent={agent_id}\n"
                    "reason=fx_rate_unavailable_over_3d"
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
            "[AgentLab] trade-executed\n"
            f"market={market}\n"
            f"date={yyyymmdd}\n"
            f"agent={agent_id}\n"
            f"proposal_id={proposal_id}\n"
            f"fills={self._orders_to_text(fills, limit=5)}"
        )
        return out

    def approve_orders(self, proposal_identifier: str, approved_by: str = "manual", note: str = "") -> Dict[str, Any]:
        return self._execute_proposal(
            proposal_identifier=proposal_identifier,
            approved_by=approved_by,
            note=note,
            allowed_statuses=[PROPOSAL_STATUS_PENDING_APPROVAL],
        )

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
        lines = [f"[AgentLab] daily-review", f"date={yyyymmdd}"]
        for r in rows:
            lines.append(
                f"{r['agent_id']}: equity={float(r['equity_krw']):.0f}, "
                f"drawdown={float(r['drawdown']):.4f}, turnover={float(r['turnover']):.4f}"
            )
        self._notify("\n".join(lines))
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
        self.storage.upsert_weekly_council(week_id, champion, decision, markdown, now_iso())
        self.identity.save_checkpoint(week_id, decision)
        self._write_json_artifact(stamp, f"weekly_council_{week_id.replace('-', '_')}", decision)
        self._write_text_artifact(stamp, f"weekly_council_{week_id.replace('-', '_')}", markdown)
        self.storage.log_event("weekly_council", decision, now_iso())
        self._append_activity_artifact(stamp, "weekly_council", decision)
        lines = [
            "[AgentLab] weekly-council",
            f"week={week_id}",
            f"champion={champion}",
            f"promoted={promoted_versions if promoted_versions else '(none)'}",
        ]
        rounds = list((discussion.get("rounds") or []))
        for round_row in rounds[:2]:
            phase = str(round_row.get("phase", ""))
            lines.append(f"round={round_row.get('round')} phase={phase}")
            speeches = list(round_row.get("speeches") or [])
            for speech in speeches[:3]:
                aid = str(speech.get("agent_id", ""))
                text = str(speech.get("thesis") or speech.get("rebuttal") or "")
                if len(text) > 120:
                    text = text[:120] + "..."
                if text:
                    lines.append(f"{aid}: {text}")
        self._notify("\n".join(lines))
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
            "[AgentLab] report\n"
            f"from={date_from}\n"
            f"to={date_to}\n"
            f"rows={len(rows)}"
        )
        return payload

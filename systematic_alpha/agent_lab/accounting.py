from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from statistics import pstdev
from typing import Any, Dict, List, Tuple

from systematic_alpha.agent_lab.storage import AgentLabStorage


def _parse_dt(text: str) -> datetime:
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class AccountingEngine:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage

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

    def allocated_capital(self, agent_id: str) -> float:
        row = self.storage.query_one("SELECT allocated_capital_krw FROM agents WHERE agent_id = ?", (agent_id,))
        return float(row["allocated_capital_krw"]) if row else 0.0

    def _fills_with_orders(self, agent_id: str) -> List[Dict[str, Any]]:
        rows = self.storage.query_all(
            """
            SELECT
                pf.paper_fill_id,
                pf.fill_price,
                pf.fill_quantity,
                pf.fill_value_krw,
                pf.fx_rate,
                pf.filled_at,
                po.market,
                po.symbol,
                po.side,
                po.broker_response_json
            FROM paper_fills pf
            JOIN paper_orders po ON po.paper_order_id = pf.paper_order_id
            WHERE po.agent_id = ?
            ORDER BY pf.filled_at ASC, pf.paper_fill_id ASC
            """,
            (agent_id,),
        )
        return [
            row
            for row in rows
            if not self._broker_response_has_api_error(row.get("broker_response_json"))
        ]

    def rebuild_agent_ledger(self, agent_id: str) -> Dict[str, Any]:
        fills = self._fills_with_orders(agent_id)
        allocated = self.allocated_capital(agent_id)
        positions: Dict[Tuple[str, str], Dict[str, float]] = {}
        cash = allocated
        realized_pnl = 0.0
        trade_pnls: List[float] = []
        turnover_total = 0.0

        for row in fills:
            market = str(row["market"])
            symbol = str(row["symbol"])
            side = str(row["side"]).upper()
            qty = float(row["fill_quantity"])
            value_krw = float(row["fill_value_krw"])
            px_krw = (value_krw / qty) if qty > 0 else 0.0
            key = (market, symbol)
            state = positions.get(key, {"qty": 0.0, "avg": 0.0})

            turnover_total += abs(value_krw)
            if side == "BUY":
                total_cost = state["avg"] * state["qty"] + value_krw
                new_qty = state["qty"] + qty
                new_avg = total_cost / new_qty if new_qty > 0 else 0.0
                state = {"qty": new_qty, "avg": new_avg}
                cash -= value_krw
            elif side == "SELL":
                sell_qty = min(state["qty"], qty)
                pnl = (px_krw - state["avg"]) * sell_qty
                realized_pnl += pnl
                trade_pnls.append(pnl)
                state = {"qty": max(0.0, state["qty"] - sell_qty), "avg": state["avg"]}
                cash += value_krw
            positions[key] = state

        live_positions: List[Dict[str, Any]] = []
        invested = 0.0
        for (market, symbol), st in positions.items():
            if st["qty"] <= 0:
                continue
            market_value = st["qty"] * st["avg"]
            invested += market_value
            live_positions.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "quantity": st["qty"],
                    "avg_price": st["avg"],
                    "market_value_krw": market_value,
                    "unrealized_pnl_krw": 0.0,
                }
            )

        closed = [x for x in trade_pnls if abs(x) > 1e-9]
        wins = [x for x in closed if x > 0]
        losses = [x for x in closed if x < 0]
        win_rate = (len(wins) / len(closed)) if closed else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else (999.0 if wins else 0.0)

        max_consecutive_loss = 0
        cur_loss = 0
        for x in closed:
            if x < 0:
                cur_loss += 1
                max_consecutive_loss = max(max_consecutive_loss, cur_loss)
            else:
                cur_loss = 0

        equity = cash + invested
        turnover = turnover_total / allocated if allocated > 0 else 0.0
        return {
            "allocated_capital_krw": allocated,
            "cash_krw": cash,
            "invested_krw": invested,
            "equity_krw": equity,
            "realized_pnl_krw": realized_pnl,
            "unrealized_pnl_krw": 0.0,
            "positions": live_positions,
            "turnover": turnover,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_consecutive_loss": max_consecutive_loss,
            "trade_count": len(closed),
        }

    def _drawdown_and_volatility(self, agent_id: str, current_equity: float) -> Tuple[float, float]:
        curve = self.storage.list_equity_curve(agent_id=agent_id)
        series = [float(x["equity_krw"]) for x in curve if float(x["equity_krw"]) > 0]
        series.append(max(current_equity, 1e-6))
        peak = -math.inf
        max_dd = 0.0
        for eq in series:
            peak = max(peak, eq)
            if peak > 0:
                dd = (eq / peak) - 1.0
                max_dd = min(max_dd, dd)

        rets: List[float] = []
        for i in range(1, len(series)):
            prev = series[i - 1]
            cur = series[i]
            if prev > 0:
                rets.append((cur / prev) - 1.0)
        volatility = pstdev(rets) if len(rets) >= 2 else 0.0
        return max_dd, volatility

    def upsert_daily_snapshot(self, agent_id: str, as_of_date: str) -> Dict[str, Any]:
        ledger = self.rebuild_agent_ledger(agent_id)
        self.storage.delete_agent_positions(agent_id)
        now_ts = datetime.now().isoformat(timespec="seconds")
        for pos in ledger["positions"]:
            self.storage.upsert_position(
                agent_id=agent_id,
                market=str(pos["market"]),
                symbol=str(pos["symbol"]),
                quantity=float(pos["quantity"]),
                avg_price=float(pos["avg_price"]),
                market_value_krw=float(pos["market_value_krw"]),
                unrealized_pnl_krw=float(pos["unrealized_pnl_krw"]),
                updated_at=now_ts,
            )

        drawdown, volatility = self._drawdown_and_volatility(agent_id, float(ledger["equity_krw"]))
        self.storage.upsert_equity_snapshot(
            agent_id=agent_id,
            as_of_date=as_of_date,
            cash_krw=float(ledger["cash_krw"]),
            invested_krw=float(ledger["invested_krw"]),
            equity_krw=float(ledger["equity_krw"]),
            realized_pnl_krw=float(ledger["realized_pnl_krw"]),
            unrealized_pnl_krw=float(ledger["unrealized_pnl_krw"]),
            drawdown=float(drawdown),
            volatility=float(volatility),
            win_rate=float(ledger["win_rate"]),
            profit_factor=float(ledger["profit_factor"]),
            turnover=float(ledger["turnover"]),
            max_consecutive_loss=int(ledger["max_consecutive_loss"]),
            created_at=now_ts,
        )
        ledger["drawdown"] = drawdown
        ledger["volatility"] = volatility
        return ledger

    def current_exposure_krw(self, agent_id: str) -> float:
        rows = self.storage.list_positions(agent_id=agent_id)
        return sum(float(x["market_value_krw"]) for x in rows)

    def daily_and_weekly_return(self, agent_id: str, as_of_date: str) -> Tuple[float, float]:
        curve = self.storage.list_equity_curve(agent_id=agent_id)
        if not curve:
            return 0.0, 0.0
        curve = sorted(curve, key=lambda r: str(r["as_of_date"]))
        current = next((r for r in reversed(curve) if r["as_of_date"] == as_of_date), None)
        if current is None:
            current = curve[-1]
        current_eq = float(current["equity_krw"])

        prev_day = None
        target_day = _parse_dt(as_of_date) - timedelta(days=1)
        for row in reversed(curve):
            d = _parse_dt(str(row["as_of_date"]))
            if d <= target_day:
                prev_day = row
                break
        day_ret = 0.0
        if prev_day and float(prev_day["equity_krw"]) > 0:
            day_ret = (current_eq / float(prev_day["equity_krw"])) - 1.0

        prev_week = None
        target_week = _parse_dt(as_of_date) - timedelta(days=7)
        for row in reversed(curve):
            d = _parse_dt(str(row["as_of_date"]))
            if d <= target_week:
                prev_week = row
                break
        week_ret = 0.0
        if prev_week and float(prev_week["equity_krw"]) > 0:
            week_ret = (current_eq / float(prev_week["equity_krw"])) - 1.0
        return day_ret, week_ret

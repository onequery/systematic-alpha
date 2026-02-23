from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List

from systematic_alpha.agent_lab.schemas import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_BLOCKED,
    RiskDecision,
    STATUS_DATA_QUALITY_LOW,
    STATUS_INVALID_SIGNAL,
    STATUS_MARKET_CLOSED,
)


class RiskEngine:
    def __init__(
        self,
        *,
        day_loss_limit: float = -0.02,
        week_loss_limit: float = -0.05,
        position_cap_ratio: float = 0.333,
        exposure_cap_ratio: float = 0.95,
    ):
        self.day_loss_limit = float(day_loss_limit)
        self.week_loss_limit = float(week_loss_limit)
        self.position_cap_ratio = float(position_cap_ratio)
        self.exposure_cap_ratio = float(exposure_cap_ratio)

    @staticmethod
    def blocked_by_signal(status_code: str) -> bool:
        return status_code in {STATUS_MARKET_CLOSED, STATUS_INVALID_SIGNAL, STATUS_DATA_QUALITY_LOW}

    @staticmethod
    def _truthy(value: str) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    def evaluate(
        self,
        *,
        status_code: str,
        allocated_capital_krw: float,
        available_cash_krw: float,
        day_return_pct: float,
        week_return_pct: float,
        current_exposure_krw: float,
        orders: List[Dict[str, Any]],
        usdkrw_rate: float = 1300.0,
        position_cap_ratio: float | None = None,
        exposure_cap_ratio: float | None = None,
        day_loss_limit: float | None = None,
        week_loss_limit: float | None = None,
        position_qty_by_symbol: Dict[str, float] | None = None,
    ) -> RiskDecision:
        dropped: List[Dict[str, Any]] = []
        if self.blocked_by_signal(status_code):
            return RiskDecision(
                allowed=False,
                blocked_reason=f"blocked_by_status:{status_code}",
                accepted_orders=[],
                dropped_orders=[],
                metrics={},
            )
        day_limit = float(self.day_loss_limit if day_loss_limit is None else day_loss_limit)
        week_limit = float(self.week_loss_limit if week_loss_limit is None else week_loss_limit)
        pos_cap_ratio = float(self.position_cap_ratio if position_cap_ratio is None else position_cap_ratio)
        exp_cap_ratio = float(self.exposure_cap_ratio if exposure_cap_ratio is None else exposure_cap_ratio)
        max_freedom = self._truthy(os.getenv("AGENT_LAB_MAX_FREEDOM", "1"))
        if max_freedom:
            day_limit = -1.0
            week_limit = -1.0
            pos_cap_ratio = max(pos_cap_ratio, 10.0)
            exp_cap_ratio = max(exp_cap_ratio, 10.0)

        if day_limit > -0.999999 and day_return_pct <= day_limit:
            return RiskDecision(
                allowed=False,
                blocked_reason=f"day_loss_limit:{day_return_pct:.4f}<={day_limit:.4f}",
                accepted_orders=[],
                dropped_orders=[],
                metrics={},
            )
        if week_limit > -0.999999 and week_return_pct <= week_limit:
            return RiskDecision(
                allowed=False,
                blocked_reason=f"week_loss_limit:{week_return_pct:.4f}<={week_limit:.4f}",
                accepted_orders=[],
                dropped_orders=[],
                metrics={},
            )

        max_position = allocated_capital_krw * pos_cap_ratio
        max_exposure = allocated_capital_krw * exp_cap_ratio
        exposure_after = float(current_exposure_krw)
        cash_after = max(0.0, float(available_cash_krw or 0.0))
        accepted: List[Dict[str, Any]] = []
        seen_symbols = set()
        position_qty = {str(k).strip().upper(): float(v) for k, v in (position_qty_by_symbol or {}).items()}
        fx_rate = float(usdkrw_rate or 1300.0)
        if fx_rate <= 0:
            fx_rate = 1300.0

        for raw in orders:
            symbol = str(raw.get("symbol", "")).strip().upper()
            side = str(raw.get("side", "BUY")).upper()
            qty = float(raw.get("quantity", 0.0) or 0.0)
            ref = float(raw.get("reference_price", 0.0) or 0.0)
            order_market = str(raw.get("market", "")).strip().upper()
            local_notional = qty * ref
            notional = local_notional * (fx_rate if order_market == "US" else 1.0)
            if qty <= 0 or ref <= 0:
                dropped.append({"order": raw, "reason": "invalid_order_fields"})
                continue
            if symbol in seen_symbols:
                dropped.append({"order": raw, "reason": "duplicate_symbol"})
                continue
            if side == "SELL":
                held_qty = float(position_qty.get(symbol, 0.0) or 0.0)
                if held_qty <= 0:
                    dropped.append({"order": raw, "reason": "no_position_to_sell"})
                    continue
                sell_qty = min(held_qty, qty)
                if sell_qty <= 0:
                    dropped.append({"order": raw, "reason": "sell_qty_zero"})
                    continue
                accepted_order = dict(raw)
                accepted_order["quantity"] = sell_qty
                accepted.append(accepted_order)
                seen_symbols.add(symbol)
                position_qty[symbol] = max(0.0, held_qty - sell_qty)
                realized_notional = sell_qty * ref * (fx_rate if order_market == "US" else 1.0)
                exposure_after = max(0.0, exposure_after - realized_notional)
                cash_after += realized_notional
                continue
            if side != "BUY":
                dropped.append({"order": raw, "reason": "invalid_side"})
                continue
            if notional > max_position:
                dropped.append({"order": raw, "reason": "position_cap"})
                continue
            if exposure_after + notional > max_exposure:
                dropped.append({"order": raw, "reason": "exposure_cap"})
                continue
            if notional > cash_after:
                dropped.append({"order": raw, "reason": "cash_cap"})
                continue
            accepted.append(raw)
            seen_symbols.add(symbol)
            exposure_after += notional
            cash_after -= notional

        allowed = len(accepted) > 0
        blocked_reason = "" if allowed else "all_orders_filtered_by_risk"
        return RiskDecision(
            allowed=allowed,
            blocked_reason=blocked_reason,
            accepted_orders=accepted,
            dropped_orders=dropped,
            metrics={
                "allocated_capital_krw": allocated_capital_krw,
                "max_position_krw": max_position,
                "max_exposure_krw": max_exposure,
                "available_cash_krw_before": available_cash_krw,
                "available_cash_krw_after": cash_after,
                "exposure_before_krw": current_exposure_krw,
                "exposure_after_krw": exposure_after,
                "day_loss_limit": day_limit,
                "week_loss_limit": week_limit,
                "position_cap_ratio": pos_cap_ratio,
                "exposure_cap_ratio": exp_cap_ratio,
                "max_freedom_mode": max_freedom,
            },
        )

    @staticmethod
    def proposal_status_from_decision(decision: RiskDecision) -> str:
        return PROPOSAL_STATUS_APPROVED if decision.allowed else PROPOSAL_STATUS_BLOCKED

    @staticmethod
    def serialize_decision(decision: RiskDecision) -> Dict[str, Any]:
        return asdict(decision)

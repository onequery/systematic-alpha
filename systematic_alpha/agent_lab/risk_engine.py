from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from systematic_alpha.agent_lab.schemas import (
    PROPOSAL_STATUS_BLOCKED,
    PROPOSAL_STATUS_PENDING_APPROVAL,
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

    def evaluate(
        self,
        *,
        status_code: str,
        allocated_capital_krw: float,
        day_return_pct: float,
        week_return_pct: float,
        current_exposure_krw: float,
        orders: List[Dict[str, Any]],
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
        if day_return_pct <= self.day_loss_limit:
            return RiskDecision(
                allowed=False,
                blocked_reason=f"day_loss_limit:{day_return_pct:.4f}<={self.day_loss_limit:.4f}",
                accepted_orders=[],
                dropped_orders=[],
                metrics={},
            )
        if week_return_pct <= self.week_loss_limit:
            return RiskDecision(
                allowed=False,
                blocked_reason=f"week_loss_limit:{week_return_pct:.4f}<={self.week_loss_limit:.4f}",
                accepted_orders=[],
                dropped_orders=[],
                metrics={},
            )

        max_position = allocated_capital_krw * self.position_cap_ratio
        max_exposure = allocated_capital_krw * self.exposure_cap_ratio
        exposure_after = float(current_exposure_krw)
        accepted: List[Dict[str, Any]] = []
        seen_symbols = set()

        for raw in orders:
            symbol = str(raw.get("symbol", "")).strip().upper()
            side = str(raw.get("side", "BUY")).upper()
            qty = float(raw.get("quantity", 0.0) or 0.0)
            ref = float(raw.get("reference_price", 0.0) or 0.0)
            notional = qty * ref
            if side != "BUY" or qty <= 0 or ref <= 0:
                dropped.append({"order": raw, "reason": "invalid_order_fields"})
                continue
            if symbol in seen_symbols:
                dropped.append({"order": raw, "reason": "duplicate_symbol"})
                continue
            if notional > max_position:
                dropped.append({"order": raw, "reason": "position_cap"})
                continue
            if exposure_after + notional > max_exposure:
                dropped.append({"order": raw, "reason": "exposure_cap"})
                continue
            accepted.append(raw)
            seen_symbols.add(symbol)
            exposure_after += notional

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
                "exposure_before_krw": current_exposure_krw,
                "exposure_after_krw": exposure_after,
            },
        )

    @staticmethod
    def proposal_status_from_decision(decision: RiskDecision) -> str:
        return PROPOSAL_STATUS_PENDING_APPROVAL if decision.allowed else PROPOSAL_STATUS_BLOCKED

    @staticmethod
    def serialize_decision(decision: RiskDecision) -> Dict[str, Any]:
        return asdict(decision)

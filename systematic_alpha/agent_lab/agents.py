from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.schemas import (
    AgentProfile,
    ORDER_SIDE_BUY,
    ORDER_TYPE_MARKET,
    ProposedOrder,
)
from systematic_alpha.agent_lab.strategy_registry import ALLOWED_PARAM_RANGES, StrategyRegistry


def build_default_agent_profiles(total_capital_krw: float, count: int = 3) -> List[AgentProfile]:
    size = max(1, min(3, int(count)))
    per_agent = float(total_capital_krw) / float(size)
    base: List[Tuple[str, str, str, str, str]] = [
        (
            "agent_a",
            "Agent-A",
            "모멘텀/수급 강화형",
            "초기 모멘텀과 체결 강도를 중시한다. 상위 신호를 빠르게 추종한다.",
            "aggressive",
        ),
        (
            "agent_b",
            "Agent-B",
            "리스크 우선/보수형",
            "조건 충족 안정성을 우선하며 과도한 변동성과 저품질 신호를 회피한다.",
            "conservative",
        ),
        (
            "agent_c",
            "Agent-C",
            "반대가설 탐색/다양성 확보형",
            "상위 신호에 편중되지 않도록 대체 가설을 선택해 포트폴리오 다양성을 확보한다.",
            "diversifier",
        ),
    ]
    profiles: List[AgentProfile] = []
    for idx in range(size):
        agent_id, name, role, philosophy, risk = base[idx]
        profiles.append(
            AgentProfile(
                agent_id=agent_id,
                name=name,
                role=role,
                philosophy=philosophy,
                allocated_capital_krw=per_agent,
                risk_style=risk,
                constraints={
                    "markets": ["KR", "US"],
                    "max_daily_picks": 3,
                    "allowed_params": [
                        "min_change_pct",
                        "min_gap_pct",
                        "min_strength",
                        "min_vol_ratio",
                        "min_bid_ask_ratio",
                        "min_pass_conditions",
                        "min_maintain_ratio",
                        "collect_seconds",
                    ],
                },
            )
        )
    return profiles


def profile_from_agent_row(row: Dict[str, Any]) -> AgentProfile:
    constraints = row.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    return AgentProfile(
        agent_id=str(row.get("agent_id", "")).strip(),
        name=str(row.get("name", "")).strip(),
        role=str(row.get("role", "")).strip(),
        philosophy=str(row.get("philosophy", "")).strip(),
        allocated_capital_krw=float(row.get("allocated_capital_krw", 0.0) or 0.0),
        risk_style=str(row.get("risk_style", "")).strip(),
        constraints=constraints,
    )


def _extract_ranked_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("all_ranked") or payload.get("final") or []
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        price = (
            metrics.get("latest_price")
            or metrics.get("current_price")
            or row.get("entry_price")
            or row.get("current_price")
            or 0.0
        )
        try:
            price_f = float(price)
        except Exception:
            price_f = 0.0
        if price_f <= 0:
            continue
        out.append(
            {
                "rank": int(row.get("rank", idx)),
                "code": str(row.get("code", "")).strip().upper(),
                "name": str(row.get("name", "")).strip(),
                "recommendation_score": float(row.get("recommendation_score", 0.0) or 0.0),
                "price": price_f,
            }
        )
    out.sort(key=lambda x: (x["rank"], -x["recommendation_score"]))
    return out


class AgentDecisionEngine:
    def __init__(self, llm: LLMClient, strategy_registry: StrategyRegistry):
        self.llm = llm
        self.registry = strategy_registry

    def _select_candidates(self, agent_id: str, candidates: List[Dict[str, Any]], max_picks: int) -> List[Dict[str, Any]]:
        if agent_id == "agent_a":
            picked = candidates[:max_picks]
        elif agent_id == "agent_b":
            stable = [x for x in candidates if x["recommendation_score"] >= 65.0]
            picked = (stable if stable else candidates)[:max_picks]
        else:
            idxs = [0, 2, 4, 1, 3]
            picked = []
            for i in idxs:
                if i < len(candidates):
                    picked.append(candidates[i])
                if len(picked) >= max_picks:
                    break
        return picked

    def _risk_budget_ratio(self, agent_id: str) -> float:
        if agent_id == "agent_a":
            return 0.95
        if agent_id == "agent_b":
            return 0.75
        return 0.85

    def propose_orders(
        self,
        *,
        agent: AgentProfile,
        market: str,
        session_payload: Dict[str, Any],
        params: Dict[str, Any],
        available_cash_krw: float,
    ) -> Tuple[List[ProposedOrder], str]:
        candidates = _extract_ranked_candidates(session_payload)
        if not candidates:
            return [], "No ranked candidates from session signal payload."

        max_picks = int(params.get("max_daily_picks", 3) or 3)
        max_picks = max(1, min(3, max_picks))
        picked = self._select_candidates(agent.agent_id, candidates, max_picks=max_picks)
        if not picked:
            return [], "No candidate passed agent-specific picker."

        budget_cap = available_cash_krw * self._risk_budget_ratio(agent.agent_id)
        per_order_budget = budget_cap / max(1, len(picked))
        orders: List[ProposedOrder] = []
        for c in picked:
            price = float(c["price"])
            qty = int(math.floor(per_order_budget / price))
            if qty <= 0:
                continue
            orders.append(
                ProposedOrder(
                    market=market,
                    symbol=c["code"],
                    side=ORDER_SIDE_BUY,
                    order_type=ORDER_TYPE_MARKET,
                    quantity=float(qty),
                    limit_price=None,
                    reference_price=price,
                    signal_rank=int(c["rank"]),
                    recommendation_score=float(c["recommendation_score"]),
                    rationale=(
                        f"{agent.name} selected rank={c['rank']} with score={c['recommendation_score']:.2f} "
                        f"under role={agent.role}."
                    ),
                )
            )

        rationale = (
            f"{agent.name} proposed {len(orders)} order(s). "
            f"available_cash_krw={available_cash_krw:.0f}, budget_cap={budget_cap:.0f}, "
            f"per_order_budget={per_order_budget:.0f}"
        )
        return orders, rationale

    def suggest_weekly_params(
        self,
        *,
        agent: AgentProfile,
        active_params: Dict[str, Any],
        last_week_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        fallback = {"params": active_params, "notes": "fallback:no-llm-or-error"}
        system_prompt = (
            "You are a quantitative trading research assistant. "
            "Return JSON with fields: params (object), notes (string). "
            "Only adjust allowed keys modestly."
        )
        user_prompt = (
            f"Agent role={agent.role}, risk_style={agent.risk_style}\n"
            f"Active params={active_params}\n"
            f"Last week metrics={last_week_metrics}\n"
            "Adjust parameters carefully for better risk-adjusted returns."
        )
        response = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.2,
        )
        result = response.get("result", {}) if isinstance(response, dict) else fallback
        params = result.get("params", active_params) if isinstance(result, dict) else active_params
        if not isinstance(params, dict):
            params = active_params
        merged = dict(active_params)
        for key, val in params.items():
            if key in ALLOWED_PARAM_RANGES:
                merged[key] = val
        return self.registry.clamp_params(merged)

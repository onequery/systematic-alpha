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
            "momentum_and_flow",
            "Prioritize early momentum and execution strength. Follow strong signals quickly.",
            "aggressive",
        ),
        (
            "agent_b",
            "Agent-B",
            "risk_first_conservative",
            "Prioritize stability and avoid unstable high-volatility setups.",
            "conservative",
        ),
        (
            "agent_c",
            "Agent-C",
            "counter_hypothesis_diversifier",
            "Avoid crowding. Select alternatives for diversification and robustness.",
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


def _short_text(text: str, limit: int = 220) -> str:
    t = str(text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "..."


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

    def _normalize_param_changes(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in ALLOWED_PARAM_RANGES:
                continue
            out[key] = value
        return out

    def _debate_payload(self, response: Dict[str, Any], fallback: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        mode = str(response.get("mode", "fallback") or "fallback")
        reason = str(response.get("reason", "") or "")
        result = response.get("result", fallback)
        if not isinstance(result, dict):
            result = dict(fallback)
        return result, {"mode": mode, "reason": reason}

    def run_weekly_council_debate(
        self,
        *,
        agent_profiles: List[AgentProfile],
        active_params_map: Dict[str, Dict[str, Any]],
        scored_rows: List[Dict[str, Any]],
        score_board: Dict[str, float],
        week_id: str,
    ) -> Dict[str, Any]:
        row_by_agent = {str(row.get("agent_id", "")): row for row in scored_rows}
        llm_warnings: List[Dict[str, str]] = []
        rounds: List[Dict[str, Any]] = []

        opening_speeches: List[Dict[str, Any]] = []
        for profile in agent_profiles:
            aid = profile.agent_id
            fallback = {
                "thesis": f"{profile.name}: fallback thesis (LLM unavailable).",
                "risk_notes": ["Fallback mode active."],
                "param_changes": {},
                "confidence": 0.4,
            }
            system_prompt = (
                "You are one trader agent in a weekly quant strategy council. "
                "Return compact JSON only."
            )
            user_prompt = (
                f"week_id={week_id}\n"
                f"agent_id={aid}\n"
                f"role={profile.role}\n"
                f"risk_style={profile.risk_style}\n"
                f"score={float(score_board.get(aid, 0.0)):.6f}\n"
                f"metrics={row_by_agent.get(aid, {})}\n"
                f"active_params={active_params_map.get(aid, {})}\n"
                f"allowed_param_keys={list(ALLOWED_PARAM_RANGES.keys())}\n\n"
                "Output JSON schema:\n"
                "{"
                '"thesis": string, '
                '"risk_notes": [string], '
                '"param_changes": {key:value}, '
                '"confidence": number(0~1)'
                "}\n"
                "Do not add keys outside schema."
            )
            resp = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                temperature=0.3,
            )
            parsed, meta = self._debate_payload(resp, fallback)
            if meta["mode"] != "live":
                llm_warnings.append(
                    {"agent_id": aid, "phase": "opening", "reason": meta["reason"] or "fallback"}
                )
            speech = {
                "agent_id": aid,
                "mode": meta["mode"],
                "reason": meta["reason"],
                "thesis": _short_text(str(parsed.get("thesis", fallback["thesis"]))),
                "risk_notes": parsed.get("risk_notes", fallback["risk_notes"]),
                "param_changes": self._normalize_param_changes(parsed.get("param_changes", {})),
                "confidence": float(parsed.get("confidence", fallback["confidence"]) or fallback["confidence"]),
            }
            opening_speeches.append(speech)

        rounds.append({"round": 1, "phase": "opening", "speeches": opening_speeches})

        rebuttal_speeches: List[Dict[str, Any]] = []
        for profile in agent_profiles:
            aid = profile.agent_id
            peer_openings = [x for x in opening_speeches if x["agent_id"] != aid]
            fallback = {
                "rebuttal": f"{profile.name}: fallback rebuttal (LLM unavailable).",
                "counter_points": [],
                "param_changes": {},
            }
            system_prompt = (
                "You are one trader agent in a weekly quant strategy debate. "
                "Review peer claims and return compact JSON only."
            )
            user_prompt = (
                f"week_id={week_id}\n"
                f"agent_id={aid}\n"
                f"role={profile.role}\n"
                f"active_params={active_params_map.get(aid, {})}\n"
                f"peer_openings={peer_openings}\n"
                f"allowed_param_keys={list(ALLOWED_PARAM_RANGES.keys())}\n\n"
                "Output JSON schema:\n"
                "{"
                '"rebuttal": string, '
                '"counter_points": [string], '
                '"param_changes": {key:value}'
                "}\n"
                "Do not add keys outside schema."
            )
            resp = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                temperature=0.3,
            )
            parsed, meta = self._debate_payload(resp, fallback)
            if meta["mode"] != "live":
                llm_warnings.append(
                    {"agent_id": aid, "phase": "rebuttal", "reason": meta["reason"] or "fallback"}
                )
            rebuttal = {
                "agent_id": aid,
                "mode": meta["mode"],
                "reason": meta["reason"],
                "rebuttal": _short_text(str(parsed.get("rebuttal", fallback["rebuttal"]))),
                "counter_points": parsed.get("counter_points", fallback["counter_points"]),
                "param_changes": self._normalize_param_changes(parsed.get("param_changes", {})),
            }
            rebuttal_speeches.append(rebuttal)

        rounds.append({"round": 2, "phase": "rebuttal", "speeches": rebuttal_speeches})

        moderator_fallback = {
            "summary": "Fallback moderator summary: retain stable constraints and review risk events.",
            "consensus_actions": [],
            "risk_watch": ["LLM fallback mode during council."],
        }
        moderator_resp = self.llm.generate_json(
            system_prompt=(
                "You are the moderator of a quant trading weekly council. "
                "Return compact JSON only."
            ),
            user_prompt=(
                f"week_id={week_id}\n"
                f"score_board={score_board}\n"
                f"opening_speeches={opening_speeches}\n"
                f"rebuttal_speeches={rebuttal_speeches}\n\n"
                "Output JSON schema:\n"
                "{"
                '"summary": string, '
                '"consensus_actions": [string], '
                '"risk_watch": [string]'
                "}\n"
                "Do not add keys outside schema."
            ),
            fallback=moderator_fallback,
            temperature=0.2,
        )
        moderator_parsed, moderator_meta = self._debate_payload(moderator_resp, moderator_fallback)
        if moderator_meta["mode"] != "live":
            llm_warnings.append(
                {"agent_id": "moderator", "phase": "summary", "reason": moderator_meta["reason"] or "fallback"}
            )
        moderator = {
            "mode": moderator_meta["mode"],
            "reason": moderator_meta["reason"],
            "summary": _short_text(str(moderator_parsed.get("summary", moderator_fallback["summary"])), limit=400),
            "consensus_actions": moderator_parsed.get("consensus_actions", moderator_fallback["consensus_actions"]),
            "risk_watch": moderator_parsed.get("risk_watch", moderator_fallback["risk_watch"]),
        }

        suggested_params: Dict[str, Dict[str, Any]] = {}
        for profile in agent_profiles:
            aid = profile.agent_id
            merged = dict(active_params_map.get(aid, {}))
            open_row = next((x for x in opening_speeches if x["agent_id"] == aid), {})
            rebut_row = next((x for x in rebuttal_speeches if x["agent_id"] == aid), {})
            for key, value in (open_row.get("param_changes") or {}).items():
                if key in ALLOWED_PARAM_RANGES:
                    merged[key] = value
            for key, value in (rebut_row.get("param_changes") or {}).items():
                if key in ALLOWED_PARAM_RANGES:
                    merged[key] = value
            suggested_params[aid] = self.registry.clamp_params(merged)

        return {
            "week_id": week_id,
            "rounds": rounds,
            "moderator": moderator,
            "agent_param_suggestions": suggested_params,
            "llm_warnings": llm_warnings,
        }

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


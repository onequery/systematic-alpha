from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.schemas import (
    AgentProfile,
    ORDER_SIDE_BUY,
    ORDER_SIDE_SELL,
    ORDER_TYPE_MARKET,
    ProposedOrder,
)
from systematic_alpha.agent_lab.strategy_registry import ALLOWED_PARAM_RANGES, StrategyRegistry


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _default_cross_market_constraints(risk_style: str) -> Dict[str, Any]:
    style = str(risk_style or "").strip().lower()
    tilt_scale = {
        "aggressive": 0.22,
        "conservative": 0.12,
        "diversifier": 0.16,
    }.get(style, 0.16)
    return {
        "enabled": True,
        "base_weights": {"KR": 0.50, "US": 0.50},
        "min_weight": 0.20,
        "max_weight": 0.80,
        "signal_tilt_scale": tilt_scale,
    }


def _normalize_markets(raw: Any) -> List[str]:
    markets: List[str] = []
    for value in list(raw or []):
        token = str(value or "").strip().upper()
        if token in {"KR", "US"} and token not in markets:
            markets.append(token)
    for required in ["KR", "US"]:
        if required not in markets:
            markets.append(required)
    return markets


def normalize_agent_constraints(raw: Any, *, risk_style: str = "") -> Dict[str, Any]:
    constraints = dict(raw) if isinstance(raw, dict) else {}
    constraints["markets"] = _normalize_markets(constraints.get("markets") or ["KR", "US"])
    constraints["budget_isolated"] = bool(constraints.get("budget_isolated", True))
    constraints["cross_agent_budget_access"] = bool(constraints.get("cross_agent_budget_access", False))
    constraints["autonomy_mode"] = str(constraints.get("autonomy_mode", "max_freedom_realtime") or "max_freedom_realtime")

    default_cross_market = _default_cross_market_constraints(risk_style)
    raw_cross_market = constraints.get("cross_market")
    cross_market = dict(raw_cross_market) if isinstance(raw_cross_market, dict) else {}

    base_weights_raw = cross_market.get("base_weights")
    base_weights = dict(base_weights_raw) if isinstance(base_weights_raw, dict) else {}
    base_kr = _safe_float(base_weights.get("KR"), _safe_float(base_weights.get("kr"), default_cross_market["base_weights"]["KR"]))
    base_kr = min(0.95, max(0.05, base_kr))
    base_us = 1.0 - base_kr

    min_weight = _safe_float(cross_market.get("min_weight"), default_cross_market["min_weight"])
    max_weight = _safe_float(cross_market.get("max_weight"), default_cross_market["max_weight"])
    min_weight = min(0.95, max(0.05, min_weight))
    max_weight = min(0.95, max(min_weight, max_weight))

    cross_market["enabled"] = bool(cross_market.get("enabled", default_cross_market["enabled"]))
    cross_market["base_weights"] = {"KR": base_kr, "US": base_us}
    cross_market["min_weight"] = min_weight
    cross_market["max_weight"] = max_weight
    cross_market["signal_tilt_scale"] = min(
        0.50,
        max(0.0, _safe_float(cross_market.get("signal_tilt_scale"), default_cross_market["signal_tilt_scale"])),
    )
    constraints["cross_market"] = cross_market
    return constraints


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
                constraints=normalize_agent_constraints({}, risk_style=risk),
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
        constraints=normalize_agent_constraints(constraints, risk_style=str(row.get("risk_style", ""))),
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
        conditions = row.get("conditions") or {}
        if not isinstance(conditions, dict):
            conditions = {}
        out.append(
            {
                "rank": int(row.get("rank", idx)),
                "code": str(row.get("code", "")).strip().upper(),
                "name": str(row.get("name", "")).strip(),
                "recommendation_score": float(row.get("recommendation_score", 0.0) or 0.0),
                "price": price_f,
                "change_pct": float(metrics.get("current_change_pct", 0.0) or 0.0),
                "gap_pct": float(metrics.get("gap_pct", 0.0) or 0.0),
                "strength_avg": float(metrics.get("strength_avg", 0.0) or 0.0),
                "strength_hit_ratio": float(metrics.get("strength_hit_ratio", 0.0) or 0.0),
                "vol_ratio": float(metrics.get("volume_ratio", 0.0) or 0.0),
                "bid_ask_avg": float(metrics.get("bid_ask_avg", 0.0) or 0.0),
                "bid_ask_hit_ratio": float(metrics.get("bid_ask_hit_ratio", 0.0) or 0.0),
                "vwap": float(metrics.get("vwap", 0.0) or 0.0),
                "latest_price": float(metrics.get("latest_price", 0.0) or 0.0),
                "conditions": conditions,
            }
        )
    out.sort(key=lambda x: (x["rank"], -x["recommendation_score"]))
    return out


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _short_text(text: str, limit: int = 220) -> str:
    t = str(text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "..."


def _contains_hangul(text: Any) -> bool:
    t = str(text or "")
    return any("\uac00" <= ch <= "\ud7a3" for ch in t)


def _korean_text_or_fallback(value: Any, fallback: str, limit: int = 220) -> str:
    text = _short_text(str(value or "").strip(), limit=limit)
    if _contains_hangul(text):
        return text
    return _short_text(str(fallback or "").strip(), limit=limit)


def _korean_list_or_fallback(value: Any, fallback: List[str], limit: int = 180) -> List[str]:
    rows = value if isinstance(value, list) else []
    out: List[str] = []
    for row in rows:
        text = _short_text(str(row or "").strip(), limit=limit)
        if text and _contains_hangul(text):
            out.append(text)
    if out:
        return out
    fb: List[str] = []
    for row in list(fallback or []):
        text = _short_text(str(row or "").strip(), limit=limit)
        if text:
            fb.append(text)
    return fb


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
            picked = []
            # Diversifier: spread picks across the ranking curve.
            if len(candidates) <= max_picks:
                return list(candidates)
            step = max(1, len(candidates) // max_picks)
            for i in range(0, len(candidates), step):
                if i < len(candidates):
                    picked.append(candidates[i])
                if len(picked) >= max_picks:
                    break
        return picked

    def _risk_budget_ratio(self, agent_id: str, params: Dict[str, Any]) -> float:
        raw = params.get("risk_budget_ratio")
        if raw is not None:
            try:
                return max(0.01, float(raw))
            except Exception:
                pass
        if agent_id == "agent_a":
            return 0.95
        if agent_id == "agent_b":
            return 0.75
        return 0.85

    @staticmethod
    def _max_freedom_mode() -> bool:
        return _truthy(os.getenv("AGENT_LAB_MAX_FREEDOM", "1"))

    @staticmethod
    def _rotation_streak_threshold() -> int:
        try:
            raw = int(float(os.getenv("AGENT_LAB_ROTATION_FORCE_SELL_STREAK", "3") or 3))
        except Exception:
            raw = 3
        return max(1, min(100, raw))

    @staticmethod
    def _rotation_sell_on_missing_rank() -> bool:
        return _truthy(os.getenv("AGENT_LAB_ROTATION_SELL_ON_MISSING_RANK", "1"))

    def _rotation_streak_key(self, agent_id: str, market: str, symbol: str) -> str:
        aid = str(agent_id or "").strip().lower()
        mkt = str(market or "").strip().upper()
        sym = str(symbol or "").strip().upper()
        return f"rotation_outside_target_streak:{aid}:{mkt}:{sym}"

    def _get_rotation_streak(self, agent_id: str, market: str, symbol: str) -> int:
        key = self._rotation_streak_key(agent_id, market, symbol)
        try:
            raw = self.registry.storage.get_system_meta(key, "0")
            return max(0, int(float(raw or 0)))
        except Exception:
            return 0

    def _set_rotation_streak(self, agent_id: str, market: str, symbol: str, value: int) -> None:
        key = self._rotation_streak_key(agent_id, market, symbol)
        try:
            self.registry.storage.upsert_system_meta(
                meta_key=key,
                meta_value=str(max(0, int(value))),
                updated_at=now_iso(),
            )
        except Exception:
            pass

    @staticmethod
    def _candidate_pass_count(candidate: Dict[str, Any], params: Dict[str, Any]) -> int:
        th_change = float(params.get("min_change_pct", 0.0) or 0.0)
        th_gap = float(params.get("min_gap_pct", 0.0) or 0.0)
        th_strength = float(params.get("min_strength", 0.0) or 0.0)
        th_vol = float(params.get("min_vol_ratio", 0.0) or 0.0)
        th_bid_ask = float(params.get("min_bid_ask_ratio", 0.0) or 0.0)
        th_maintain = float(params.get("min_maintain_ratio", 0.0) or 0.0)
        cond = candidate.get("conditions") if isinstance(candidate.get("conditions"), dict) else {}

        checks = [
            float(candidate.get("change_pct", 0.0) or 0.0) >= th_change,
            float(candidate.get("gap_pct", 0.0) or 0.0) >= th_gap,
            float(candidate.get("strength_avg", 0.0) or 0.0) >= th_strength,
            float(candidate.get("vol_ratio", 0.0) or 0.0) >= th_vol,
            float(candidate.get("bid_ask_avg", 0.0) or 0.0) >= th_bid_ask,
            float(candidate.get("strength_hit_ratio", 0.0) or 0.0) >= th_maintain,
            float(candidate.get("bid_ask_hit_ratio", 0.0) or 0.0) >= th_maintain,
            bool(cond.get("price_above_vwap", True)),
            bool(cond.get("low_not_broken", True)),
        ]
        return sum(1 for x in checks if bool(x))

    def _filter_candidates(self, candidates: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._max_freedom_mode():
            return list(candidates)
        min_pass = int(params.get("min_pass_conditions", 1) or 1)
        min_pass = max(1, min(9, min_pass))
        filtered = [x for x in candidates if self._candidate_pass_count(x, params) >= min_pass]
        return filtered if filtered else list(candidates)

    def propose_orders(
        self,
        *,
        agent: AgentProfile,
        market: str,
        session_payload: Dict[str, Any],
        params: Dict[str, Any],
        available_cash_krw: float,
        current_positions: Optional[List[Dict[str, Any]]] = None,
        usdkrw_rate: float = 1300.0,
        market_budget_cap_krw: Optional[float] = None,
        max_picks_override: Optional[int] = None,
        min_recommendation_score: Optional[float] = None,
    ) -> Tuple[List[ProposedOrder], str]:
        candidates = self._filter_candidates(_extract_ranked_candidates(session_payload), params=params)
        if not candidates:
            return [], "No ranked candidates from session signal payload."

        max_freedom = self._max_freedom_mode()
        market_norm = str(market).strip().upper()
        fx_rate = float(usdkrw_rate or 1300.0)
        if market_norm != "US":
            fx_rate = 1.0
        if fx_rate <= 0:
            fx_rate = 1300.0 if market_norm == "US" else 1.0

        if max_freedom:
            max_picks = int(float(os.getenv("AGENT_LAB_MAX_FREEDOM_PICKS", "20") or 20))
            max_picks = max(1, min(len(candidates), max_picks))
        else:
            max_picks = int(params.get("max_daily_picks", 3) or 3)
            max_picks = max(1, min(200, max_picks))
        if max_picks_override is not None:
            try:
                override = int(float(max_picks_override))
            except Exception:
                override = max_picks
            max_picks = max(1, min(len(candidates), override))
        picked = self._select_candidates(agent.agent_id, candidates, max_picks=max_picks)
        if not picked:
            return [], "No candidate passed agent-specific picker."

        held_market_positions = [
            row
            for row in list(current_positions or [])
            if str(row.get("market", "")).strip().upper() == market_norm
        ]
        held_symbols = {str(row.get("symbol", "")).strip().upper(): row for row in held_market_positions}
        target_symbols = {str(row.get("code", "")).strip().upper() for row in picked}
        ranked_by_code = {str(row.get("code", "")).strip().upper(): row for row in candidates}

        allow_rotation_sell = (not max_freedom) or _truthy(os.getenv("AGENT_LAB_ENABLE_ROTATION_SELL", "0"))
        rotation_drop_rank = int(float(os.getenv("AGENT_LAB_ROTATION_DROP_RANK", str(max(10, max_picks * 2))) or max(10, max_picks * 2)))
        rotation_drop_score = float(os.getenv("AGENT_LAB_ROTATION_DROP_SCORE", "45") or 45.0)
        rotation_force_streak = self._rotation_streak_threshold()
        rotation_sell_missing_rank = self._rotation_sell_on_missing_rank()
        allow_add_to_held = max_freedom or _truthy(os.getenv("AGENT_LAB_ALLOW_ADD_TO_HELD", "0"))
        expand_pool = max_freedom or _truthy(os.getenv("AGENT_LAB_EXPAND_BUY_POOL", "1"))

        est_sell_krw = sum(float(row.get("market_value_krw", 0.0) or 0.0) for sym, row in held_symbols.items() if sym not in target_symbols)
        base_cash = max(0.0, float(available_cash_krw))
        budget_ratio = self._risk_budget_ratio(agent.agent_id, params)
        tradable_cash_krw = base_cash + max(0.0, est_sell_krw)
        if max_freedom:
            budget_cap = tradable_cash_krw
        else:
            budget_cap = tradable_cash_krw * min(1.0, max(0.01, budget_ratio))
        applied_market_budget_cap = None
        if market_budget_cap_krw is not None:
            try:
                cap_val = max(0.0, float(market_budget_cap_krw))
            except Exception:
                cap_val = None
            if cap_val is not None:
                applied_market_budget_cap = cap_val
                budget_cap = min(budget_cap, cap_val)
        per_order_budget = budget_cap / max(1, max_picks)
        orders: List[ProposedOrder] = []
        sell_symbols: set[str] = set()

        # Track how long each held symbol stays outside current target set.
        outside_streak_by_symbol: Dict[str, int] = {}
        for sym in held_symbols.keys():
            if not sym:
                continue
            if sym in target_symbols:
                self._set_rotation_streak(agent.agent_id, market_norm, sym, 0)
                outside_streak_by_symbol[sym] = 0
            else:
                nxt = self._get_rotation_streak(agent.agent_id, market_norm, sym) + 1
                self._set_rotation_streak(agent.agent_id, market_norm, sym, nxt)
                outside_streak_by_symbol[sym] = nxt

        # Rotate out stale symbols.
        for sym, row in held_symbols.items():
            if sym in target_symbols:
                continue
            if not allow_rotation_sell:
                continue
            rank_row = ranked_by_code.get(sym)
            outside_streak = int(outside_streak_by_symbol.get(sym, 0) or 0)
            force_reason = ""
            if max_freedom:
                should_sell = False
                if rank_row is None and rotation_sell_missing_rank:
                    should_sell = True
                    force_reason = "missing_rank"
                if outside_streak >= rotation_force_streak:
                    should_sell = True
                    if not force_reason:
                        force_reason = f"outside_target_streak={outside_streak}"
                if rank_row:
                    rank_val = int(rank_row.get("rank", 999999) or 999999)
                    score_val = float(rank_row.get("recommendation_score", 0.0) or 0.0)
                    if rank_val >= rotation_drop_rank or score_val <= rotation_drop_score:
                        should_sell = True
                        if not force_reason:
                            force_reason = f"rank={rank_val},score={score_val:.2f}"
                if not should_sell:
                    continue
            qty = float(row.get("quantity", 0.0) or 0.0)
            ref_price_krw = float(row.get("avg_price", 0.0) or 0.0)
            if qty <= 0 or ref_price_krw <= 0:
                continue
            ref_price_local = ref_price_krw if market_norm != "US" else (ref_price_krw / fx_rate)
            rotation_score = 100.0 if force_reason else 60.0
            reason_text = "outside current target set."
            if force_reason:
                reason_text = f"outside current target set ({force_reason})."
            orders.append(
                ProposedOrder(
                    market=market,
                    symbol=sym,
                    side=ORDER_SIDE_SELL,
                    order_type=ORDER_TYPE_MARKET,
                    quantity=float(qty),
                    limit_price=None,
                    reference_price=float(ref_price_local),
                    signal_rank=0,
                    recommendation_score=float(rotation_score),
                    rationale=(
                        f"{agent.name} rotation sell: symbol={sym} is {reason_text}"
                    ),
                )
            )
            sell_symbols.add(sym)

        buy_pool = list(picked)
        if expand_pool:
            seen_codes = {str(x.get("code", "")).strip().upper() for x in buy_pool}
            for c in candidates:
                code = str(c.get("code", "")).strip().upper()
                if not code or code in seen_codes:
                    continue
                buy_pool.append(c)
                seen_codes.add(code)
        if min_recommendation_score is not None:
            score_floor = float(min_recommendation_score)
            filtered_pool = [x for x in buy_pool if float(x.get("recommendation_score", 0.0) or 0.0) >= score_floor]
            if filtered_pool:
                buy_pool = filtered_pool

        remaining_budget = max(0.0, float(budget_cap))
        remaining_slots = max(1, int(max_picks))
        for c in buy_pool:
            code = str(c.get("code", "")).strip().upper()
            if not code:
                continue
            if code in sell_symbols:
                continue
            if code in held_symbols and not allow_add_to_held:
                continue
            price = float(c["price"])  # KR: KRW price, US: USD price
            price_krw = price * fx_rate
            if price_krw <= 0:
                continue
            if remaining_budget <= 0:
                break

            dynamic_budget = remaining_budget / max(1, remaining_slots)
            qty = int(math.floor(dynamic_budget / price_krw))
            if qty <= 0 and remaining_budget >= price_krw:
                # Avoid zero-qty due to equal-weight sizing when at least 1 share is affordable.
                qty = 1
            if qty <= 0:
                continue
            notional_krw = qty * price_krw
            if notional_krw > remaining_budget:
                qty = int(math.floor(remaining_budget / price_krw))
                if qty <= 0:
                    continue
                notional_krw = qty * price_krw
            orders.append(
                ProposedOrder(
                    market=market,
                    symbol=code,
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
            remaining_budget = max(0.0, remaining_budget - notional_krw)
            remaining_slots = max(0, remaining_slots - 1)
            if remaining_slots <= 0:
                break

        rationale = (
            f"{agent.name} proposed {len(orders)} order(s). "
            f"available_cash_krw={available_cash_krw:.0f}, budget_cap={budget_cap:.0f}, "
            f"per_order_budget={per_order_budget:.0f}, fx_rate={fx_rate:.2f}, "
            f"held_positions={len(held_market_positions)}, remaining_budget={remaining_budget:.0f}, "
            f"max_freedom={max_freedom}, allow_rotation_sell={allow_rotation_sell}, "
            f"allow_add_to_held={allow_add_to_held}, expand_pool={expand_pool}, "
            f"market_budget_cap_krw={applied_market_budget_cap}, "
            f"max_picks={max_picks}, min_recommendation_score={min_recommendation_score}"
        )
        return orders, rationale

    def _normalize_param_changes(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        if self._max_freedom_mode():
            return dict(raw)
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
        operator_directives_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        row_by_agent = {str(row.get("agent_id", "")): row for row in scored_rows}
        directives_map = operator_directives_map or {}
        llm_warnings: List[Dict[str, str]] = []
        rounds: List[Dict[str, Any]] = []

        opening_speeches: List[Dict[str, Any]] = []
        for profile in agent_profiles:
            aid = profile.agent_id
            fallback = {
                "thesis": f"{profile.name}: LLM 미가용으로 기본 전략 관점에서 보수적으로 유지한다.",
                "risk_notes": ["LLM fallback 모드가 활성화됨."],
                "param_changes": {},
                "confidence": 0.4,
            }
            system_prompt = (
                "너는 주간 트레이딩 전략 회의에 참여하는 에이전트다. "
                "반드시 간결한 JSON만 반환하라."
            )
            user_prompt = (
                f"week_id={week_id}\n"
                f"agent_id={aid}\n"
                f"role={profile.role}\n"
                f"risk_style={profile.risk_style}\n"
                f"score={float(score_board.get(aid, 0.0)):.6f}\n"
                f"metrics={row_by_agent.get(aid, {})}\n"
                f"active_params={active_params_map.get(aid, {})}\n"
                f"operator_directives={directives_map.get(aid, [])}\n"
                f"allowed_param_keys={list(ALLOWED_PARAM_RANGES.keys())}\n\n"
                "반환 JSON 스키마:\n"
                "{"
                '"thesis": string(한국어), '
                '"risk_notes": [string(한국어)], '
                '"param_changes": {key:value}, '
                '"confidence": number(0~1)'
                "}\n"
                "스키마 외 키는 추가하지 마라. 설명 문장은 반드시 한국어로 작성하라."
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
                "thesis": _korean_text_or_fallback(parsed.get("thesis", fallback["thesis"]), fallback["thesis"]),
                "risk_notes": _korean_list_or_fallback(parsed.get("risk_notes", fallback["risk_notes"]), fallback["risk_notes"]),
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
                "rebuttal": f"{profile.name}: LLM 미가용으로 동료 주장 대비 리스크 관점 반박을 보류한다.",
                "counter_points": [],
                "param_changes": {},
            }
            system_prompt = (
                "너는 주간 트레이딩 전략 토론의 에이전트다. "
                "동료 주장을 검토하고 간결한 JSON만 반환하라."
            )
            user_prompt = (
                f"week_id={week_id}\n"
                f"agent_id={aid}\n"
                f"role={profile.role}\n"
                f"active_params={active_params_map.get(aid, {})}\n"
                f"peer_openings={peer_openings}\n"
                f"allowed_param_keys={list(ALLOWED_PARAM_RANGES.keys())}\n\n"
                "반환 JSON 스키마:\n"
                "{"
                '"rebuttal": string(한국어), '
                '"counter_points": [string(한국어)], '
                '"param_changes": {key:value}'
                "}\n"
                "스키마 외 키는 추가하지 마라. 설명 문장은 반드시 한국어로 작성하라."
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
                "rebuttal": _korean_text_or_fallback(parsed.get("rebuttal", fallback["rebuttal"]), fallback["rebuttal"]),
                "counter_points": _korean_list_or_fallback(parsed.get("counter_points", fallback["counter_points"]), fallback["counter_points"]),
                "param_changes": self._normalize_param_changes(parsed.get("param_changes", {})),
            }
            rebuttal_speeches.append(rebuttal)

        rounds.append({"round": 2, "phase": "rebuttal", "speeches": rebuttal_speeches})

        moderator_fallback = {
            "summary": "LLM 미가용으로 이번 주 전략은 유지하되 리스크 이벤트와 회전율을 우선 점검한다.",
            "consensus_actions": [],
            "risk_watch": ["LLM fallback 모드로 요약 품질이 제한됨."],
        }
        moderator_resp = self.llm.generate_json(
            system_prompt=(
                "너는 정량 트레이딩 주간 회의의 사회자다. "
                "간결한 JSON만 반환하라."
            ),
            user_prompt=(
                f"week_id={week_id}\n"
                f"score_board={score_board}\n"
                f"opening_speeches={opening_speeches}\n"
                f"rebuttal_speeches={rebuttal_speeches}\n\n"
                "반환 JSON 스키마:\n"
                "{"
                '"summary": string(한국어), '
                '"consensus_actions": [string(한국어)], '
                '"risk_watch": [string(한국어)]'
                "}\n"
                "스키마 외 키는 추가하지 마라. 설명 문장은 반드시 한국어로 작성하라."
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
            "summary": _korean_text_or_fallback(
                moderator_parsed.get("summary", moderator_fallback["summary"]),
                moderator_fallback["summary"],
                limit=400,
            ),
            "consensus_actions": _korean_list_or_fallback(
                moderator_parsed.get("consensus_actions", moderator_fallback["consensus_actions"]),
                moderator_fallback["consensus_actions"],
                limit=220,
            ),
            "risk_watch": _korean_list_or_fallback(
                moderator_parsed.get("risk_watch", moderator_fallback["risk_watch"]),
                moderator_fallback["risk_watch"],
                limit=220,
            ),
        }

        suggested_params: Dict[str, Dict[str, Any]] = {}
        for profile in agent_profiles:
            aid = profile.agent_id
            merged = dict(active_params_map.get(aid, {}))
            open_row = next((x for x in opening_speeches if x["agent_id"] == aid), {})
            rebut_row = next((x for x in rebuttal_speeches if x["agent_id"] == aid), {})
            for key, value in (open_row.get("param_changes") or {}).items():
                if self._max_freedom_mode() or key in ALLOWED_PARAM_RANGES:
                    merged[key] = value
            for key, value in (rebut_row.get("param_changes") or {}).items():
                if self._max_freedom_mode() or key in ALLOWED_PARAM_RANGES:
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
            if self._max_freedom_mode() or key in ALLOWED_PARAM_RANGES:
                merged[key] = val
        return self.registry.clamp_params(merged)

from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import Any, Dict, List

from systematic_alpha.agent_lab.storage import AgentLabStorage


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


DEFAULT_STRATEGY_PARAMS: Dict[str, Any] = {
    "min_change_pct": 0.0,
    "min_gap_pct": 0.0,
    "min_strength": 0.0,
    "min_vol_ratio": 0.0,
    "min_bid_ask_ratio": 0.0,
    "min_pass_conditions": 1,
    "min_maintain_ratio": 0.0,
    "collect_seconds": 60,
    "intraday_monitor_enabled": 1,
    "intraday_monitor_interval_sec": 30,
    "max_daily_picks": 20,
    "position_cap_ratio": 1.0,
    "exposure_cap_ratio": 1.0,
    "day_loss_limit": -1.0,
    "week_loss_limit": -1.0,
    "risk_budget_ratio": 1.0,
}


ALLOWED_PARAM_RANGES: Dict[str, List[float]] = {
    "min_change_pct": [-30.0, 30.0],
    "min_gap_pct": [-30.0, 30.0],
    "min_strength": [0.0, 500.0],
    "min_vol_ratio": [0.0, 5.0],
    "min_bid_ask_ratio": [0.0, 10.0],
    "min_pass_conditions": [1.0, 8.0],
    "min_maintain_ratio": [0.0, 1.0],
    "collect_seconds": [10.0, 1800.0],
    "intraday_monitor_enabled": [0.0, 1.0],
    "intraday_monitor_interval_sec": [10.0, 3600.0],
    "max_daily_picks": [1.0, 200.0],
    "position_cap_ratio": [0.01, 2.0],
    "exposure_cap_ratio": [0.01, 3.0],
    "day_loss_limit": [-1.0, -0.0001],
    "week_loss_limit": [-1.0, -0.0001],
    "risk_budget_ratio": [0.01, 2.0],
}


class StrategyRegistry:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage

    @staticmethod
    def _truthy(value: str) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def clamp_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        if cls._truthy(os.getenv("AGENT_LAB_MAX_FREEDOM", "1")):
            return copy.deepcopy(params)
        out = copy.deepcopy(params)
        for key, bounds in ALLOWED_PARAM_RANGES.items():
            if key not in out:
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            val = float(out[key])
            if key in {
                "min_pass_conditions",
                "collect_seconds",
                "intraday_monitor_interval_sec",
                "intraday_monitor_enabled",
                "max_daily_picks",
            }:
                out[key] = int(max(lo, min(hi, val)))
            else:
                out[key] = max(lo, min(hi, val))
        return out

    def initialize_default_versions(self, agent_ids: List[str]) -> None:
        for agent_id in agent_ids:
            active = self.storage.get_active_strategy(agent_id)
            if active is not None:
                continue
            tag = "v1.0.0"
            self.storage.insert_strategy_version(
                agent_id=agent_id,
                version_tag=tag,
                params=copy.deepcopy(DEFAULT_STRATEGY_PARAMS),
                promoted=True,
                notes="bootstrap default strategy",
                created_at=now_iso(),
            )

    def get_active_strategy(self, agent_id: str) -> Dict[str, Any]:
        row = self.storage.get_active_strategy(agent_id)
        if row is None:
            raise RuntimeError(f"active strategy not found for agent={agent_id}")
        return row

    def register_strategy_version(
        self,
        agent_id: str,
        params: Dict[str, Any],
        notes: str,
        promote: bool,
    ) -> Dict[str, Any]:
        active = self.storage.get_active_strategy(agent_id)
        if active is None:
            base_version = "v1.0.0"
            major, minor, patch = 1, 0, 0
        else:
            base_version = str(active["version_tag"])
            try:
                major, minor, patch = [int(x) for x in base_version.strip("v").split(".")]
            except Exception:
                major, minor, patch = 1, 0, 0
        patch += 1
        tag = f"v{major}.{minor}.{patch}"
        clamped = self.clamp_params(params)
        strategy_version_id = self.storage.insert_strategy_version(
            agent_id=agent_id,
            version_tag=tag,
            params=clamped,
            promoted=promote,
            notes=notes,
            created_at=now_iso(),
        )
        return {
            "strategy_version_id": strategy_version_id,
            "version_tag": tag,
            "params": clamped,
            "promoted": promote,
            "notes": notes,
        }

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List

from systematic_alpha.agent_lab.storage import AgentLabStorage


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


DEFAULT_STRATEGY_PARAMS: Dict[str, Any] = {
    "min_change_pct": 3.0,
    "min_gap_pct": 2.0,
    "min_strength": 100.0,
    "min_vol_ratio": 0.10,
    "min_bid_ask_ratio": 1.2,
    "min_pass_conditions": 5,
    "min_maintain_ratio": 0.6,
    "collect_seconds": 600,
    "max_daily_picks": 3,
    "position_cap_ratio": 0.333,
    "exposure_cap_ratio": 0.95,
}


ALLOWED_PARAM_RANGES: Dict[str, List[float]] = {
    "min_change_pct": [1.0, 6.0],
    "min_gap_pct": [0.5, 4.0],
    "min_strength": [90.0, 160.0],
    "min_vol_ratio": [0.05, 0.4],
    "min_bid_ask_ratio": [1.0, 2.0],
    "min_pass_conditions": [4.0, 8.0],
    "min_maintain_ratio": [0.4, 0.9],
    "collect_seconds": [120.0, 900.0],
    "position_cap_ratio": [0.20, 0.40],
    "exposure_cap_ratio": [0.70, 0.95],
}


class StrategyRegistry:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage

    @staticmethod
    def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(params)
        for key, bounds in ALLOWED_PARAM_RANGES.items():
            if key not in out:
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            val = float(out[key])
            if key in {"min_pass_conditions", "collect_seconds"}:
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

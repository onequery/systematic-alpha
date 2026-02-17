from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StrategyConfig:
    market: str
    api_key: str
    api_secret: str
    acc_no: str
    user_id: Optional[str]
    mock: bool
    us_exchange: str
    us_poll_interval: float
    universe_file: Optional[str]
    max_symbols_scan: int
    pre_candidates: int
    final_picks: int
    collect_seconds: int
    rest_sleep_sec: float
    min_change_pct: float
    min_gap_pct: float
    min_prev_turnover: float
    min_strength: float
    min_vol_ratio: float
    min_bid_ask_ratio: float
    min_pass_conditions: int
    min_maintain_ratio: float
    min_strength_samples: int
    min_bid_ask_samples: int
    long_only: bool
    min_exec_ticks: int
    min_orderbook_ticks: int
    min_realtime_cum_volume: float
    min_realtime_coverage_ratio: float
    invalidate_on_low_coverage: bool
    stage1_log_interval: int
    realtime_log_interval: int
    overnight_report_path: Optional[str]
    output_json_path: Optional[str]


@dataclass
class PrevDayStats:
    prev_close: float
    prev_volume: float
    prev_turnover: float
    prev_day_change_pct: Optional[float]


@dataclass
class Stage1Candidate:
    code: str
    name: str
    current_price: float
    open_price: float
    current_change_pct: float
    gap_pct: float
    prev_close: float
    prev_day_volume: float
    prev_day_turnover: float


@dataclass
class RealtimeStats:
    got_execution: bool = False
    got_orderbook: bool = False
    execution_ticks: int = 0
    orderbook_ticks: int = 0
    strength_values: List[float] = field(default_factory=list)
    bid_ask_ratios: List[float] = field(default_factory=list)
    cum_trade_volume: float = 0.0
    cum_trade_value: float = 0.0
    latest_price: Optional[float] = None
    latest_acml_volume: Optional[float] = None
    first_reported_low: Optional[float] = None
    low_broken_after_start: bool = False


@dataclass
class FinalSelection:
    code: str
    name: str
    score: int
    max_score: int
    passed: bool
    conditions: Dict[str, bool]
    metrics: Dict[str, Optional[float]]


@dataclass
class RealtimeQuality:
    realtime_ready: bool
    quality_ok: bool
    coverage_ratio: float
    eligible_count: int
    total_count: int
    min_exec_ticks: int
    min_orderbook_ticks: int
    min_realtime_cum_volume: float

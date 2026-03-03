from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from systematic_alpha.credentials import load_credentials
from systematic_alpha.models import StrategyConfig
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector_kr import DayTradingSelector
from systematic_alpha.selector_us import USDayTradingSelector
from systematic_alpha.trader.config import TraderConfig


def build_strategy_config(
    *,
    cfg: TraderConfig,
    market: str,
    universe_file: Optional[str] = None,
    output_json_path: Optional[str] = None,
    analytics_dir: Optional[str] = None,
) -> StrategyConfig:
    liquidity_top_n = 20
    key, secret, acc_no, user_id = load_credentials(None)
    market_upper = str(market or "").upper()
    exchange = "NYSE"
    if cfg.us_sync_exchanges:
        exchange = str(cfg.us_sync_exchanges[0]).upper()
    return StrategyConfig(
        market=market_upper,
        api_key=key,
        api_secret=secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=cfg.use_mock,
        us_exchange=exchange,
        us_poll_interval=max(1.0, float(cfg.fallback_poll_seconds)),
        # Kept for selector internals; objective universe is now fixed top-N by liquidity.
        kr_universe_size=liquidity_top_n,
        us_universe_size=liquidity_top_n,
        universe_file=universe_file,
        pre_candidates=max(
            int(cfg.candidates_max_kr if market_upper == "KR" else cfg.candidates_max_us),
            20,
        ),
        final_picks=max(
            int(cfg.candidates_max_kr if market_upper == "KR" else cfg.candidates_max_us),
            1,
        ),
        collect_seconds=30,
        rest_sleep_sec=float(cfg.rest_sleep_sec),
        # Stage1 directional gates for long-only breakout flow.
        min_change_pct=0.0,
        min_gap_pct=0.0,
        min_prev_turnover=0.0,
        min_strength=0.0,
        min_vol_ratio=0.0,
        min_bid_ask_ratio=0.0,
        min_pass_conditions=1,
        min_maintain_ratio=0.0,
        min_strength_samples=1,
        min_bid_ask_samples=1,
        # Enforce positive direction at Stage1: change>=0 and gap>=0.
        long_only=True,
        min_exec_ticks=1,
        min_orderbook_ticks=1,
        min_realtime_cum_volume=0.0,
        min_realtime_coverage_ratio=0.0,
        invalidate_on_low_coverage=False,
        stage1_log_interval=20,
        realtime_log_interval=10,
        overnight_report_path=None,
        output_json_path=output_json_path,
        analytics_dir=analytics_dir,
        enable_analytics_log=False,
        test_assume_open=False,
        skip_overnight_report_update=True,
        skip_overnight_report_append=True,
    )


def make_selector(
    *,
    cfg: TraderConfig,
    market: str,
    universe_file: Optional[str] = None,
    output_json_path: Optional[str] = None,
    analytics_dir: Optional[str] = None,
):
    market_upper = str(market or "").upper()
    strategy = build_strategy_config(
        cfg=cfg,
        market=market_upper,
        universe_file=universe_file,
        output_json_path=output_json_path,
        analytics_dir=analytics_dir,
    )
    mojito = import_mojito_module()
    if market_upper == "KR":
        return DayTradingSelector(mojito, strategy)
    return USDayTradingSelector(mojito, strategy)


def as_dict(strategy: StrategyConfig) -> Dict[str, object]:
    return asdict(strategy)

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systematic_alpha.credentials import load_credentials
from systematic_alpha.dotenv import load_dotenv
from systematic_alpha.models import StrategyConfig
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector import DayTradingSelector


def build_prefetch_config(
    *,
    api_key: str,
    api_secret: str,
    acc_no: str,
    user_id: str | None,
    kr_universe_size: int,
    max_symbols_scan: int,
) -> StrategyConfig:
    return StrategyConfig(
        market="KR",
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=False,
        us_exchange="NASD",
        us_poll_interval=2.0,
        kr_universe_size=max(50, kr_universe_size),
        us_universe_size=500,
        universe_file=None,
        max_symbols_scan=max(50, max_symbols_scan),
        pre_candidates=40,
        final_picks=3,
        collect_seconds=0,
        rest_sleep_sec=0.03,
        min_change_pct=3.0,
        min_gap_pct=2.0,
        min_prev_turnover=10_000_000_000,
        min_strength=100.0,
        min_vol_ratio=0.10,
        min_bid_ask_ratio=1.2,
        min_pass_conditions=5,
        min_maintain_ratio=0.6,
        min_strength_samples=3,
        min_bid_ask_samples=3,
        long_only=True,
        min_exec_ticks=30,
        min_orderbook_ticks=30,
        min_realtime_cum_volume=1.0,
        min_realtime_coverage_ratio=0.8,
        invalidate_on_low_coverage=True,
        stage1_log_interval=20,
        realtime_log_interval=10,
        overnight_report_path=None,
        output_json_path=None,
        analytics_dir=None,
        enable_analytics_log=False,
        test_assume_open=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch KR objective universe cache (prev-day turnover rank)."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--key-file", type=str, default=None)
    parser.add_argument("--kr-universe-size", type=int, default=500)
    parser.add_argument("--max-symbols-scan", type=int, default=500)
    parser.add_argument("--force-refresh", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"[prefetch-kr] project root not found: {project_root}", flush=True)
        return 1

    os.chdir(project_root)
    load_dotenv(".env", override=False)

    api_key, api_secret, acc_no, user_id = load_credentials(args.key_file)
    config = build_prefetch_config(
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        kr_universe_size=args.kr_universe_size,
        max_symbols_scan=args.max_symbols_scan,
    )
    mojito_module = import_mojito_module()
    selector = DayTradingSelector(mojito_module, config)
    cache_path = selector._liquidity_cache_path()

    if args.force_refresh and cache_path.exists():
        cache_path.unlink(missing_ok=True)
        print(f"[prefetch-kr] removed existing cache: {cache_path}", flush=True)

    print(
        f"[prefetch-kr] start: kr_universe_size={config.kr_universe_size}, "
        f"max_symbols_scan={config.max_symbols_scan}, cache={cache_path}",
        flush=True,
    )
    codes, _ = selector.load_universe()
    if not codes:
        print("[prefetch-kr] failed: no symbols loaded", flush=True)
        return 2
    if not cache_path.exists():
        print(f"[prefetch-kr] failed: cache not created ({cache_path})", flush=True)
        return 3

    prev_success, prev_total = selector.prefetch_prev_day_stats(codes, force_refresh=args.force_refresh)
    prev_cache = selector._prev_stats_cache_path()
    print(
        f"[prefetch-kr] prev-day cache: success={prev_success}/{prev_total}, cache={prev_cache}",
        flush=True,
    )

    print(
        f"[prefetch-kr] success: selected={len(codes)}, cache={cache_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

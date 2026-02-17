from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from systematic_alpha.cli import run
from systematic_alpha.credentials import load_credentials
from systematic_alpha.dotenv import load_dotenv
from systematic_alpha.models import StrategyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KIS intraday selector for KR/US markets (REST + WebSocket/polling, mojito2)."
    )
    parser.add_argument(
        "--market",
        choices=["kr", "us"],
        default="kr",
        help="Target market: kr or us.",
    )
    parser.add_argument("--key-file", type=str, default=None, help="Path to KIS key file.")
    parser.add_argument(
        "--universe-file",
        type=str,
        default=None,
        help="Optional text/csv file containing symbols (KR 6-digit codes or US tickers).",
    )
    parser.add_argument(
        "--us-universe-file",
        type=str,
        default=None,
        help="Optional text/csv file containing US symbols (used when --market us, override objective pool).",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="NASD",
        help="US exchange for KIS overseas routing (NASD/NYSE/AMEX).",
    )
    parser.add_argument(
        "--us-poll-interval",
        type=float,
        default=2.0,
        help="US realtime polling interval in seconds (REST polling mode).",
    )
    parser.add_argument(
        "--collect-seconds",
        type=int,
        default=600,
        help="Realtime collection duration in seconds.",
    )
    parser.add_argument(
        "--max-symbols-scan",
        type=int,
        default=500,
        help="Maximum symbol count to scan in stage1.",
    )
    parser.add_argument(
        "--kr-universe-size",
        type=int,
        default=500,
        help="Objective KR universe size before stage1 scan (liquidity-ranked).",
    )
    parser.add_argument(
        "--us-universe-size",
        type=int,
        default=500,
        help="Objective US universe size before stage1 scan (S&P 500 based).",
    )
    parser.add_argument(
        "--pre-candidates",
        type=int,
        default=40,
        help="Max candidate count after stage1.",
    )
    parser.add_argument(
        "--final-picks",
        type=int,
        default=3,
        help="Number of final picks.",
    )
    parser.add_argument(
        "--min-change-pct",
        type=float,
        default=3.0,
        help="Current-day change %% threshold. In long-only mode, change must be >= threshold.",
    )
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=2.0,
        help="Opening gap %% threshold. In long-only mode, gap must be >= threshold.",
    )
    parser.add_argument(
        "--min-prev-turnover",
        type=float,
        default=10_000_000_000,
        help="Previous day turnover threshold (quote-currency). Default is 10,000,000,000.",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=100.0,
        help="Execution strength threshold.",
    )
    parser.add_argument(
        "--min-vol-ratio",
        type=float,
        default=0.10,
        help="Intraday volume ratio threshold (today volume / prev day volume).",
    )
    parser.add_argument(
        "--min-bid-ask-ratio",
        type=float,
        default=1.2,
        help="Bid/ask remaining quantity ratio threshold.",
    )
    parser.add_argument(
        "--min-pass-conditions",
        type=int,
        default=5,
        help="Pass cut for 8-condition realtime mode.",
    )
    parser.add_argument(
        "--min-maintain-ratio",
        type=float,
        default=0.6,
        help="Required hit ratio for 'maintained' conditions.",
    )
    parser.add_argument(
        "--rest-sleep",
        type=float,
        default=0.03,
        help="Small delay between REST calls to reduce burst rate.",
    )
    parser.add_argument(
        "--long-only",
        dest="long_only",
        action="store_true",
        default=True,
        help="Use long-only directional filters (change/gap must be positive and above thresholds).",
    )
    parser.add_argument(
        "--allow-short-bias",
        dest="long_only",
        action="store_false",
        help="Use absolute-value directional-neutral filters (legacy behavior).",
    )
    parser.add_argument(
        "--min-exec-ticks",
        type=int,
        default=30,
        help="Minimum execution tick samples per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-orderbook-ticks",
        type=int,
        default=30,
        help="Minimum orderbook tick samples per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-realtime-cum-volume",
        type=float,
        default=1.0,
        help="Minimum cumulative realtime trade volume per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-realtime-coverage-ratio",
        type=float,
        default=0.8,
        help="Minimum eligible symbol ratio required to validate realtime signal quality.",
    )
    parser.add_argument(
        "--invalidate-on-low-coverage",
        dest="invalidate_on_low_coverage",
        action="store_true",
        default=True,
        help="If realtime coverage is too low, invalidate today's signal and emit no picks.",
    )
    parser.add_argument(
        "--allow-low-coverage",
        dest="invalidate_on_low_coverage",
        action="store_false",
        help="Do not invalidate signal when realtime coverage is low.",
    )
    parser.add_argument(
        "--stage1-log-interval",
        type=int,
        default=20,
        help="Print stage1 scan progress every N symbols.",
    )
    parser.add_argument(
        "--realtime-log-interval",
        type=int,
        default=10,
        help="Print realtime collection heartbeat every N seconds.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use KIS mock server for REST calls.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Optional user id for websocket notice subscription.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output json file path.",
    )
    parser.add_argument(
        "--analytics-dir",
        type=str,
        default=None,
        help="Directory where analytics datasets are accumulated. Default: out/{kr|us}/YYYYMMDD/analytics",
    )
    parser.add_argument(
        "--disable-analytics-log",
        dest="enable_analytics_log",
        action="store_false",
        default=True,
        help="Disable analytics dataset accumulation.",
    )
    parser.add_argument(
        "--enable-analytics-log",
        dest="enable_analytics_log",
        action="store_true",
        help="Enable analytics dataset accumulation.",
    )
    parser.add_argument(
        "--overnight-report-path",
        type=str,
        default=None,
        help="CSV path for overnight performance tracking. Default: out/{kr|us}/YYYYMMDD/selection_overnight_report.csv",
    )
    parser.add_argument(
        "--test-assume-open",
        dest="test_assume_open",
        action="store_true",
        default=False,
        help="Testing mode: assume market-open conditions and inject realtime-like fallback when closed.",
    )
    parser.add_argument(
        "--normal-market-mode",
        dest="test_assume_open",
        action="store_false",
        help="Disable test-assume-open behavior.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StrategyConfig:
    api_key, api_secret, acc_no, file_user_id = load_credentials(args.key_file)
    user_id = args.user_id or file_user_id
    market = args.market.strip().upper()
    market_tag = market.lower()
    run_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    out_base_dir = Path("out") / market_tag / run_date
    analytics_dir = args.analytics_dir or str(out_base_dir / "analytics")
    overnight_report_path = args.overnight_report_path or str(out_base_dir / "selection_overnight_report.csv")
    universe_file = args.universe_file
    if market == "US":
        if args.us_universe_file:
            universe_file = args.us_universe_file
    return StrategyConfig(
        market=market,
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=args.mock,
        us_exchange=args.exchange,
        us_poll_interval=max(0.2, args.us_poll_interval),
        kr_universe_size=max(50, args.kr_universe_size),
        us_universe_size=max(50, args.us_universe_size),
        universe_file=universe_file,
        max_symbols_scan=args.max_symbols_scan,
        pre_candidates=args.pre_candidates,
        final_picks=args.final_picks,
        collect_seconds=args.collect_seconds,
        rest_sleep_sec=args.rest_sleep,
        min_change_pct=args.min_change_pct,
        min_gap_pct=args.min_gap_pct,
        min_prev_turnover=args.min_prev_turnover,
        min_strength=args.min_strength,
        min_vol_ratio=args.min_vol_ratio,
        min_bid_ask_ratio=args.min_bid_ask_ratio,
        min_pass_conditions=args.min_pass_conditions,
        min_maintain_ratio=args.min_maintain_ratio,
        min_strength_samples=3,
        min_bid_ask_samples=3,
        long_only=args.long_only,
        min_exec_ticks=max(1, args.min_exec_ticks),
        min_orderbook_ticks=max(1, args.min_orderbook_ticks),
        min_realtime_cum_volume=max(0.0, args.min_realtime_cum_volume),
        min_realtime_coverage_ratio=min(1.0, max(0.0, args.min_realtime_coverage_ratio)),
        invalidate_on_low_coverage=args.invalidate_on_low_coverage,
        stage1_log_interval=max(1, args.stage1_log_interval),
        realtime_log_interval=max(1, args.realtime_log_interval),
        overnight_report_path=overnight_report_path,
        output_json_path=args.output_json,
        analytics_dir=analytics_dir,
        enable_analytics_log=args.enable_analytics_log,
        test_assume_open=args.test_assume_open,
    )


def main() -> None:
    load_dotenv(".env", override=False)
    args = parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()

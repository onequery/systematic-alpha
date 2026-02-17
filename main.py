from __future__ import annotations

import argparse
import os
from pathlib import Path

from systematic_alpha.cli import run
from systematic_alpha.credentials import load_credentials
from systematic_alpha.dotenv import load_dotenv
from systematic_alpha.helpers import env_bool, env_float, env_int
from systematic_alpha.models import StrategyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KIS intraday selector for KR/US markets (REST + WebSocket/polling, mojito2)."
    )
    parser.add_argument(
        "--market",
        choices=["kr", "us"],
        default=os.getenv("MARKET", "kr").strip().lower(),
        help="Target market: kr or us.",
    )
    parser.add_argument("--key-file", type=str, default=None, help="Path to KIS key file.")
    parser.add_argument(
        "--universe-file",
        type=str,
        default=os.getenv("UNIVERSE_CODES_FILE"),
        help="Optional text/csv file containing symbols (KR 6-digit codes or US tickers).",
    )
    parser.add_argument(
        "--us-universe-file",
        type=str,
        default=os.getenv("US_UNIVERSE_FILE"),
        help="Optional text/csv file containing US symbols (used when --market us).",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=os.getenv("US_EXCHANGE", "NASD"),
        help="US exchange for KIS overseas routing (NASD/NYSE/AMEX).",
    )
    parser.add_argument(
        "--us-poll-interval",
        type=float,
        default=env_float("US_POLL_INTERVAL", 2.0),
        help="US realtime polling interval in seconds (REST polling mode).",
    )
    parser.add_argument(
        "--collect-seconds",
        type=int,
        default=env_int("COLLECT_SECONDS", 600),
        help="Realtime collection duration in seconds.",
    )
    parser.add_argument(
        "--max-symbols-scan",
        type=int,
        default=env_int("MAX_SYMBOLS_SCAN", 400),
        help="Maximum symbol count to scan in stage1.",
    )
    parser.add_argument(
        "--pre-candidates",
        type=int,
        default=env_int("PRE_CANDIDATES", 40),
        help="Max candidate count after stage1.",
    )
    parser.add_argument(
        "--final-picks",
        type=int,
        default=env_int("FINAL_PICKS", 3),
        help="Number of final picks.",
    )
    parser.add_argument(
        "--min-change-pct",
        type=float,
        default=env_float("MIN_CHANGE_PCT", 3.0),
        help="Current-day change %% threshold. In long-only mode, change must be >= threshold.",
    )
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=env_float("MIN_GAP_PCT", 2.0),
        help="Opening gap %% threshold. In long-only mode, gap must be >= threshold.",
    )
    parser.add_argument(
        "--min-prev-turnover",
        type=float,
        default=env_float("MIN_PREV_TURNOVER", 10_000_000_000),
        help="Previous day turnover threshold (quote-currency). Default is 10,000,000,000.",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=env_float("MIN_STRENGTH", 100.0),
        help="Execution strength threshold.",
    )
    parser.add_argument(
        "--min-vol-ratio",
        type=float,
        default=env_float("MIN_VOL_RATIO", 0.10),
        help="Intraday volume ratio threshold (today volume / prev day volume).",
    )
    parser.add_argument(
        "--min-bid-ask-ratio",
        type=float,
        default=env_float("MIN_BID_ASK_RATIO", 1.2),
        help="Bid/ask remaining quantity ratio threshold.",
    )
    parser.add_argument(
        "--min-pass-conditions",
        type=int,
        default=env_int("MIN_PASS_CONDITIONS", 5),
        help="Pass cut for 8-condition realtime mode.",
    )
    parser.add_argument(
        "--min-maintain-ratio",
        type=float,
        default=env_float("MIN_MAINTAIN_RATIO", 0.6),
        help="Required hit ratio for 'maintained' conditions.",
    )
    parser.add_argument(
        "--rest-sleep",
        type=float,
        default=env_float("REST_SLEEP_SEC", 0.03),
        help="Small delay between REST calls to reduce burst rate.",
    )
    parser.add_argument(
        "--long-only",
        dest="long_only",
        action="store_true",
        default=env_bool("LONG_ONLY", True),
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
        default=env_int("MIN_EXEC_TICKS", 30),
        help="Minimum execution tick samples per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-orderbook-ticks",
        type=int,
        default=env_int("MIN_ORDERBOOK_TICKS", 30),
        help="Minimum orderbook tick samples per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-realtime-cum-volume",
        type=float,
        default=env_float("MIN_REALTIME_CUM_VOLUME", 1.0),
        help="Minimum cumulative realtime trade volume per symbol for realtime quality eligibility.",
    )
    parser.add_argument(
        "--min-realtime-coverage-ratio",
        type=float,
        default=env_float("MIN_REALTIME_COVERAGE_RATIO", 0.8),
        help="Minimum eligible symbol ratio required to validate realtime signal quality.",
    )
    parser.add_argument(
        "--invalidate-on-low-coverage",
        dest="invalidate_on_low_coverage",
        action="store_true",
        default=env_bool("INVALIDATE_ON_LOW_COVERAGE", True),
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
        default=env_int("STAGE1_LOG_INTERVAL", 20),
        help="Print stage1 scan progress every N symbols.",
    )
    parser.add_argument(
        "--realtime-log-interval",
        type=int,
        default=env_int("REALTIME_LOG_INTERVAL", 10),
        help="Print realtime collection heartbeat every N seconds.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=os.getenv("KIS_MOCK", "0") == "1",
        help="Use KIS mock server for REST calls.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=os.getenv("KIS_USER_ID"),
        help="Optional user id for websocket notice subscription.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.getenv("SELECTION_OUTPUT_JSON"),
        help="Optional output json file path.",
    )
    parser.add_argument(
        "--overnight-report-path",
        type=str,
        default=os.getenv("OVERNIGHT_REPORT_PATH", "./out/selection_overnight_report.csv"),
        help="CSV path for selected-symbol overnight performance tracking report.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StrategyConfig:
    api_key, api_secret, acc_no, file_user_id = load_credentials(args.key_file)
    user_id = args.user_id or file_user_id
    market = args.market.strip().upper()
    universe_file = args.universe_file
    if market == "US":
        if args.us_universe_file:
            universe_file = args.us_universe_file
        elif not universe_file:
            universe_file = str(Path("systematic_alpha") / "data" / "us_universe_default.txt")
    return StrategyConfig(
        market=market,
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=args.mock,
        us_exchange=args.exchange,
        us_poll_interval=max(0.2, args.us_poll_interval),
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
        overnight_report_path=args.overnight_report_path,
        output_json_path=args.output_json,
    )


def main() -> None:
    load_dotenv(".env", override=False)
    args = parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
